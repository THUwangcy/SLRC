# -*- coding: UTF-8 -*-

from tqdm import tqdm
import tensorflow as tf
from abc import ABCMeta, abstractmethod

from src.common.evaluate import evaluate_metric
from src.Corpus import *


class SLRC:
    """
    This is a virtual class with general flow control functions of SLRC model.
    Concret subclass should implement at least 4 abstract functions according to selected CF method:
    1. _init_graph()
    2. _init_cf_weights()
    3. get_pred_feed_dict()
    4. get_train_feed_dict()
    """
    __metaclass__ = ABCMeta

    def __init__(self, cf, user_num, item_num, click_num, time_span, avg_repeat_interval, sample_candidate,
                 epoch, learning_rate, batch_size, l2_regularize, emb_dim, topk, random_seed, model_path):
        self.cf = cf     # CF method used to calculate base intensity
        self.model_path = model_path
        self.random_seed = random_seed
        self.d_type = tf.float32
        self.sample_candidate = sample_candidate if sample_candidate > 0 else self.item_num

        self.user_num, self.item_num, self.click_num = user_num, item_num, click_num
        self.avg_interval = avg_repeat_interval / TIME_SCALAR
        self.time_span = time_span / TIME_SCALAR

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_regularize = l2_regularize
        self.emb_dim = emb_dim
        self.topk = topk
        self.max_topk = np.max(topk)
        self.eps, self.inf = EPS, INF

        self._init_graph()
        self.valid_loss, self.test_loss = [], []   # ndcg @max_topk as loss
        self._init_optimizer()

    '''
    Tensorflow part
    '''
    @abstractmethod
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.debug = None
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)

            # Input (defined in subclass)

            # Variables.
            self.weights = self._init_weights()

            # Basic intensity (implemented in subclass, with different CF method)
            base_intensity = None
            self.neg_pred_val = None

            # Self-excitation (implemented in subclass, the same)
            sum_k_t = None
            self.ranking_score = base_intensity + sum_k_t

            # Loss (implemented in subclass)
            self.loss = None

            # Predict topk (implemented in subclass)
            self.prediction = None

            self.var_list = list(self.weights.values())

    def _init_weights(self):
        all_weights = dict()
        self._init_self_excitation_weights(all_weights)
        self._init_cf_weights(all_weights)
        return all_weights

    def _init_self_excitation_weights(self, all_weights):
        all_weights['global_alpha'] = tf.Variable(
            1,
            dtype=self.d_type, name="global_alpha"
        )
        all_weights['item_alpha'] = tf.Variable(
            tf.zeros(shape=[self.item_num]),
            dtype=self.d_type, name="item_alpha"
        )
        all_weights['item_pi'] = tf.Variable(
            tf.random_normal(shape=[self.item_num], mean=0.5, stddev=0.01),
            dtype=self.d_type, name="item_pi"
        )
        all_weights['item_mu'] = tf.Variable(
            tf.random_normal(shape=[self.item_num], mean=self.avg_interval, stddev=0.01),
            dtype=self.d_type, name="item_mu"
        )
        all_weights['item_beta'] = tf.Variable(
            tf.ones(shape=[self.item_num]),
            dtype=self.d_type, name="item_beta"
        )
        all_weights['item_sigma'] = tf.Variable(
            tf.ones(shape=[self.item_num]),
            dtype=self.d_type, name="item_sigma"
        )

    @abstractmethod
    def _init_cf_weights(self, all_weights):
        pass

    def _init_optimizer(self):
        with self.graph.as_default():
            # self.optimizer = tf.train.AdagradOptimizer(
            #     learning_rate=self.learning_rate,
            #     initial_accumulator_value=1e-8).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print("#params: {}".format(total_parameters))

    def _repeat_params(self, items):
        item_alpha = tf.nn.embedding_lookup(self.weights['item_alpha'], items)
        item_pi = tf.nn.embedding_lookup(self.weights['item_pi'], items)
        item_beta = tf.nn.embedding_lookup(self.weights['item_beta'], items)
        item_mu = tf.nn.embedding_lookup(self.weights['item_mu'], items)
        item_sigma = tf.nn.embedding_lookup(self.weights['item_sigma'], items)

        alpha = tf.clip_by_value(item_alpha + self.weights['global_alpha'], 0., self.inf)
        beta = tf.clip_by_value(item_beta, self.eps, self.inf)
        pi = tf.clip_by_value(item_pi, 0., 1.)
        mu = item_mu
        sigma = tf.clip_by_value(item_sigma, self.eps, self.inf)
        return alpha, pi, beta, mu, sigma

    """
    Input [batch_size, None] or [None], params should have the same rank
    """
    @staticmethod
    def _kernel_function(delta_t, pi, exp_dist, normal_dist):
        return (1.0 - pi) * exp_dist.prob(delta_t) + pi * normal_dist.prob(delta_t)

    @staticmethod
    def eva_termination(valid):
        if len(valid) > 100 and valid[-1] < valid[-2] < valid[-3] < valid[-4] < valid[-5]:
            return True
        return False

    """
    Python part
    """
    @abstractmethod
    def _get_pred_feed_dict(self, batch_data, clicks_per_user, eval_batch_size):
        pass

    def predict(self, data, clicks_per_user):
        eval_batch_size = self.batch_size * 2
        num_example = len(data)
        total_batch = int((num_example + eval_batch_size - 1) / eval_batch_size)

        predictions = []
        for batch in range(total_batch):
            batch_start = batch * eval_batch_size
            batch_data = data[batch_start: batch_start + eval_batch_size]
            feed_dict = self._get_pred_feed_dict(batch_data, clicks_per_user, eval_batch_size)
            [prediction] = self.sess.run([self.prediction], feed_dict=feed_dict)
            predictions.extend(prediction)
        return predictions

    def evaluate(self, data, clicks_per_user, topk):
        pred = self.predict(data, clicks_per_user)
        purchase = [[clicks_per_user[d['user_id']][order].item_id for order in d['gt_click_order']] for d in data]
        result = evaluate_metric(pred, purchase, topk=topk)
        return result

    @abstractmethod
    def _get_train_feed_dict(self, batch_data, clicks_per_user, batch_neg_items):
        pass

    def fit(self, data, clicks_per_user, epoch):
        num_example = len(data)
        total_batch = int((num_example + self.batch_size - 1) / self.batch_size)
        total_loss = 0
        # sample negative items that haven't been bought
        neg_items = np.random.randint(0, self.item_num, size=num_example)
        for i, d in enumerate(data):
            consumed_items = [click.item_id for click in clicks_per_user[d['user_id']]]
            while neg_items[i] in consumed_items:
                neg_items[i] = np.random.randint(0, self.item_num)

        for batch in tqdm(range(total_batch), leave=False, desc='Epoch %d' % epoch, ncols=100, mininterval=1):
            batch_start = batch * self.batch_size
            batch_data = data[batch_start: batch_start + self.batch_size]
            batch_neg_itmes = neg_items[batch_start: batch_start + self.batch_size]
            feed_dict = self._get_train_feed_dict(batch_data, clicks_per_user, batch_neg_itmes)
            [_, loss] = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            total_loss += loss
        return total_loss / total_batch

    def train(self, corpus):
        train_data = copy.deepcopy(corpus.data['train'])
        validation_data = corpus.data['dev']
        test_data = corpus.data['test']
        """
        When sample more users, the result will be a little bit better, but the time cost will be high.
        """
        sample_users = 5000
        if self.user_num > sample_users:
            valid_indice = np.random.choice(len(corpus.data['dev']), replace=False, size=sample_users)
            validation_data = np.array(corpus.data['dev'])[valid_indice]
            test_indice = np.random.choice(len(corpus.data['test']), replace=False, size=sample_users)
            test_data = np.array(corpus.data['test'])[test_indice]

        self.saver.save(self.sess, self.model_path)
        try:
            for epoch in range(self.epoch):
                t1 = time.time()
                np.random.shuffle(train_data)
                loss = self.fit(train_data, corpus.book, epoch + 1)
                t2 = time.time()

                # output validation
                valid_result = self.evaluate(validation_data, corpus.book, self.max_topk)
                test_result = self.evaluate(test_data, corpus.book, self.max_topk)
                self.valid_loss.append(valid_result['ndcg'])
                self.test_loss.append(test_result['ndcg'])
                best_valid_score = max(self.valid_loss)
                if best_valid_score == self.valid_loss[-1]:
                    self.saver.save(self.sess, self.model_path)
                print("Epoch {:<3} loss={:<.4f} [{:<.1f} s]\tvalidation=(ndcg:{:<.4f}), test=(ndcg:{:<.4f}) [{:<.1f} s]"
                      .format(epoch + 1, loss, t2 - t1, valid_result['ndcg'], test_result['ndcg'], time.time() - t2))
                sys.stdout.flush()
                if self.eva_termination(self.valid_loss):
                    print("\nEarly stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            print("Early stop manually")
            save_here = input("Exit completely without evaluation? (y/n) (default n):")
            if save_here.lower().startswith('y'):
                exit(1)
        self.saver.restore(self.sess, self.model_path)
