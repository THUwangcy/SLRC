# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

from src.common import utils
from src.models.SLRC import SLRC
from src.Corpus import *


class SLRCNCF(SLRC):

    def __init__(self, cf, user_num, item_num, click_num, time_span, avg_repeat_interval, sample_candidate,
                 epoch, learning_rate, batch_size, l2_regularize, emb_dim, topk, layers, reg_layers, random_seed,
                 model_path, dropout):
        self.layers = layers
        self.reg_layers = reg_layers
        self.dropout = dropout

        SLRC.__init__(
            self, cf, user_num, item_num, click_num, time_span, avg_repeat_interval, sample_candidate,
            epoch, learning_rate, batch_size, l2_regularize, emb_dim, topk, random_seed, model_path
        )

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.debug = None
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)

            # Input
            self.users = tf.placeholder(shape=[None], dtype=tf.int32, name="u_ids")
            self.items = tf.placeholder(shape=[None], dtype=tf.int32, name="i_ids")
            self.neg_items = tf.placeholder(shape=[None], dtype=tf.int32, name="neg_i_ids")
            self.t = tf.placeholder(shape=[None], dtype=self.d_type, name="t")
            self.length = tf.placeholder(shape=[None], dtype=tf.int32, name="length")
            self.history_time = tf.placeholder(shape=[None, None], dtype=self.d_type, name="history_time")
            self.dropout_keep = tf.placeholder(dtype=self.d_type, name="dropout_keep")
            # used for prediction
            self.history_time_list = tf.placeholder(shape=[None], dtype=self.d_type, name="history_time_list")
            self.history_indice = tf.placeholder(shape=[None, 2], dtype=tf.int32, name="history_indice")
            self.candidate_indice = tf.placeholder(shape=[None, 2], dtype=tf.int32, name="candidate_indice")
            self.train_phase = tf.placeholder(dtype=tf.bool, name="train_phase")

            # Variables.
            self.weights = self._init_weights()

            # Basic intensity (NCF)
            mf_u_emb = tf.nn.embedding_lookup(self.weights['mf_user_emb'], self.users)
            mf_i_emb = tf.nn.embedding_lookup(self.weights['mf_item_emb'], self.items)
            mf_vector = mf_u_emb * mf_i_emb
            mlp_u_emb = tf.nn.embedding_lookup(self.weights['mlp_user_emb'], self.users)
            mlp_i_emb = tf.nn.embedding_lookup(self.weights['mlp_item_emb'], self.items)
            mlp_vector = tf.concat((mlp_u_emb, mlp_i_emb), axis=1)
            for idx in range(1, len(self.layers)):
                layer = tf.layers.dense(
                    mlp_vector, self.layers[idx],
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_layers[idx]),
                    activation=tf.nn.relu, name="layer%d" % idx
                )
                mlp_vector = tf.nn.dropout(layer, keep_prob=self.dropout_keep)
            predict_vector = tf.concat((mf_vector, mlp_vector), axis=1)
            pred_val = tf.layers.dense(
                predict_vector, 1, name='prediction',
                kernel_initializer=tf.random_normal_initializer(stddev=0.1)
            )
            base_intensity = tf.squeeze(pred_val)

            neg_mf_i_emb = tf.nn.embedding_lookup(self.weights['mf_item_emb'], self.neg_items)
            neg_mf_vector = mf_u_emb * neg_mf_i_emb
            neg_mlp_i_emb = tf.nn.embedding_lookup(self.weights['mlp_item_emb'], self.neg_items)
            neg_mlp_vector = tf.concat((mlp_u_emb, neg_mlp_i_emb), axis=1)
            for idx in range(1, len(self.layers)):
                layer = tf.layers.dense(
                    neg_mlp_vector, self.layers[idx],
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_layers[idx]),
                    activation=tf.nn.relu, name="layer%d" % idx, reuse=True
                )
                neg_mlp_vector = tf.nn.dropout(layer, keep_prob=self.dropout_keep)
            neg_predict_vector = tf.concat((neg_mf_vector, neg_mlp_vector), axis=1)
            neg_pred_val = tf.layers.dense(
                neg_predict_vector, 1, name='prediction',
                kernel_initializer=tf.random_normal_initializer(stddev=0.1), reuse=True
            )
            self.neg_pred_val = tf.squeeze(neg_pred_val)

            # Self-excitation (all the same)
            self.repeat_param = self._repeat_params(self.items)
            alpha, pi, beta, mu, sigma = self.repeat_param
            delta_t = tf.expand_dims(self.t, -1) - self.history_time
            delta_t = tf.clip_by_value(delta_t, clip_value_min=self.eps, clip_value_max=self.inf)  # avoid 0
            mask = tf.sequence_mask(self.length, dtype=self.d_type)
            exp_dist = tf.contrib.distributions.Exponential(1.0 / tf.expand_dims(beta, -1))
            norm_dist = tf.contrib.distributions.Normal(loc=tf.expand_dims(mu, -1), scale=tf.expand_dims(sigma, -1))
            sum_k_t = tf.reduce_sum(
                mask * self._kernel_function(delta_t, tf.expand_dims(pi, -1), exp_dist, norm_dist), axis=1
            )

            self.ranking_score = base_intensity + alpha * sum_k_t

            # Loss
            batch_size = tf.cast(tf.shape(self.users)[0], self.d_type)
            mf_u_emb = tf.nn.embedding_lookup(self.weights['mf_user_emb'], self.users)
            mf_i_emb = tf.nn.embedding_lookup(self.weights['mf_item_emb'], self.items)
            mlp_u_emb = tf.nn.embedding_lookup(self.weights['mlp_user_emb'], self.users)
            mlp_i_emb = tf.nn.embedding_lookup(self.weights['mlp_item_emb'], self.items)
            self.mf_reg_loss = (tf.nn.l2_loss(mf_u_emb) + tf.nn.l2_loss(mf_i_emb)) / batch_size
            self.mlp_reg_loss = (tf.nn.l2_loss(mlp_u_emb) + tf.nn.l2_loss(mlp_i_emb)) / batch_size
            self.loss = -tf.reduce_mean(
                tf.log_sigmoid(self.ranking_score - self.neg_pred_val)
            ) + self.l2_regularize * self.mf_reg_loss \
              + self.reg_layers[0] * self.mlp_reg_loss \
              + tf.losses.get_regularization_loss()

            # Predict topk (item_score is calculated in different methods)
            # (Here for NCF, we only calculate item scores for candidate items, and then construct
            #  [batch_user * total_items] matrix, 0 for items not in candidate list)
            test_batch_size = tf.cast(tf.shape(self.users)[0] / self.sample_candidate, tf.int32)
            row_idx = tf.reshape(
                tf.tile(tf.expand_dims(tf.range(test_batch_size), -1), [1, self.sample_candidate]), (-1, 1)
            )
            score_indice = tf.concat((row_idx, tf.expand_dims(self.items, -1)), axis=1)
            self.item_score = tf.scatter_nd(score_indice, base_intensity, [test_batch_size, self.item_num])

            # (The following are all the same, see SLRC_BPR for detailed explanation)
            history_items = self.history_indice[:, 1]
            pred_alpha, pred_pi, pred_beta, pred_mu, pred_sigma = self._repeat_params(history_items)
            pred_exp_dist = tf.contrib.distributions.Exponential(1.0 / pred_beta)
            pred_norm_dist = tf.contrib.distributions.Normal(loc=pred_mu, scale=pred_sigma)
            pred_delta_t = tf.clip_by_value(
                self.t - self.history_time_list, clip_value_min=self.eps, clip_value_max=self.inf
            )
            extra_value = pred_alpha * self._kernel_function(pred_delta_t, pred_pi, pred_exp_dist, pred_norm_dist)
            delta = tf.scatter_nd(self.history_indice, extra_value, tf.shape(self.item_score))
            self.item_score = self.item_score + delta

            if self.sample_candidate > 0:
                self.item_score = self.item_score - tf.reduce_min(self.item_score)
                candidate_mask = tf.scatter_nd(
                    self.candidate_indice,
                    tf.ones(shape=[tf.shape(self.candidate_indice)[0]], dtype=self.d_type),
                    tf.shape(self.item_score)
                )
                self.item_score = self.item_score * candidate_mask
            self.topk_info = tf.nn.top_k(self.item_score, self.max_topk)
            self.prediction = self.topk_info.indices
            self.prediction_val = self.topk_info.values

            self.var_list = list(self.weights.values())

    def _init_cf_weights(self, all_weights):
        all_weights['mf_user_emb'] = tf.Variable(
            tf.random_normal(shape=[self.user_num, self.emb_dim], mean=0.0, stddev=0.01),
            dtype=self.d_type,
            name="mf_user_emb"
        )
        all_weights['mf_item_emb'] = tf.Variable(
            tf.random_normal(shape=[self.item_num, self.emb_dim], mean=0.0, stddev=0.01),
            dtype=self.d_type,
            name="mf_item_emb"
        )
        all_weights['mlp_user_emb'] = tf.Variable(
            tf.random_normal(shape=[self.user_num, int(self.layers[0] / 2)], mean=0.0, stddev=0.01),
            dtype=self.d_type,
            name="mlp_user_emb"
        )
        all_weights['mlp_item_emb'] = tf.Variable(
            tf.random_normal(shape=[self.item_num, int(self.layers[0] / 2)], mean=0.0, stddev=0.01),
            dtype=self.d_type,
            name="mlp_item_emb"
        )
        pre_size = self.layers[0]
        for i, layer_size in enumerate(self.layers[1:]):
            all_weights['layer_%d' % i] = tf.Variable(
                tf.random_normal([pre_size, self.layers[i]], 0.0, 0.01, dtype=self.d_type),
                name='layer_%d' % i)
            all_weights['bias_%d' % i] = tf.Variable(
                tf.random_normal([1, self.layers[i]], 0.0, 0.01, dtype=self.d_type),
                name='bias_%d' % i)
            pre_size = self.layers[i]
        all_weights['prediction'] = tf.Variable(
            tf.random_normal([pre_size + self.emb_dim, 1], 0.0, 0.01, dtype=self.d_type))

    @staticmethod
    def _batch_norm_layer(x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def _get_pred_feed_dict(self, batch_data, clicks_per_user, eval_batch_size):
        user_idx = np.expand_dims(
            np.tile(np.reshape(np.arange(eval_batch_size), (-1, 1)), [1, self.sample_candidate]), -1
        )
        history_indice, history_time_list, t = [], [], []
        batch_user_idx = user_idx[:len(batch_data)]
        batch_cand_items = [d['candidates'] for d in batch_data]
        candidate_items_index = np.stack([batch_user_idx, np.expand_dims(batch_cand_items, -1)], axis=-1)
        candidate_items_index = np.reshape(candidate_items_index, (-1, 2))
        for i, d in enumerate(batch_data):
            order_time = clicks_per_user[d['user_id']][d['gt_click_order'][0]].time
            for click in clicks_per_user[d['user_id']]:
                if click.time >= order_time:
                    break
                history_indice.append([i, click.item_id])
                history_time_list.append(click.time)
                t.append(order_time)
        feed_dict = {
            self.users: np.concatenate([[d['user_id']] * self.sample_candidate for d in batch_data]),
            self.items: np.concatenate(batch_cand_items),
            self.neg_items: [],
            self.history_indice: history_indice,
            self.history_time_list: np.array(history_time_list) / TIME_SCALAR,
            self.t: np.array(t) / TIME_SCALAR,
            self.candidate_indice: candidate_items_index,
            self.train_phase: False,
            self.dropout_keep: 1.
        }
        return feed_dict

    def _get_train_feed_dict(self, batch_data, clicks_per_user, batch_neg_items):
        # previous purchase time
        unaligned_history_time = [clicks_per_user[d['user_id']][d['click_order']].repeat_info for d in batch_data]
        feed_dict = {
            self.users: [d['user_id'] for d in batch_data],
            self.items: [clicks_per_user[d['user_id']][d['click_order']].item_id for d in batch_data],
            self.t: [clicks_per_user[d['user_id']][d['click_order']].time / TIME_SCALAR for d in batch_data],
            self.length: [len(clicks_per_user[d['user_id']][d['click_order']].repeat_info) for d in batch_data],
            self.history_time: np.array(utils.pad_lst(unaligned_history_time)) / TIME_SCALAR,
            self.neg_items: batch_neg_items,
            self.train_phase: True,
            self.dropout_keep: 1. - self.dropout
        }
        return feed_dict
