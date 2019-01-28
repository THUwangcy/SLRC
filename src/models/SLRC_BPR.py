# -*- coding: UTF-8 -*-

import tensorflow as tf

from src.common import utils
from src.models.SLRC import SLRC
from src.Corpus import *


class SLRCBPR(SLRC):

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
            self.t = tf.placeholder(shape=[None], dtype=self.d_type, name="t")  # current time
            # how many time this item has been consumed
            self.length = tf.placeholder(shape=[None], dtype=tf.int32, name="length")
            # the timestamps when consuming the same item previously
            self.history_time = tf.placeholder(shape=[None, None], dtype=self.d_type, name="history_time")
            # the following is used for prediction
            self.history_time_list = tf.placeholder(shape=[None], dtype=self.d_type, name="history_time_list")
            self.history_indice = tf.placeholder(shape=[None, 2], dtype=tf.int32, name="history_indice")
            self.candidate_indice = tf.placeholder(shape=[None, 2], dtype=tf.int32, name="candidate_indice")
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._init_weights()

            # Basic intensity (BPR)
            u_emb = tf.nn.embedding_lookup(self.weights['user_emb'], self.users)
            i_emb = tf.nn.embedding_lookup(self.weights['item_emb'], self.items)
            neg_i_emb = tf.nn.embedding_lookup(self.weights['item_emb'], self.neg_items)
            base_intensity = tf.reduce_sum(u_emb * i_emb, axis=1) + self.weights['global_bias']
            self.neg_pred_val = tf.reduce_sum(u_emb * neg_i_emb, axis=1) + self.weights['global_bias']

            # Self-excitation
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
            self.reg_loss = (tf.nn.l2_loss(u_emb) + tf.nn.l2_loss(i_emb)) / batch_size
            self.loss = -tf.reduce_mean(
                tf.log_sigmoid(self.ranking_score - self.neg_pred_val)
            ) + self.l2_regularize * self.reg_loss

            # Predict topk
            # 1. first get users' base intensity score for every item   [batch_user * total_item]
            self.item_score = tf.matmul(u_emb, self.weights['item_emb'], transpose_b=True) + self.weights['global_bias']
            # 2. the items user consumed will add an extra value on base intensity (self-excitation), named 'delta'
            #    (according to all consumed items of these users, calculate their contribution to corresponding
            #     user-item score, then construct 'delta' with scatter_nd)
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
        all_weights['user_emb'] = tf.Variable(
            tf.random_normal(shape=[self.user_num, self.emb_dim], mean=0.0, stddev=0.01),
            dtype=self.d_type,
            name="user_emb"
        )
        all_weights['item_emb'] = tf.Variable(
            tf.random_normal(shape=[self.item_num, self.emb_dim], mean=0.0, stddev=0.01),
            dtype=self.d_type,
            name="item_emb"
        )
        all_weights['global_bias'] = tf.Variable(
            self.click_num / self.user_num / self.item_num,
            dtype=self.d_type, name="global_bias"
        )

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
            self.users: [d['user_id'] for d in batch_data],
            self.history_indice: history_indice,
            self.history_time_list: np.array(history_time_list) / TIME_SCALAR,
            self.t: np.array(t) / TIME_SCALAR,
            self.candidate_indice: candidate_items_index,
            self.train_phase: False
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
            self.train_phase: True
        }
        return feed_dict
