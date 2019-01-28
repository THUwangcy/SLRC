# -*- coding: UTF-8 -*-

import os
import sys
import time
import argparse
import numpy as np

from src.common.constants import *


class Click:
    def __init__(self, info):
        self.user_name = info[0].strip()
        self.item_name = info[1].strip()
        self.user_id, self.item_id = None, None
        self.time = int(info[3])
        self.seq_order = None


class Preprocess:
    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset
        self.dataset_path = self.path + self.dataset + '.txt'
        self.n_users = 0
        self.n_items = 0
        self.n_clicks = 0
        self.user2id = dict()
        self.item2id = dict()
        self.id2user = dict()
        self.id2item = dict()

        self.clicks_per_user = list()        # only item ids purchased by users
        self.pos_per_user = list()           # total clicks per user
        self.train_data = list()             # train data
        self.validation_data = list()
        self.test_data = list()
        self.validation_candidate_data = list()
        self.test_candidate_data = list()

    def load_data(self, user_min, item_min, sample_candidate, filt=True):
        print('Loading data from \"{}\", dataset = \"{}\", userMin = {}, itemMin = {} '.format(
            self.path, self.dataset, user_min, item_min), end='')
        self._load_clicks(user_min, item_min)
        if filt:
            print('Filtering noise users')
            self._filter()
        self._assign_id()
        print('\"n_users\": {}, \"n_items\": {}, \"n_clicks\": {}'.format(self.n_users, self.n_items, self.n_clicks))
        # print('Filling repeat info', end=' ')
        # self._fill_repeat_info()
        print('Generating validation set and test set', end=' ')
        self._gen_valid_test(sample_candidate)
        # self._show_user_seq(1)
        print('Preprocess completed!')

    def _load_clicks(self, user_min, item_min):
        # First pass, count interactions for each user/item
        user_cnt, item_cnt = dict(), dict()
        total_clicks, illegal_records = 0, 0
        with open(self.dataset_path) as f:
            for n_read, line in enumerate(f):
                if n_read % 100000 == 0:
                    print('.', end='')
                    sys.stdout.flush()
                # print(line.replace("\n", ""))
                info = line.replace("\n", "").split('\t')
                try:
                    user_name, item_name, score, t = info[0].strip(), info[1].strip(), info[2], int(info[3])
                except (ValueError, IndexError):
                    illegal_records += 1
                    continue
                if user_name not in user_cnt:
                    user_cnt[user_name] = set()
                if item_name not in item_cnt:
                    item_cnt[item_name] = 0
                user_cnt[user_name].add(t)
                item_cnt[item_name] += 1
                total_clicks += 1
        print("\nFirst pass: #users = {}, #items = {}, #clicks = {} (#illegal records = {})".format(
            len(user_cnt), len(item_cnt), total_clicks, illegal_records), end=' ')

        # Second pass, ignore users and items exceeding the limit of user_min/item_min
        with open(self.dataset_path) as f2:
            for n_read, line in enumerate(f2):
                if n_read % 100000 == 0:
                    print('.', end='')
                    sys.stdout.flush()
                info = line.replace("\n", "").split('\t')
                try:
                    click = Click(info)
                except (ValueError, IndexError):
                    continue
                # Only preserve users who have more than 5 transactions(every timestamp means a transaction)
                # and items that have been bought more than 5 times
                if len(user_cnt[click.user_name]) < user_min or item_cnt[click.item_name] < item_min:
                    continue
                if click.user_name not in self.user2id:
                    self.user2id[click.user_name] = self.n_users
                    self.id2user[self.n_users] = click.user_name
                    self.pos_per_user.append([])
                    self.n_users += 1
                self.pos_per_user[self.user2id[click.user_name]].append(click)

        # Rank clicks for each user according to timestamp
        print("\nSorting clicks for each user ", end='')
        for u in range(self.n_users):
            self.pos_per_user[u].sort(key=lambda x: x.time)
            if u % 10000 == 0:
                print('.', end='')
                sys.stdout.flush()
        print()

    def _filter(self):
        # filter noise users
        purchase_freq, frequent_user = list(), list()
        day_format = '%Y%m%d'
        for clicks in self.pos_per_user:
            # purchase num per day
            purchase_freq.append(len(clicks) / np.ceil(1e-10 + ((clicks[-1].time - clicks[0].time) / 3600 / 24)))
            frequent_user.append(time.strftime(day_format, time.localtime(clicks[0].time))
                                 != time.strftime(day_format, time.localtime(clicks[-1].time)))
        frequent_user, purchase_freq = np.array(frequent_user), np.array(purchase_freq)
        freq_mean, freq_std = np.mean(purchase_freq), np.std(purchase_freq)
        freq_threshold = freq_mean + 2 * freq_std
        indice = np.logical_and(frequent_user, purchase_freq < freq_threshold)
        self.pos_per_user = np.array(self.pos_per_user)[indice]
        self.pos_per_user = [d for d in self.pos_per_user if len(d) > 0]

    def _assign_id(self):
        # assign user_id and item_id
        self.n_users = len(self.pos_per_user)
        self.n_clicks, items = 0, set()
        self.user2id, self.id2user, self.item2id, self.id2item = dict(), dict(), dict(), dict()
        for i, clicks in enumerate(self.pos_per_user):
            self.n_clicks += len(clicks)
            self.user2id[clicks[0].user_name] = i
            self.id2user[i] = clicks[0].user_name
            for j, click in enumerate(clicks):
                if click.item_name not in items:
                    self.item2id[click.item_name] = len(items)
                    self.id2item[len(items)] = click.item_name
                    items.add(click.item_name)
                clicks[j].user_id = i
                clicks[j].item_id = self.item2id[click.item_name]
                clicks[j].seq_order = j
        self.n_items = len(items)

    def _gen_valid_test(self, sample_candidate):
        for user_id, u in enumerate(self.pos_per_user):
            if user_id % 2000 == 0:
                print('.', end='')
                sys.stdout.flush()
            order_num = 1
            self.clicks_per_user.append(set())
            test_end, valid_end, valid_start = len(u), -1, -1
            cur_time = u[-1].time
            for idx, click in enumerate(u[::-1]):
                i = len(u) - 1 - idx
                if click.time != cur_time:
                    cur_time = click.time
                    order_num += 1
                    if valid_end < 0:
                        valid_end = i + 1
                    elif valid_start < 0:
                        valid_start = i + 1
                self.clicks_per_user[-1].add(click.item_id)
            if order_num >= 3:
                self.test_data.append(u[valid_end:])
                self.validation_data.append(u[valid_start:valid_end])
                self.train_data.append(u[:valid_start])

                self.validation_candidate_data.append(
                    self._gen_candidate_items(self.validation_data[-1], sample_candidate)
                )
                self.test_candidate_data.append(
                    self._gen_candidate_items(self.test_data[-1], sample_candidate)
                )
            else:
                self.train_data.append(u)
        print()

    def _gen_candidate_items(self, gt_clicks, sample_candidate):
        if sample_candidate > 0:
            gt_items = [gt.item_id for gt in gt_clicks]
            legal_pool = list(set(np.arange(self.n_items)).difference(set(gt_items)))
            neg_items = np.random.choice(legal_pool, replace=False, size=sample_candidate-len(gt_items))
            return np.concatenate((neg_items, gt_items))
        else:
            return np.array([0])


def save_split(corpus_path, dataset, corpus):
    if not os.path.exists(corpus_path + '/data_{}/'.format(dataset)):
        os.makedirs(corpus_path + '/data_{}/'.format(dataset))
    print('\nSaving dataset in ' + corpus_path + '/data_{}/'.format(dataset))
    # Book
    with open(corpus_path + '/data_{}/book.csv'.format(dataset), 'w') as f:
        f.write('user_id\tsequence (item_id,time)\n')
        for user_seq in corpus.pos_per_user:
            user_id = user_seq[0].user_id
            seq = [(click.item_id, click.time) for click in user_seq]
            target_line = '{}\t{}\n'.format(user_id, str(seq))
            f.write(target_line)
    # Train
    with open(corpus_path + '/data_{}/train.csv'.format(dataset), 'w') as f:
        f.write('user_id\tconsumption_order\n')
        for user_seq in corpus.train_data:
            for order, click in enumerate(user_seq):
                target_line = '{}\t{}\n'.format(click.user_id, order)
                f.write(target_line)
    # Dev
    with open(corpus_path + '/data_{}/dev.csv'.format(dataset), 'w') as f:
        f.write('user_id\tgt_order\tcandidate_item_id\n')
        for dev_basket, candidates in zip(corpus.validation_data, corpus.validation_candidate_data):
            user_id = dev_basket[0].user_id
            gt_items = [click.seq_order for click in dev_basket]
            target_line = '{}\t{}\t{}\n'.format(user_id, str(gt_items), str(candidates.tolist()))
            f.write(target_line)
    # Test
    with open(corpus_path + '/data_{}/test.csv'.format(dataset), 'w') as f:
        f.write('user_id\tgt_order\tcandidate_item_id\n')
        for test_basket, candidates in zip(corpus.test_data, corpus.test_candidate_data):
            user_id = test_basket[0].user_id
            gt_items = [click.seq_order for click in test_basket]
            target_line = '{}\t{}\t{}\n'.format(user_id, str(gt_items), str(candidates.tolist()))
            f.write(target_line)


if __name__ == '__main__':
    Preprocess.__module__ = "Preprocess"
    Click.__module__ = "Preprocess"

    parser = argparse.ArgumentParser(description="Preprocess.")
    parser.add_argument('--dataset', nargs='?', default='order', help='Choose a dataset.')
    args = parser.parse_args()

    corpus_path = '../data/'
    np.random.seed(2018)
    preprocessor = Preprocess(corpus_path, args.dataset)
    preprocessor.load_data(5, 5, sample_candidate=NEG_SAMPLE[args.dataset], filt=True)

    save_split(corpus_path, args.dataset, preprocessor)
