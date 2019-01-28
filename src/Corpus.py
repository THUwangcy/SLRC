# -*- coding: UTF-8 -*-

import sys
import time
import copy
import numpy as np

from src.common.constants import *


class Click:
    def __init__(self, u_id, i_id, t):
        self.user_id = u_id
        self.item_id = i_id
        self.time = int(t)
        self.repeat_info = list()  # timestamps when consume the same item before (in seq)


class Corpus:
    def __init__(self, path, dataset):
        self.prefix = path
        self.dataset = dataset
        self.dataset_path = '{}/data_{}/'.format(self.prefix, self.dataset)
        self.n_users = 0
        self.n_items = 0
        self.n_clicks = 0
        self.read_skip = 1

        # total clicks per user
        self.book = list()
        self.data = {
            'train': list(),
            'dev': list(),
            'test': list()
        }

        self.total_avg_interval = 0.
        self.min_time, self.max_time = INF, EPS

    def load_data(self):
        print('Loading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self._load_book()
        print('\nFilling repeat info', end=' ')
        self._fill_repeat_info()
        print('\nLoading train data...')
        self._load_train_data()
        print('Loading dev data...')
        self._load_dev_test_data('dev')
        print('Loading test data...')
        self._load_dev_test_data('test')
        print('\nLoad completed!')

    def _load_book(self):
        t1 = time.time()
        # first pass
        with open(self.dataset_path + '/book.csv') as f:
            for i, line in enumerate(f):
                if i < self.read_skip:
                    continue
                [user_id, seq] = line.split('\t')
                user_id, seq = int(user_id), eval(seq)
                self.n_users = max(self.n_users, user_id + 1)
                for (item_id, t) in seq:
                    self.n_items = max(self.n_items, item_id + 1)
                    self.min_time = min(self.min_time, t)
                    self.max_time = max(self.max_time, t)
                    self.n_clicks += 1
        for i in range(self.n_users):
            self.book.append(list())

        # second pass
        with open(self.dataset_path + '/book.csv') as f:
            for i, line in enumerate(f):
                if i < self.read_skip:
                    continue
                [user_id, seq] = line.split('\t')
                user_id, seq = int(user_id), eval(seq)
                for (item_id, t) in seq:
                    # record click on book
                    click = Click(user_id, item_id, t)
                    self.book[user_id].append(click)

        print('"n_users": {}, "n_items": {}, "n_clicks": {}'.format(self.n_users, self.n_items, self.n_clicks))
        print('"time span": {} - {}'.format(
            time.strftime("%Y.%m.%d", time.localtime(self.min_time)),
            time.strftime("%Y.%m.%d", time.localtime(self.max_time)))
        )
        print('Done! [{:<.2f} s]'.format(time.time() - t1))

    def _fill_repeat_info(self):
        t1 = time.time()
        repeat_num = 0
        for u in range(len(self.book)):
            if u % 2000 == 0:
                print('.', end='')
                sys.stdout.flush()
            item_prev_repeat = {}
            for i, click in enumerate(self.book[u]):
                if click.item_id in item_prev_repeat:
                    # only consider repeat purchase in different timestamp
                    end = len(item_prev_repeat[click.item_id]) - 1
                    while end >= 0 and item_prev_repeat[click.item_id][end] == click.time:
                        end -= 1
                    if end >= 0:
                        self.book[u][i].repeat_info = copy.copy(item_prev_repeat[click.item_id][:end + 1])
                        self.total_avg_interval += click.time - item_prev_repeat[click.item_id][end]
                        repeat_num += 1
                else:
                    item_prev_repeat[click.item_id] = list()
                item_prev_repeat[click.item_id].append(click.time)
        self.total_avg_interval /= repeat_num
        print('\n"repeat consumption": {} ({:<.4f})'.format(repeat_num, repeat_num / self.n_clicks))
        print('"average repeat interval": {:<.2f} d'.format(self.total_avg_interval / 3600 / 60 / 24))
        print('Done! [{:<.2f} s]'.format(time.time() - t1))

    def _load_train_data(self):
        t1 = time.time()
        with open(self.dataset_path + '/train.csv') as f:
            for i, line in enumerate(f):
                if i < self.read_skip:
                    continue
                [user_id, click_order] = line.split('\t')
                user_id, click_order = int(user_id), int(click_order)
                self.data['train'].append({
                    'user_id': user_id,
                    'click_order': click_order
                })
        print('Done! [{:<.2f} s]'.format(time.time() - t1))

    def _load_dev_test_data(self, data_type):
        t1 = time.time()
        with open(self.dataset_path + '/{}.csv'.format(data_type)) as f:
            for i, line in enumerate(f):
                if i < self.read_skip:
                    continue
                [user_id, gt_click_orders, candidates] = line.split('\t')
                user_id = int(user_id)
                gt_click_orders = list(map(int, gt_click_orders.strip()[1:-1].split(',')))
                candidates = list(map(int, candidates.strip()[1:-1].split(',')))

                self.data[data_type].append({
                    'user_id': user_id,
                    'gt_click_order': gt_click_orders,
                    'candidates': candidates
                })
        print('Done! [{:<.2f} s]'.format(time.time() - t1))


if __name__ == '__main__':
    Corpus.__module__ = "Corpus"
    Click.__module__ = "Corpus"

    dataset = 'order'
    corpus_path = '../data/'
    np.random.seed(2018)
    corpus = Corpus(corpus_path, dataset)
    corpus.load_data()
