# -*- coding: UTF-8 -*-

import numpy as np


def pad_lst(lst, dtype=np.int64):
    inner_max_len = max(map(len, lst))
    result = np.zeros([len(lst), inner_max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result


def history_pop_rerank(data, pred, clicks_per_user, topk):
    for i in range(len(pred)):
        user_id, order_time = data[i]['user_id'], clicks_per_user[data[i]['user_id']][data[i]['gt_click_order'][0]].time
        history_items = [d.item_id for d in clicks_per_user[user_id] if d.time < order_time]
        pred[i] = np.array(sorted(pred[i][:topk], key=history_items.count, reverse=True))


def repeat_ratio_in_pred(data, pred, clicks_per_user, topk):
    repeat_case = 0
    for i in range(len(pred)):
        user_id = data[i]['user_id']
        history_items = [d.item_id for d in clicks_per_user[user_id]]
        for item in pred[i][:topk]:
            if item in history_items:
                repeat_case += 1
                break
    print(repeat_case / len(pred))
