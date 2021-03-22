# -*- coding: utf-8 -*-
# @Author: xiaodong
# @Date  : 2021/3/22

import math
from typing import Dict, List, TypeVar
from collections import Counter

userid = TypeVar("userid", int)
itemid = TypeVar("itemid", int)
User_Items = Dict[userid, List[itemid]]


class Metric(object):
    def __init__(self, recommend_user_items_dict: User_Items, real_user_items_dict: User_Items):
        self.recommend_user_items_dict = recommend_user_items_dict
        self.real_user_items_dict = real_user_items_dict

    def _measure(self, use="recall") -> float:

        assert use in ("recall", "precision")

        hit, total = 0, 0
        for user, real_items in self.real_user_items_dict.items():
            recom_items = self.real_user_items_dict.get(user, [])
            hit += len(set(recom_items) & set(real_items))
            if use == "recall":
                total += len(set(real_items))
            elif use == "precision":
                total += len(set(recom_items))
        return round(hit / total, 2) if total else 0.0

    def recall(self) -> float:
       return self._measure(use="recall")

    def precision(self) -> float:
        return self._measure(use="precision")

    def coverage(self) -> float:
        recom_item_set = set()
        real_item_set = set()
        for user_id, recom_items in self.recommend_user_items_dict.items():
            recom_item_set += set(recom_items)
            real_item_set += self.real_user_items_dict.get(user_id, set())
        return round(len(recom_item_set)/len(real_item_set) if len(real_item_set) else 0, 2)

    def popularity(self) -> float:
        pop_detail = Counter()
        for real_items in self.real_user_items_dict.values():
            pop_detail.update(set(real_items))

        appear_num, total_popularity = 0, 0
        for recom_items in self.recommend_user_items_dict.values():
            recom_item_set = set(recom_items)
            total_popularity += sum([
                math.log(pop_detail[recom_item]+1)
                for recom_item in recom_item_set
            ])
            appear_num += len(recom_item_set)
        return round(total_popularity/appear_num, 2) if appear_num else 0.0
