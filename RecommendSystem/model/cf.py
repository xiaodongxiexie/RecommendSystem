# -*- coding: utf-8 -*-
# @Author: xiaodong
# @Date  : 2021/3/22

import numpy as np
import pandas as pd


class CF(object):

    user: str = "user_id"
    item: str = "item_id"
    rating: str = "rating"

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self._precheck()

    def build(self) -> np.ndarray:
        user_num = len(getattr(self.frame, self.user).unique())
        item_num = len(getattr(self.frame, self.item).unique())
        matrix = np.zeros(shape=(user_num, item_num), dtype=np.float32)

        for user_id, item_id, rating in self.frame[[self.user, self.item, self.rating]].values:
            matrix[user_id][item_id] = rating
        return matrix

    def _precheck(self):
        assert self.user in self.frame.columns
        assert self.item in self.frame.columns
        assert self.rating in self.frame.columns
