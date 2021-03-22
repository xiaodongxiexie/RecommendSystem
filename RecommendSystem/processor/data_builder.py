# -*- coding: utf-8 -*-
# @Author: xiaodong
# @Date  : 2021/3/22

import pandas as pd


class Builder(object):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self._pre_check()

    def _pre_check(self):
        assert "user_id" in self.frame
        assert "movie_id" in self.frame

    def frame2dict(self, groupby: str = "user_id") -> dict:

        assert groupby in ("user_id", "movie_id")

        d = {}
        if groupby == "user_id":
            frame = self.frame.groupby(groupby)["movie_id"]
        elif groupby == "movie_id":
            frame = self.frame.groupby(groupby)["user_id"]
        (
            frame
            .apply(list)
            .reset_index()
            .apply(lambda obj: {obj[0]: obj[1]}, axis=1)
            .apply(lambda obj: d.update(obj))
        )
        return d
