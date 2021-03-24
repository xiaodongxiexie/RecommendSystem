# -*- coding: utf-8 -*-
# @Author: xiaodong
# @Date  : 2021/3/24

import json
from functools import partial

from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class RecommendSystemByALS:
    _mapping = {
        "user_id": "user_id",  # 用于als名：原始列名
        "item_id": "item_id",
        "score": "score",
    }

    def __init__(self, table: DataFrame, limit: int = None, use_implicit: bool = False):
        """
        :param table: 从数据库读入表
        :param limit: 限制使用条数
        :param use_implicit: 是否使用隐式反馈
        """

        self.table = self.preprocess(table)
        self.use_implicit = use_implicit
        if limit:
            self.table = self.table.limit(limit)

    def preprocess(self, table: DataFrame) -> DataFrame:
        """
        用来对数据进行预处理~
        :param table:待输入训练数据
        :return:
        """
        for k, v in self._mapping.items():
            table = table.withColumnRenamed(v, k)

        return table

    def train(self, **kwargs):
        result = self.table

        valid = {"handleInvalid": "skip"}

        user_id_string2index = (
            StringIndexer(
                inputCol="user_id",
                outputCol="user_id",
                **valid
            )
        )
        item_id_string2index = (
            StringIndexer(
                inputCol="item_id",
                outputCol="item_id",
                **valid
            )
        )

        model = ALS(
            rank=kwargs.pop("rank") or 120,  # 20-200
            maxIter=kwargs.pop("maxIter") or 20,
            regParam=kwargs.pop("regParam") or 0.15,
            implicitPrefs=kwargs.pop("implicitPrefs") or self.use_implicit,
            coldStartStrategy="drop",
            userCol="user_id",
            itemCol="item_id",
            ratingCol="score",
        )
        pipeline = Pipeline(stages=[user_id_string2index, item_id_string2index, model])
        pipeline = pipeline.fit(result)

        return pipeline


class RecommendUsersItems(object):

    def __init__(self, table: DataFrame, limit: int = None, use_implicit: bool = False):
        self.model = RecommendSystemByALS(table, limit, use_implicit=use_implicit)
        self.preprocess()

    def preprocess(self):
        pipeline = self.model.train()

        self.user_id_converter = partial(IndexToString, labels=pipeline.stages[0].labels)
        self.item_id_converter = partial(IndexToString, labels=pipeline.stages[1].labels)
        self.model_after_train = pipeline.stages[2]

    def _recommend(self, dataframe: DataFrame, style: str = "recommendForAllUsers", unfold: bool = False) -> DataFrame:
        """
        :param dataframe: 推荐结果
        :param unfold: 是否将user-item展开，默认不展开
        :return: user推荐item详情
        """

        c = {
            "recommendForAllUsers": (["user_id", "item_id", "score"], ["user_id", "item_detail"]),
            "recommendForAllItems": (["item_id", "user_id", "score"], ["item_id", "user_detail"]),
        }

        # 将user_id及item_id转换回去
        dataframe = self.expand(dataframe, to_columns=c[style][0])
        dataframe = self.user_id_converter(inputCol="user_id", outputCol="nuser_id").transform(dataframe)
        dataframe = self.item_id_converter(inputCol="item_id", outputCol="nitem_id").transform(dataframe)
        dataframe: DataFrame = (
            dataframe
            .select("nuser_id", "nitem_id", "score")
            .withColumnRenamed("nuser_id", "user_id")
            .withColumnRenamed("nitem_id", "item_id")
        )
        dataframe = dataframe.select(*c[style][0])

        if not unfold:
            dataframe = (
                dataframe
                .rdd
                .map(lambda obj: (obj[0], (obj[1], obj[2])))
                .groupByKey()
                .mapValues(list)
                .map(lambda obj: (obj[0], json.dumps(obj[1])))
                .toDF(c[style][1])
            )
        return dataframe

    def expand(self, dataframe: DataFrame, to_columns: list = None):
        dataframe = (
            dataframe
            .rdd
            .flatMapValues(lambda obj: obj)
            .map(lambda obj: (obj[0], int(obj[1][0]), round(float(obj[1][1]), 3)))
            .toDF(to_columns)
        )
        return dataframe


class RecommendItemsForUsers(RecommendUsersItems):

    """给user推荐item"""

    def recommend(self, topk:int = 50, unfold: bool = False) -> DataFrame:
        """
        :param topk: 给每个user推荐个数
        :param unfold: 是否将user-item展开，默认不展开，同一user的推荐item一行表示
        :return: user推荐item详情
        """
        origin_recom_items_to_user = self.model_after_train.recommendForAllUsers(topk)
        return self._recommend(origin_recom_items_to_user, style="recommendForAllUsers", unfold=unfold)


class RecommendUsersForItems(RecommendUsersItems):

    """给item推荐user"""

    def recommend(self, topk: int = 50, unfold: bool = False) -> DataFrame:
        origin_recom_users_to_item = self.model_after_train.recommendForAllItems(topk)
        return self._recommend(origin_recom_users_to_item, style="recommendForAllItems", unfold=unfold)
     
