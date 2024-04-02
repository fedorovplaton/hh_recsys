from __future__ import annotations

import json
from abc import ABC
from collections import Counter
from itertools import chain, cycle, islice
from typing import Any, List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.contexts.RecommenderContext import RecommenderContext
from src.selectors.AbstractSelector import AbstractSelector


class I2ISelector(AbstractSelector, ABC):
    i2i = Optional[Dict]

    def __init__(self, name: str = "i2i_selector"):
        super().__init__(name)

    def to_str(self) -> str:
        return "i2i_selector"

    def train(self, data: pd.DataFrame, item_data: Dict = None) -> None:
        raise Exception("Not implemented")

    def get_candidates(self, context, user_id: str, session_id: str, n: int = 10) -> List[str]:
        raise Exception("Not implemented")

    def get_candidates_pandas_batch(self,
                                    recommender_context: RecommenderContext,
                                    data: pd.DataFrame,
                                    result_df: pd.DataFrame,
                                    n: int = 10,
                                    item_data: Dict = None) -> None:
        assert self.i2i is not None
        recommendations_column_name = f"_recos_{self.name}"

        result_df[recommendations_column_name] = data[["vacancy_id", "action_type", "items", "item_actions"]].apply(
            lambda row: self.get_i2i_recos(row["vacancy_id"],
                                           row["action_type"],
                                           row["items"],
                                           row["item_actions"],
                                           item_data,
                                           self.i2i,
                                           n),
            axis=1
        )

    def get_i2i_recos(self, vacancy_id, action_type, items, item_actions, item_data, i2i, n) -> List[str]:
        if isinstance(items, np.ndarray) or isinstance(items, list):
            vacancy_id = list(chain(*items)) + list(vacancy_id)
        if isinstance(item_actions, np.ndarray) or isinstance(item_actions, list):
            action_type = list(chain(*item_actions)) + list(action_type)
        vacancy_id = vacancy_id[::-1]
        action_type = action_type[::-1]

        """
            Сбор якорей
        """
        recommendations = []
        scored_history = map(
            lambda iva: (1 if iva[1][1] == 1 else 2 if iva[1][1] == 3 else 3, iva[0], iva[1][0]),
            enumerate(zip(vacancy_id, action_type))
        )
        sorted_filtered_scored_history = sorted(filter(lambda x: x[2] in i2i, scored_history))
        anchors = list(map(lambda x: x[2], sorted_filtered_scored_history))

        """
            Сбор списков
        """

        for anchor in anchors:
            anchor_neighbours = i2i[anchor]
            recommendations.append(list(anchor_neighbours))

        output = []
        used = set()

        for item in self.roundrobin(*recommendations):
            if item not in used:
                output.append(item)
                used.add(item)

                if len(output) == n:
                    break

        return output

    def roundrobin(self, *iterables):
        pending = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while pending:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                pending -= 1
                nexts = cycle(islice(nexts, pending))

    def save_selector_params(self, path: str) -> None:
        assert self.i2i is not None

        vacancy_ids = []
        neighbours = []

        for key in self.i2i:
            vacancy_ids.append(key)
            neighbours.append(self.i2i[key])

        pd.DataFrame \
            .from_dict({"vacancy_id": vacancy_ids, "neighbours": neighbours}) \
            .to_parquet(path)

    def load_selector_params(self, path: str) -> None:
        data = pd.read_parquet(path)
        i2i = {}

        for _, row in data.iterrows():
            i2i[row["vacancy_id"]] = row["neighbours"]

        self.i2i = i2i

    @staticmethod
    def load_selector(path: str, name: str = "i2i_selector") -> I2ISelector:
        selector = I2ISelector(name)
        selector.load_selector_params(path)

        return selector

    @staticmethod
    def get_name() -> str:
        return "I2ISelector"

    @staticmethod
    def get_file_extension() -> str:
        return "parquet"
