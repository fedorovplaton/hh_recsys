from __future__ import annotations

from abc import ABC
from typing import List, Any, Dict
from itertools import chain
import pandas as pd
from collections import Counter
import random
import json

from src.contexts.RecommenderContext import RecommenderContext
from src.selectors.AbstractSelector import AbstractSelector


class PopularSelector(AbstractSelector, ABC):
    vacancies: List[str] | None

    def __init__(self, name: str = "popular_selector"):
        super().__init__(name)

    def to_str(self) -> str:
        return "popular_model"

    def train(self, data: pd.DataFrame) -> None:
        vacancies = list(chain(*data["vacancy_id"].values))
        counter = Counter(vacancies)
        self.vacancies = list(map(lambda x: x[0], counter.most_common()))

    def get_candidates(self, context, user_id: str, session_id: str, n: int = 10) -> List[str]:
        if self.vacancies is None:
            raise Exception("PopularSelector::get_candidates, self.vacancies is None")

        return self.vacancies[:n]

    def get_candidates_pandas_batch(self,
                                    recommender_context: RecommenderContext,
                                    data: pd.DataFrame, result_df: pd.DataFrame, n: int = 10, item_data: Dict = None) -> None:
        recommendations_column_name = f"_recos_{self.name}"

        result_df[recommendations_column_name] = data["user_id"].apply(lambda user_id: self.vacancies[:n])

    def save_selector_params(self, path: str) -> None:
        if self.vacancies is None:
            raise Exception("PopularSelector::save_selector_params, self.vacancies is None")

        with open(path, "w+") as file:
            json.dump({
                "vacancies": self.vacancies
            }, file)

    def load_selector_params(self, path: str) -> None:
        with open(path, "r") as file:
            data = json.load(file)
            self.vacancies = data["vacancies"]

    @staticmethod
    def load_selector(path: str, name: str = "popular_selector") -> PopularSelector:
        selector = PopularSelector(name)
        selector.load_selector_params(path)
        return selector

    @staticmethod
    def get_name() -> str:
        return "PopularSelector"

    @staticmethod
    def get_file_extension() -> str:
        return "json"
