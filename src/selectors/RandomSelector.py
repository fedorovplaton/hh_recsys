from __future__ import annotations

from abc import ABC
from typing import List, Dict
from itertools import chain
import pandas as pd
import random
import json

from src.contexts.RecommenderContext import RecommenderContext
from src.selectors.AbstractSelector import AbstractSelector


class RandomSelector(AbstractSelector, ABC):
    random_state: int
    vacancies: List[str] | None

    def __init__(self, random_state, name: str = "random_selector"):
        super().__init__(name)
        self.set_random_state(random_state)

    def to_str(self) -> str:
        return f"random_model_{self.random_state}"

    def train(self, data: pd.DataFrame) -> None:
        self.set_vacancies(list(set(chain(*data["vacancy_id"].values))))

    def set_vacancies(self, vacancies: List[str]):
        self.vacancies = vacancies

    def set_random_state(self, random_state):
        self.random_state = random_state

    def get_random_vacancies(self, num) -> List[str]:
        if self.vacancies is None:
            raise Exception("RandomSelector::get_random_vacancies, self.vacancies is None")

        return random.sample(self.vacancies, num)

    def get_candidates(self,
                       context,
                       user_id: str,
                       session_id: str,
                       n: int = 10) -> List[str]:
        return self.get_random_vacancies(n)

    def get_candidates_pandas_batch(self, recommender_context: RecommenderContext,
                                    data: pd.DataFrame, result_df: pd.DataFrame, n: int = 10, item_data: Dict = None) -> None:
        recommendations_column_name = f"_recos_{self.name}"

        result_df[recommendations_column_name] = data["user_id"].apply(
            lambda _: self.get_random_vacancies(n)
        )

    def save_selector_params(self, path: str) -> None:
        if self.vacancies is None:
            raise Exception("RandomSelector::get_random_vacancies, self.vacancies is None")

        with open(path, "w+") as file:
            json.dump({
                "random_state": self.random_state,
                "vacancies": self.vacancies
            }, file)

    def load_selector_params(self, path: str) -> None:
        with open(path, "r") as file:
            data = json.load(file)
            self.set_random_state(data["random_state"])
            self.set_vacancies(data["vacancies"])

    @staticmethod
    def get_file_extension() -> str:
        return "json"

    @staticmethod
    def load_selector(path: str, name: str = "random_selector") -> RandomSelector:
        random_selector = RandomSelector(0, name)
        random_selector.load_selector_params(path)
        return random_selector

    @staticmethod
    def get_name() -> str:
        return "RandomSelector"
