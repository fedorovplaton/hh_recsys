from abc import ABC
from typing import Any, List, Dict

import numpy as np
import pandas as pd
from itertools import chain

from src.contexts.RecommenderContext import RecommenderContext
from src.selectors.AbstractSelector import AbstractSelector


class AlreadyViewedSelector(AbstractSelector, ABC):
    def __init__(self, name: str = "already_viewed_selector"):
        super().__init__(name)

    def to_str(self) -> str:
        return "already_viewed_selector"

    def train(self, data: pd.DataFrame) -> None:
        pass

    def get_candidates(self, context, user_id: str, session_id: str, n: int = 10) -> List[str]:
        raise Exception('not implemented')
        return []

    def get_candidates_pandas_batch(self, recommender_context: RecommenderContext,
                                    data: pd.DataFrame, result_df: pd.DataFrame, n: int = 10, item_data: Dict = None) -> None:
        recommendations_column_name = f"_recos_{self.name}"

        result_df[recommendations_column_name] = data[["vacancy_id", "action_type", "items", "item_actions"]].apply(
            lambda row: self.get_not_responded_vacancies(row["vacancy_id"], row["action_type"], row["items"], row["item_actions"]),
            axis=1
        )

    def get_not_responded_vacancies(self, vacancy_id, action_type, items, item_actions):
        session = list(map(lambda x: x[0], filter(lambda x: x[1] != 1, zip(vacancy_id, action_type))))

        if isinstance(items, list) or isinstance(items, np.ndarray):
            previous_sessions = list(map(lambda x: x[0], filter(lambda x: x[1] != 1, zip(
                chain(*items),
                chain(*item_actions),
            ))))

            session = previous_sessions + session

        return session[::-1]

    def save_selector_params(self, path: str) -> None:
        pass

    def load_selector_params(self, path: str) -> None:
        pass

    @staticmethod
    def load_selector(path: str, name: str = "already_viewed_selector") -> Any:
        return AlreadyViewedSelector(name)

    @staticmethod
    def get_name() -> str:
        return "AlreadyViewedSelector"

    @staticmethod
    def get_file_extension() -> str:
        return ""
