from abc import ABC, abstractmethod
from typing import List, Any, Dict
import pandas as pd

from src.contexts.RecommenderContext import RecommenderContext


class AbstractSelector(ABC):
    name: str

    def __init__(self, name: str = "default_selector_name"):
        self.name = name

    @abstractmethod
    def to_str(self) -> str:
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def get_candidates(self,
                       context,
                       user_id: str,
                       session_id: str,
                       n: int = 10) -> List[str]:
        pass

    @abstractmethod
    def get_candidates_pandas_batch(self,
                                    recommender_context: RecommenderContext,
                                    data: pd.DataFrame,
                                    result_df: pd.DataFrame,
                                    n: int = 10,
                                    item_data: Dict = None) -> None:
        pass

    @abstractmethod
    def save_selector_params(self, path: str) -> None:
        pass

    @abstractmethod
    def load_selector_params(self, path: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load_selector(path: str, name: str = "default_selector_name") -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_file_extension() -> str:
        pass
