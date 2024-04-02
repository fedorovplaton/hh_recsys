from __future__ import annotations

import json
from abc import ABC
from collections import Counter
from itertools import chain, cycle, islice
from typing import Any, List, Dict, Optional
from implicit.cpu.als import AlternatingLeastSquares
import implicit
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os

from os.path import join as path_join

from src.contexts.RecommenderContext import RecommenderContext
from src.selectors.AbstractSelector import AbstractSelector


class IAlsSelector(AbstractSelector, ABC):
    action_weights = {
        1: 4.0,
        2: 1.0,
        3: 2.0
    }
    alpha = 40
    factors = 50
    random_state = 137
    iterations = 15
    calculate_training_loss = True
    regularization = 0.001
    item_factors: np.ndarray | None
    vac2idx: Dict | None

    def __init__(self, name: str = "ials_selector"):
        super().__init__(name)

    def to_str(self) -> str:
        return "ials_selector"

    def sparsity(self, pairs):
        return len(pairs) / (len(pairs["user_id"].unique()) * len(pairs["vacancy_id"].unique()))

    def train(self, data: pd.DataFrame,
              item_data: Dict = None,
              min_user_counts: int = 10,
              min_item_counts: int = 15) -> None:

        print("TrainAlsI2ITask::start")

        print("TrainAlsI2ITask::load_data")

        train = data
        pairs = train[['user_id', 'vacancy_id', 'action_type']] \
            .explode(['vacancy_id', 'action_type']) \
            .reset_index(drop=True)

        print("TrainAlsI2ITask::generate_dataset")
        user_counts = pairs.groupby("user_id").count().reset_index()
        used_users = user_counts[user_counts["vacancy_id"] >= min_user_counts]["user_id"]
        pairs = pairs.merge(used_users, "inner", "user_id")
        item_counts = pairs.groupby("vacancy_id").count().reset_index()
        used_items = item_counts[item_counts["user_id"] >= min_item_counts]["vacancy_id"]
        pairs = pairs.merge(used_items, "inner", "vacancy_id")

        print("TrainAlsI2ITask::Sparsity::", self.sparsity(pairs))

        unique_users = pairs["user_id"].unique().tolist()
        unique_vacancies = pairs['vacancy_id'].explode().unique().tolist()

        user2idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        vac2idx = {vac_id: idx for idx, vac_id in enumerate(unique_vacancies)}

        users = pairs['user_id'].map(user2idx).to_numpy()
        vacancies = pairs['vacancy_id'].map(vac2idx).to_numpy()
        preferences = pairs['action_type'].map(self.action_weights).to_numpy()

        uv_mat = csr_matrix((preferences, (users, vacancies)))

        os.system("OPENBLAS_NUM_THREADS=1")

        als_model = implicit.als.AlternatingLeastSquares(
            alpha=self.alpha,
            factors=self.factors,
            random_state=self.random_state,
            iterations=self.iterations,
            calculate_training_loss=self.calculate_training_loss,
            regularization=self.regularization
        )

        print("TrainAlsI2ITask::fit_model")
        als_model.fit(uv_mat)

        self.item_factors = als_model.item_factors
        self.vac2idx = vac2idx

    def get_candidates(self, context, user_id: str, session_id: str, n: int = 10) -> List[str]:
        raise Exception("Not implemented")

    def merge_history(self, vacancy_id, items):
        if isinstance(items, np.ndarray) or isinstance(items, list):
            vacancy_id = list(chain(*items)) + list(vacancy_id)
        return vacancy_id

    def prepare_one_step_matrix(self, test, vac2idx, action_weights):
        test["history"] = test[["vacancy_id", "items"]].apply(
            lambda row: self.merge_history(row["vacancy_id"], row["items"]),
            axis=1
        )
        test["history_actions"] = test[["action_type", "item_actions"]].apply(
            lambda row: self.merge_history(row["action_type"], row["item_actions"]),
            axis=1
        )
        pairs = test[['user_id', 'history', 'history_actions']] \
            .explode(['history', 'history_actions']) \
            .reset_index(drop=True)
        used_items = pd.Series(vac2idx.keys())
        used_items.name = "history"
        one_step_pairs = pairs.merge(used_items, "inner", "history")

        unique_users = one_step_pairs["user_id"].unique().tolist()
        user2idx = {user_id: idx for idx, user_id in enumerate(unique_users)}

        users = one_step_pairs['user_id'].map(user2idx).to_numpy()
        vacancies = one_step_pairs['history'].map(vac2idx).to_numpy()
        preferences = one_step_pairs['history_actions'].map(action_weights).to_numpy()

        return csr_matrix((preferences, (users, vacancies))), user2idx

    def get_model(self):
        assert self.item_factors is not None
        assert self.vac2idx is not None

        ials = implicit.als.AlternatingLeastSquares(
            alpha=self.alpha,
            factors=self.factors,
            random_state=self.random_state,
            iterations=self.iterations,
            calculate_training_loss=self.calculate_training_loss,
            regularization=self.regularization
        )
        ials.item_factors = self.item_factors

        return ials

    def get_candidates_pandas_batch(self,
                                    recommender_context: RecommenderContext,
                                    data: pd.DataFrame,
                                    result_df: pd.DataFrame,
                                    n: int = 10,
                                    item_data: Dict = None) -> None:
        assert self.item_factors is not None
        assert self.vac2idx is not None
        recommendations_column_name = f"_recos_{self.name}"

        one_step_matrix, user2idx = self.prepare_one_step_matrix(data, self.vac2idx, self.action_weights)
        idx2vac = {idx: vac for vac, idx in self.vac2idx.items()}

        ials = self.get_model()
        os.system("OPENBLAS_NUM_THREADS=1")

        I, D = ials.recommend(
            np.array([i for i in range(len(user2idx))]),
            one_step_matrix,
            N=100,
            recalculate_user=True,
            filter_already_liked_items=True
        )

        recommender_context.set_ials_item_embeddings(ials.item_factors)
        recommender_context.set_ials_user_embeddings(ials.recalculate_user(
            np.array([i for i in range(len(user2idx))]),
            one_step_matrix
        ))
        recommender_context.set_ials_item_id_2_idx_map(self.vac2idx)
        recommender_context.set_ials_user_id_2_idx_map(user2idx)

        result_df[recommendations_column_name] = data["user_id"] \
            .apply(
                lambda user_id: list(map(lambda x: idx2vac[x], I[user2idx[user_id]]))
                if user_id in user2idx else []
            )

    def save_selector_params(self, root: str) -> None:
        assert self.item_factors is not None
        assert self.vac2idx is not None

        item_factors_root = path_join(root, "item_factors")
        item_factors_path = path_join(item_factors_root, "item_factors.pickle")
        item_names_path = path_join(item_factors_root, "item_names.parquet")

        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(item_factors_root):
            os.makedirs(item_factors_root)

        sorted_vacancies = list(map(lambda x: x[1], sorted([(idx, vac) for vac, idx in self.vac2idx.items()])))

        pd.DataFrame({"vacancy_id": sorted_vacancies}).to_parquet(item_names_path)
        pickle.dump(self.item_factors, open(item_factors_path, 'wb+'), protocol=4)

    def load_selector_params(self, root: str) -> None:
        item_factors_root = path_join(root, "item_factors")
        item_factors_path = path_join(item_factors_root, "item_factors.pickle")
        item_names_path = path_join(item_factors_root, "item_names.parquet")

        self.vac2idx = {vac: i for i, vac in enumerate(pd.read_parquet(item_names_path)["vacancy_id"].tolist())}
        self.item_factors = pickle.load(open(item_factors_path, 'rb'))

    @staticmethod
    def load_selector(root: str, name: str = "ials_selector") -> IAlsSelector:
        selector = IAlsSelector(name)
        selector.load_selector_params(root)

        return selector

    @staticmethod
    def get_name() -> str:
        return "IAlsSelector"

    @staticmethod
    def get_file_extension() -> str:
        return "parquet"
