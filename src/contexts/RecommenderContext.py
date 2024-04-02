from __future__ import annotations

from typing import Dict

import numpy as np


class RecommenderContext:
    ials_user_embeddings: np.ndarray | None
    ials_item_embeddings: np.ndarray | None

    ials_user_id_2_idx_map: Dict | None
    ials_item_id_2_idx_map: Dict | None

    def __init__(self):
        pass

    def set_ials_user_embeddings(self, user_embeddings: np.ndarray) -> None:
        self.ials_user_embeddings = user_embeddings

    def set_ials_item_embeddings(self, item_embeddings: np.ndarray) -> None:
        self.ials_item_embeddings = item_embeddings

    def get_ials_user_embeddings(self) -> np.ndarray | None:
        return self.ials_user_embeddings

    def get_ials_item_embeddings(self) -> np.ndarray | None:
        return self.ials_item_embeddings

    def set_ials_user_id_2_idx_map(self, ials_user_id_2_idx_map) -> None:
        self.ials_user_id_2_idx_map = ials_user_id_2_idx_map

    def set_ials_item_id_2_idx_map(self, ials_item_id_2_idx_map) -> None:
        self.ials_item_id_2_idx_map = ials_item_id_2_idx_map

    def get_ials_user_id_2_idx_map(self) -> Dict | None:
        return self.ials_user_id_2_idx_map

    def get_ials_item_id_2_idx_map(self) -> Dict | None:
        return self.ials_item_id_2_idx_map
