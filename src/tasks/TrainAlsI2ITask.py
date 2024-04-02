import os
from abc import ABC

from src.selectors.IAlsSelector import IAlsSelector
from src.tasks.AbstractTask import AbstractTask
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import implicit
import pickle
from os.path import join as path_join


class TrainAlsI2ITask(AbstractTask, ABC):

    def __init__(self):
        super().__init__()

    def sparsity(self, pairs):
        return len(pairs) / (len(pairs["user_id"].unique()) * len(pairs["vacancy_id"].unique()))

    def run(self,
            train_path: str,
            output_root: str,
            min_user_counts: int = 10,
            min_item_counts: int = 15) -> None:

        print("TrainAlsI2ITask::start")

        train = pd.read_parquet(train_path)
        selector = IAlsSelector()
        selector.train(train, None, min_user_counts, min_item_counts)
        selector.save_selector_params(output_root)

        """
            i2i
        """
        ials = selector.get_model()
        idx2vac = {idx: vac for vac, idx in selector.vac2idx.items()}
        i2i_path = path_join(output_root, "i2i.parquet")

        i2i_ids, i2i_scores = ials.similar_items([i for i in range(len(ials.item_factors))], 101)
        vacancy_ids_ = []
        neighbours_  = []
        for vacancy_als_idx, (i, d) in enumerate(zip(i2i_ids, i2i_scores)):
            vacancy_ids_.append(idx2vac[vacancy_als_idx])
            neighbours_.append(list(map(
                lambda x: idx2vac[x[0]], filter(lambda x: x[1] >= 0.9 and x[0] != vacancy_als_idx, zip(i, d)))))
        als_i2i = pd.DataFrame.from_dict({
            "vacancy_id": vacancy_ids_,
            "neighbours": neighbours_
        })
        als_i2i = als_i2i[als_i2i["neighbours"].apply(len) > 0]

        als_i2i.to_parquet(i2i_path)
        print("TrainAlsI2ITask::end")


if __name__ == '__main__':
    task = TrainAlsI2ITask()

    task.run(
        "../../data/raw/hh_recsys_train_hh.pq",
        "../dumps/production/als_10_15",
        10,
        15
    )

    task.run(
        "../../data/processed/train.parquet",
        "../dumps/als_10_15",
        10,
        15
    )
