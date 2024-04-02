import json
from collections import Counter
from itertools import chain

from tqdm import tqdm

from src.managers.ItemDataManager import ItemDataManager
from src.selectors.RandomSelector import RandomSelector
from src.selectors.TopSelector import TopSelector
from src.tasks.AbstractTask import AbstractTask
import pandas as pd


class TrainTopSelectorTask(AbstractTask):
    def __init__(self):
        super().__init__()

    def run(self, train_path: str, item_data_path: str, output_path: str) -> None:
        print("TrainTopSelectorTask::run")

        item_data_manager = ItemDataManager()
        item_data = item_data_manager.get_item_data_dict(item_data_path)
        train = pd.read_parquet(train_path)
        top_selector = TopSelector()
        top_selector.train(train, item_data)
        top_selector.save_selector_params(output_path)


if __name__ == '__main__':
    train_random_selector = TrainTopSelectorTask()
    train_random_selector.run(
        "../../data/raw/hh_recsys_train_hh.pq",
        "../dumps/production/item_data/item_schedule_region.parquet",
        f"../dumps/production/top_selector/top_selector.{RandomSelector.get_file_extension()}")
