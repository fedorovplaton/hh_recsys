from __future__ import annotations

import json
from abc import ABC
from collections import Counter
from itertools import chain, cycle, islice
from typing import Any, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.contexts.RecommenderContext import RecommenderContext
from src.selectors.AbstractSelector import AbstractSelector


class TopSelector(AbstractSelector, ABC):
    tops: Dict | None

    def __init__(self, name: str = "top_selector"):
        super().__init__(name)

    def to_str(self) -> str:
        return "top_selector"

    def train(self, data: pd.DataFrame, item_data: Dict = None) -> None:
        assert item_data is not None

        train = data
        vacancy_counter = Counter(chain(*train["vacancy_id"].values))

        vacancy_ids = []
        vacancy_counts = []
        vacancy_work_schedule = []
        vacancy_region_id = []

        for vacancy_id in vacancy_counter:
            if vacancy_id in item_data["work_schedule"]:
                vacancy_ids.append(vacancy_id)
                vacancy_counts.append(vacancy_counter[vacancy_id])
                vacancy_work_schedule.append(item_data["work_schedule"][vacancy_id])
                vacancy_region_id.append(item_data["region_id"][vacancy_id])

        data = pd.DataFrame.from_dict({
            "vacancy_id": vacancy_ids,
            "vacancy_count": vacancy_counts,
            "vacancy_work_schedule": vacancy_work_schedule,
            "vacancy_region_id": vacancy_region_id
        })

        tops = {}

        for region_id in tqdm(data["vacancy_region_id"].unique(), total=len(data["vacancy_region_id"].unique())):
            top_vacancies = data[data["vacancy_region_id"] == region_id] \
                .sort_values(by="vacancy_count", ascending=False)["vacancy_id"].head(100).to_list()
            tops[region_id] = top_vacancies

        for work_schedule in tqdm(data["vacancy_work_schedule"].unique(),
                                  total=len(data["vacancy_work_schedule"].unique())):
            top_vacancies = data[data["vacancy_work_schedule"] == work_schedule] \
                .sort_values(by="vacancy_count", ascending=False)["vacancy_id"].head(100).to_list()
            tops[work_schedule] = top_vacancies

        self.tops = tops

    def get_candidates(self, context, user_id: str, session_id: str, n: int = 10) -> List[str]:
        raise Exception("Not implemented")

    def get_candidates_pandas_batch(self,
                                    recommender_context: RecommenderContext,
                                    data: pd.DataFrame,
                                    result_df: pd.DataFrame,
                                    n: int = 10,
                                    item_data: Dict = None) -> None:
        assert item_data is not None

        recommendations_column_name = f"_recos_{self.name}"

        result_df[recommendations_column_name] = data[["vacancy_id", "action_type", "items", "item_actions"]].apply(
            lambda row: self.get_top_recos(row["vacancy_id"],
                                           row["action_type"],
                                           row["items"],
                                           row["item_actions"],
                                           item_data,
                                           self.tops,
                                           n),
            axis=1
        )

    def get_top_recos(self, vacancy_id, action_type, items, item_actions, item_data, tops, n):
        if isinstance(items, np.ndarray) or isinstance(items, list):
            vacancy_id = list(vacancy_id) + list(chain(*items))

        if isinstance(item_actions, np.ndarray) or isinstance(item_actions, list):
            action_type = list(action_type) + list(chain(*item_actions))

        get_workSchedule = lambda vacancy_id: item_data["work_schedule"][vacancy_id] if vacancy_id in item_data["work_schedule"] else None
        get_region = lambda vacancy_id: item_data["region_id"][vacancy_id] if vacancy_id in item_data["region_id"] else None

        recommendations = []

        work_schedules = Counter(filter(lambda x: x, map(get_workSchedule, vacancy_id)))
        vacancy_ids_not_remote = filter(lambda vacancy: get_workSchedule(vacancy) not in ["remote", "flyInFlyOut"],
                                        vacancy_id)
        vacancy_responded_not_remote = filter(
            lambda vacancy: get_workSchedule(vacancy) not in ["remote", "flyInFlyOut"], map(
                lambda x: x[0],
                filter(
                    lambda x: x[1] == 1 and get_workSchedule(x[0]) == "fullDay",
                    zip(vacancy_id, action_type)
                )
            ))
        regions_not_remote = Counter(map(get_region, vacancy_ids_not_remote))
        regions_responded_not_remote = Counter(map(get_region, vacancy_responded_not_remote))

        if "remote" in work_schedules or "flyInFlyOut" in work_schedules:
            recommendations.append(tops["remote"])
            recommendations.append(tops["flyInFlyOut"])

        if len(regions_responded_not_remote) > 0:
            if regions_responded_not_remote.most_common(1)[0][0] in tops:
                recommendations.append(tops[regions_responded_not_remote.most_common(1)[0][0]])

        if len(regions_not_remote) > 0:
            if regions_not_remote.most_common(1)[0][0] in tops:
                recommendations.append(tops[regions_not_remote.most_common(1)[0][0]])

        output = []

        for item in self.roundrobin(*recommendations):
            output.append(item)

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
        assert self.tops is not None

        with open(path, mode="w+") as file:
            json.dump(self.tops, file)

    def load_selector_params(self, path: str) -> None:
        with open(path, mode="r") as file:
            self.tops = json.load(file)

    @staticmethod
    def load_selector(path: str, name: str = "top_selector") -> TopSelector:
        top_selector = TopSelector(name)
        top_selector.load_selector_params(path)

        return top_selector

    @staticmethod
    def get_name() -> str:
        return "TopSelector"

    @staticmethod
    def get_file_extension() -> str:
        return "json"


if __name__ == '__main__':
    print(pd.read_parquet("../dumps/recommendations/top_model/output.parquet").head(20))
