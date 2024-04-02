from typing import Dict

import pandas as pd
import polars as pl


class ItemDataManager:
    def __init__(self):
        pass

    @staticmethod
    def prepare_items_data(vacancies_path: str, items_data_path: str) -> None:
        if vacancies_path.endswith(".pq"):
            items = pl.read_parquet(vacancies_path).to_pandas()
        else:
            items = pd.read_parquet(vacancies_path)

        items = items[["vacancy_id", "workSchedule", "area.regionId"]]
        items = items.rename(columns={
            "vacancy_id": "item_id",
            "workSchedule": "work_schedule",
            "area.regionId": "region_id"
        })
        items = items.dropna()
        items.to_parquet(items_data_path)

    @staticmethod
    def get_item_data_dict(item_data_path: str) -> Dict:
        return pd.read_parquet(item_data_path).set_index("item_id").to_dict()


if __name__ == '__main__':
    vacancies_path = "../../data/raw/hh_recsys_vacancies.pq"
    items_data_path = "../dumps/production/item_data/item_schedule_region.parquet"

    # ItemDataManager.prepare_items_data(vacancies_path, items_data_path)
    d = ItemDataManager.get_item_data_dict(items_data_path)

    print(d["work_schedule"]["v_1446558"], d["region_id"]["v_1446558"])
