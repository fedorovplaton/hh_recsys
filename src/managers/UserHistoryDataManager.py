from __future__ import annotations

import pandas as pd


class UserHistoryDataManager:
    train_path: str | None

    def __init__(self):
        pass

    @staticmethod
    def get_session_min_start(action_dt):
        return str(action_dt[0])[:19]

    @staticmethod
    def get_union_sorted_session(row):
        row_data = list(map(
            lambda x: (x[1], x[2]),
            sorted(zip(row["action_dt"], row["vacancy_id"], row["action_type"]))))

        return row_data

    @staticmethod
    def create_history_data(train_path: str, output_path: str):
        data = pd.read_parquet(train_path)

        data["session_min_start"] = data["action_dt"].apply(UserHistoryDataManager.get_session_min_start)
        data = data.sort_values(by="session_min_start", ascending=True)

        data["union_session"] = data[["vacancy_id", "action_type", "action_dt"]].apply(
            UserHistoryDataManager.get_union_sorted_session, axis=1)

        data = data[["user_id", "union_session"]]
        data = data.groupby("user_id", as_index=False)["union_session"].apply(list)

        data["items"] = data["union_session"].apply(
            lambda sessions: list(map(lambda session: list(map(lambda event: event[0], session)), sessions))
        )
        data["item_actions"] = data["union_session"].apply(
            lambda sessions: list(map(lambda session: list(map(lambda event: event[1], session)), sessions))
        )
        data = data[["user_id", "items", "item_actions"]]

        data.to_parquet(output_path)

    @staticmethod
    def load_history_data(history_data_path: str):
        return pd.read_parquet(history_data_path)


if __name__ == '__main__':
    output_path_ = "../dumps/production/user_history/history_data.parquet"

    UserHistoryDataManager.create_history_data("../../data/raw/hh_recsys_train_hh.pq", output_path_)
    print(UserHistoryDataManager.load_history_data(output_path_).head(2))
