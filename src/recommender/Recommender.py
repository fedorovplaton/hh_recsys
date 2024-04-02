from __future__ import annotations

from typing import List, Dict

from itertools import cycle
from itertools import islice
from itertools import chain

import numpy as np
import pandas as pd
import os
from os.path import join as path_join
from tqdm import tqdm

from catboost import CatBoostRanker, Pool

from numpy import dot
from numpy.linalg import norm

from src.contexts.RecommenderContext import RecommenderContext
from src.managers.ConfigManager import ConfigManager
from src.managers.ItemDataManager import ItemDataManager
from src.managers.SubmitManager import SubmitManager
from src.managers.UserHistoryDataManager import UserHistoryDataManager
from src.selectors.AbstractSelector import AbstractSelector
from src.selectors.AlreadyViewedSelector import AlreadyViewedSelector
from src.selectors.I2IContentSelector import I2IContentSelector
from src.selectors.I2ISelector import I2ISelector
from src.selectors.IAlsSelector import IAlsSelector
from src.selectors.PopularSelector import PopularSelector
from src.selectors.RandomSelector import RandomSelector
from src.selectors.TopSelector import TopSelector


class Recommender:
    config_path: str | None

    config_manager: ConfigManager = ConfigManager()
    selectors: List[AbstractSelector] = []
    recommender_context: RecommenderContext = RecommenderContext()
    user_history_data_manager: UserHistoryDataManager | None
    user_history_data: pd.DataFrame | None
    item_data: Dict | None
    selector_candidates = {}
    selector_2_id = {
        'already_viewed': 0,
        'ials': 1,
        'als_i2i': 2,
        'content_i2i': 3,
        'top_selector': 4,
        'random_selector': 5
    }

    def init(self, config_path: str):
        print("Recommender::init::start")

        self.config_path = config_path
        self.config_manager.load_config(config_path)

        print("Recommender::loading_selectors")
        for (selector_class, selector_path, selector_name, selector_candidates) in self.config_manager.used_selectors:
            print(f"Recommender::Selector::{selector_class}::{selector_name}")
            selector = self.get_selector(selector_class, selector_path, selector_name)

            if selector is not None:
                self.selectors.append(selector)
                self.selector_candidates[selector.name] = selector_candidates

        print("Recommender::UserHistoryDataManager")
        self.user_history_data_manager = UserHistoryDataManager()
        self.user_history_data = self.user_history_data_manager \
            .load_history_data(self.config_manager.user_history_data_path)
        print("Recommender::ItemDataManager")
        self.item_data = ItemDataManager.get_item_data_dict(self.config_manager.item_data_path)
        print("Recommender::init::end")

    def get_selector(self, selector_class: str, selector_path: str, selector_name: str) -> AbstractSelector | None:
        if selector_class == "RandomSelector":
            return RandomSelector.load_selector(selector_path, selector_name)
        elif selector_class == "PopularSelector":
            return PopularSelector.load_selector(selector_path, selector_name)
        elif selector_class == "AlreadyViewedSelector":
            return AlreadyViewedSelector.load_selector(selector_path, selector_name)
        elif selector_class == "TopSelector":
            return TopSelector.load_selector(selector_path, selector_name)
        elif selector_class == "I2IContentSelector":
            return I2IContentSelector.load_selector(selector_path, selector_name)
        elif selector_class == "I2ISelector":
            return I2ISelector.load_selector(selector_path, selector_name)
        elif selector_class == "IAlsSelector":
            return IAlsSelector.load_selector(selector_path, selector_name)

        return None

    def run_inference(self) -> str:
        assert self.recommender_context is not None
        assert self.user_history_data is not None
        assert self.config_path is not None

        test_path = self.config_manager.test_data_path
        num_recommendations = self.config_manager.num_recommendations
        output_path = self.config_manager.output_path

        test = pd.read_parquet(test_path)
        test = test.merge(self.user_history_data, "left", "user_id")

        output_df = test[["user_id", "session_id"]]
        output_df["blacklist"] = test[["vacancy_id", "action_type", "items", "item_actions"]].apply(
            lambda row: self.get_blacklist(row["vacancy_id"], row["action_type"], row["items"], row["item_actions"]),
            axis=1
        )

        """
            Генерируем рекомендации селекторами
        """
        for selector in tqdm(self.selectors, total=len(self.selectors), desc="Selector"):
            selector.get_candidates_pandas_batch(
                self.recommender_context,
                test,
                output_df,
                self.selector_candidates[selector.name],
                self.item_data
            )

        """
            Надо добавить фичей и сделать audit
        """
        print("Recommender::Inference::PrepareAudit")
        # Вычитаем blacklist из рекомендаций
        recos_columns = list(filter(lambda col: col.startswith("_recos_"), output_df.columns))
        for col in recos_columns:
            output_df[col] = output_df[["blacklist", col]].apply(
                lambda row: self.filter_blacklist(row[col], row["blacklist"]), axis=1
            )
        # Добавляем источник, откуда пришла рекомендация
        for selector_name in self.selector_2_id:
            recommendation_col = f"_recos_{selector_name}"
            selector_id = str(self.selector_2_id[selector_name])

            if recommendation_col in output_df.columns:
                output_df[recommendation_col] = output_df[recommendation_col] \
                    .map(lambda x: list(map(lambda y: y + "::" + selector_id, x)))
        # Объединение рекомендаций в один список
        output_df["recommendations"] = output_df[recos_columns[0]]
        for i in range(1, len(recos_columns)):
            output_df["recommendations"] += output_df[recos_columns[i]]
        output_df["history"] = test["history"]
        output_df["history_actions"] = test["history_actions"]
        print("Recommender::Inference::AddViewedFeatures")
        # Фичи виделили мы уже контент
        output_df["is_viewed"] = output_df[["history", "history_actions", "recommendations"]].apply(
            lambda row: self.is_already_action_type_flags(row["history"], row["history_actions"], 2,
                                                          row["recommendations"]),
            axis=1
        )
        print("Recommender::Inference::AddBookmarkedFeatures")
        output_df["is_bookmark"] = output_df[["history", "history_actions", "recommendations"]].apply(
            lambda row: self.is_already_action_type_flags(row["history"], row["history_actions"], 3,
                                                          row["recommendations"]),
            axis=1
        )
        print("Recommender::Inference::ProcessAudit")
        audit = output_df[["user_id", "session_id", "recommendations", "is_viewed", "is_bookmark"]]
        audit["recommendations_wos"] = audit["recommendations"].map(
            lambda x: list(map(lambda y: y.split("::")[0], x))
        )

        users_w_embeddings = \
            list(
                map(
                    lambda x: x[1],
                    sorted(
                        [(idx, user_id) for user_id, idx in self.recommender_context.get_ials_user_id_2_idx_map().items()]
                    )
                )
            )
        user_embeddings_df = pd.DataFrame.from_dict({
            "user_id": users_w_embeddings,
            "user_embedding": self.recommender_context.get_ials_user_embeddings().tolist()})
        user_embeddings_df["norm"] = user_embeddings_df["user_embedding"].map(norm)
        user_embeddings_df = user_embeddings_df.dropna()
        user_embeddings_df = user_embeddings_df[user_embeddings_df["norm"] >= 1e-8]

        audit = audit.merge(user_embeddings_df, "left", "user_id")

        print("Recommender::Inference::AddCosFeature")
        audit["ials_cos"] = audit[["user_embedding", "recommendations_wos"]].apply(
            lambda row: self.calculate_cos(row["user_embedding"], row["recommendations_wos"], self.recommender_context),
            axis=1
        )
        print("Recommender::Inference::AddDotFeature")
        audit["ials_dot"] = audit[["user_embedding", "recommendations_wos"]].apply(
            lambda row: self.calculate_dot(row["user_embedding"], row["recommendations_wos"], self.recommender_context),
            axis=1
        )
        print("Recommender::Inference::AddItemNormFeature")
        audit["ials_item_norm"] = audit["recommendations_wos"].map(
            lambda r: self.calculate_item_norm(r, self.recommender_context)
        )

        print("Recommender::Inference::ProcessAudit")
        audit = audit[[
            "user_id", "session_id",
            "recommendations", "is_viewed", "is_bookmark",
            "ials_cos", "ials_dot", "ials_item_norm"
        ]]

        # print(audit["ials_cos"].head(20))
        # print(audit["ials_dot"].head(20))
        # print(audit["ials_item_norm"].head(20))

        audit = audit.explode(["recommendations", "is_viewed", "is_bookmark",
                               "ials_cos", "ials_dot", "ials_item_norm"])

        audit["selector_id"] = audit["recommendations"].map(lambda x: int(x.split("::")[1]))
        audit["vacancy_id"] = audit["recommendations"].map(lambda x: x.split("::")[0])

        audit = audit[["user_id", "session_id", "vacancy_id", "selector_id",
                       "is_viewed", "is_bookmark",
                       "ials_cos", "ials_dot", "ials_item_norm"]]

        # print(audit["ials_cos"].head(20))
        # print(audit["ials_dot"].head(20))
        # print(audit["ials_item_norm"].head(20))

        # print(audit.iloc[0])
        # print(audit.iloc[1])
        # print(audit.iloc[2])

        if self.config_manager.dump_audit:
            print("Recommender::Inference::SaveAudit")
            audit_root = "../dumps/audit/"
            audit_name = "__".join(map(lambda x: x.split("_recos_")[1], recos_columns)) + ".parquet"
            audit_path = path_join(audit_root, audit_name)
            audit.to_parquet(audit_path)

        """
            Вопрос: есть ли у нас бустинг в конфиге?
        """

        if self.config_manager.catboost_model_path is not None:
            print("Recommender::Inference::Boosting::Load")
            catboost_model = CatBoostRanker()
            catboost_model = catboost_model.load_model(self.config_manager.catboost_model_path)
            audit_pool = Pool(
                data=audit[['selector_id', 'is_viewed', 'is_bookmark', 'ials_cos', 'ials_dot', 'ials_item_norm']]
            )
            print("Recommender::Inference::Boosting::Predict")
            audit["__predict"] = catboost_model.predict(audit_pool)

            print("Recommender::Inference::Boosting::Processing")
            audit = audit[["user_id", "session_id", "vacancy_id", "__predict"]] \
                .sort_values(by="__predict", ascending=False)

            audit = audit.groupby(["user_id", "session_id"])["vacancy_id"].apply(list).reset_index()
            audit["predictions"] = audit["vacancy_id"].map(lambda x: x[:num_recommendations])
            audit = audit[["user_id", "session_id", "predictions"]]

            # """
            #     Ранжируем рекомендации и вычищаем уже откликнутое
            # """
            # recos_columns = list(filter(lambda col: col.startswith("_recos_"), output_df.columns))
            # output_df["predictions"] = list(map(
            #     lambda blacklist_recos: self.get_first_n_recos(
            #         num_recommendations,
            #         blacklist_recos[0],
            #         *filter(
            #             lambda x: isinstance(x, list) or isinstance(x, np.ndarray),
            #             blacklist_recos[1]
            #         )
            #     ),
            #     zip(output_df["blacklist"], output_df[recos_columns].values)
            # ))
            #
            # """
            #     Берём итоговые колонки только
            # """
            # output_df = output_df[["user_id", "session_id", "predictions"]]

            print("Статистики:")
            # print("Строк в test:", test.shape[0])
            print("Строк в output:", audit.shape[0])
            # print("Юзеров в test:", len(test["user_id"].unique()))
            print("Юзеров в test:", len(audit["user_id"].unique()))

            """
                ...
            """
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            os.system(f"cp {self.config_path} {path_join(output_path, 'config.json')}")
            audit.to_parquet(path_join(output_path, 'output.parquet'))

            return path_join(output_path, 'output.parquet')

    def get_blacklist(self, vacancy_id, action_type, items, item_actions):
        session = list(map(lambda x: x[0], filter(lambda x: x[1] == 1, zip(vacancy_id, action_type))))

        if isinstance(items, list) or isinstance(items, np.ndarray):
            previous_sessions = list(map(lambda x: x[0], filter(lambda x: x[1] == 1, zip(
                chain(*items),
                chain(*item_actions),
            ))))

            session = previous_sessions + session

        return list(set(session))

    def is_already_action_type_flags(self, history, history_actions, action_type, recommendations):
        history_data = {
            vacancy_id: action_type_ for vacancy_id, action_type_ in zip(history, history_actions)
        }
        flags = []

        for recommended_item in recommendations:
            recommended_item_ = recommended_item.split("::")[0]

            if recommended_item_ not in history_data:
                flags.append(0)
                continue

            if history_data[recommended_item_] == action_type:
                flags.append(1)
            else:
                flags.append(0)
        return flags

    def calculate_cos(self, user_embedding, vacancies, recommender_context: RecommenderContext):
        if isinstance(user_embedding, list) or isinstance(user_embedding, np.ndarray):
            if np.isnan(user_embedding[0]):
                return [float(0.0) for _ in vacancies]
        else:
            if np.isnan(user_embedding):
                return [float(0.0) for _ in vacancies]

        item_id_2_idx = recommender_context.get_ials_item_id_2_idx_map()
        result = []

        for vacancy_id in vacancies:
            if vacancy_id not in item_id_2_idx:
                result.append(float(0.0))
            else:
                item_embedding = recommender_context.get_ials_item_embeddings()[item_id_2_idx[vacancy_id]]

                _tmp = dot(user_embedding, item_embedding) / norm(user_embedding) / norm(item_embedding)

                # print(_tmp, user_embedding, item_embedding)

                result.append(float(_tmp))

        return result

    def calculate_dot(self, user_embedding, vacancies, recommender_context: RecommenderContext):
        if isinstance(user_embedding, list) or isinstance(user_embedding, np.ndarray):
            if np.isnan(user_embedding[0]):
                return [float(0.0) for _ in vacancies]
        else:
            if np.isnan(user_embedding):
                return [float(0.0) for _ in vacancies]

        item_id_2_idx = recommender_context.get_ials_item_id_2_idx_map()
        result = []

        for vacancy_id in vacancies:
            if vacancy_id not in item_id_2_idx:
                result.append(float(0.0))
            else:
                item_embedding = recommender_context.get_ials_item_embeddings()[item_id_2_idx[vacancy_id]]
                result.append(float(dot(user_embedding, item_embedding)))

        return result

    def calculate_item_norm(self, vacancies, recommender_context: RecommenderContext):
        item_id_2_idx = recommender_context.get_ials_item_id_2_idx_map()
        result = []

        for vacancy_id in vacancies:
            if vacancy_id not in item_id_2_idx:
                result.append(float(0.0))
            else:
                item_embedding = recommender_context.get_ials_item_embeddings()[item_id_2_idx[vacancy_id]]
                result.append(float(norm(item_embedding)))

        return result

    def filter_blacklist(self, recommendations_list, blacklist):
        if len(blacklist) == 0:
            return recommendations_list
        blacklist = set(blacklist)
        return list(filter(lambda x: x not in blacklist, recommendations_list))

    def get_first_n_recos(self, n: int, blacklist: List[str], *iterables):
        blacklist = set(blacklist)
        output = []
        for value in self.roundrobin(*iterables):
            if value not in blacklist:
                output.append(value)

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


if __name__ == '__main__':
    is_prod = False

    recommender = Recommender()
    recommender.init("../config_prod.json" if is_prod else "../config.json")
    output_path = recommender.run_inference()
    SubmitManager.prepare_submit(output_path)

    print("output_path:", output_path)

    if is_prod:
        exit(0)

    result = pd.read_parquet(output_path)

    test_path = "../../data/processed/test_inference.parquet"
    test = pd.read_parquet(test_path)[["user_id", "session_id", "target_vacancy_id"]]

    analysis = pd.merge(test, result, "inner", ["user_id", "session_id"])

    print("Средняя длина рекомендаций:", analysis["predictions"].apply(len).mean())


    def reciprocal_rank(row):
        predictions = row["predictions"]
        target = row["target_vacancy_id"]

        for idx, prediction in enumerate(predictions):
            if prediction == target:
                return 1 / (idx + 1)

        return 0


    def recall(row):
        predictions = row["predictions"]
        target = row["target_vacancy_id"]

        return int(target in set(predictions))


    analysis["rr"] = analysis[["predictions", "target_vacancy_id"]].apply(reciprocal_rank, axis=1)
    analysis["recall"] = analysis[["predictions", "target_vacancy_id"]].apply(recall, axis=1)
    print("mrr:", analysis["rr"].mean())
    print("mean recall:", analysis["recall"].mean())

"""

Посдение 10 поскоренных событий
Средняя длина рекомендаций: 99.74521990590875
mrr: 0.002695191388650043
mean recall: 0.024342871940955586

Последние 10 поскоренных событий + фильтр по remote/flyInFlyOut
Средняя длина рекомендаций: 70.70680650869697
mrr: 0.0019128752670166027
mean recall: 0.018990893003582372

Последние 10 поскоренных событий + фильтры по remote/most_freq/most
Средняя длина рекомендаций: 84.84902240062152
mrr: 0.003047516944555497
mean recall: 0.02861582286676162

Средняя длина рекомендаций: 73.8940394492641
mrr: 0.01051688647523176
mean recall: 0.08097026198800121

Уже просмотренные
Средняя длина рекомендаций: 32.510574474513355
mrr: 0.08740992185094804
mean recall: 0.18427640381544305

I2I ALS
Средняя длина рекомендаций: 85.96251456687816
mrr: 0.013008181565252271
mean recall: 0.11431222754542708

iAls
Средняя длина рекомендаций: 93.09637878199318
mrr: 0.01019388580551786
mean recall: 0.08474686002848635

Old content I2I
Средняя длина рекомендаций: 94.6797013250464
mrr: 0.003036423262191057
mean recall: 0.028702145107686997

New content I2I
Средняя длина рекомендаций: 86.86674004057146
mrr: 0.005445368473200986
mean recall: 0.04952738573093358
Без фильтров каких-то по region_id
Средняя длина рекомендаций: 95.34658379731538
mrr: 0.006010454260337809
mean recall: 0.06504380853726963

Уже просмотренные + I2I ALS
Средняя длина рекомендаций: 113.44855194440848
mrr: 0.08126225619877876
mean recall: 0.2611247787992576

Уже просмотренные + I2I ALS + iAls
Средняя длина рекомендаций: 206.5470887824248
mrr: 0.07821118464079169
mean recall: 0.31032845612672105

Уже просмотренные + I2I ALS + iAls + ContentI2I
Средняя длина рекомендаций: 223.4838145798265
mrr: 0.07919815693122664
mean recall: 0.4041823125728344

"""
