from __future__ import annotations

import json
from typing import List, Tuple


class ConfigManager:
    random_state: int
    num_recommendations: int

    # Data pathes
    train_data_path: str
    val_data_path: str | None
    test_data_path: str
    user_history_data_path: str
    item_data_path: str
    dump_audit: bool = False

    catboost_model_path: str | None

    # Selectors
    used_selectors: List[Tuple[str, str, str, int]] = []

    output_path: str

    def __init__(self):
        pass

    def load_config(self, path):
        with open(path, mode="r") as file:
            config = json.load(file)

            self.random_state = config['random_state']
            self.num_recommendations = config['num_recommendations']

            self.train_data_path = config['train_data_path']
            self.test_data_path = config['test_data_path']

            self.user_history_data_path = config['user_history_data_path']
            self.item_data_path = config['item_data_path']

            if "dump_audit" in config:
                self.dump_audit = bool(config["dump_audit"])

            if "catboost_model_path" in config:
                self.catboost_model_path = config["catboost_model_path"]

            if "val_data_path" in config:
                self.val_data_path = \
                    config['val_data_path'] if config['val_data_path'] != "" else None
            else:
                self.val_data_path = None

            for used_selector in config['used_selectors']:
                self.used_selectors.append((
                    used_selector["selector_class"],
                    used_selector["selector_path"],
                    used_selector["selector_name"],
                    used_selector["selector_candidates"]
                ))

            self.output_path = config["output_path"]
