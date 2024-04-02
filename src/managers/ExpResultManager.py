from __future__ import annotations

import json
from typing import List, Dict


class ExpResultManager:
    exp_results_path: str | None
    exp_results: List[Dict] = []

    def __init__(self, exp_results_path):
        self.exp_results_path = exp_results_path
        self.load_exp_results(self.exp_results_path)

    def load_exp_results(self, path):
        with open(path, mode="r+") as file:
            for row in file.readlines():
                past_exp_result = json.loads(row)
                self.exp_results.append(past_exp_result)

    def add_and_save_exp_results(self, exp_result: Dict):
        self.exp_results.append(exp_result)

        with open(self.exp_results_path, mode="a+") as file:
            file.write("\n")
            file.write(json.dumps(exp_result))
