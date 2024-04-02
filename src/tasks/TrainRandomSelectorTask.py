from src.selectors.RandomSelector import RandomSelector
from src.tasks.AbstractTask import AbstractTask
import pandas as pd


class TrainRandomSelectorTask(AbstractTask):
    random_state: int

    def __init__(self, random_state: int):
        super().__init__()
        self.random_state = random_state

    def run(self, train_path: str, output_path: str) -> None:
        print("TrainRandomSelectorTask::run")

        train = pd.read_parquet(train_path)
        random_selector = RandomSelector(self.random_state)
        random_selector.train(train)
        random_selector.save_selector_params(output_path)


if __name__ == '__main__':
    train_random_selector = TrainRandomSelectorTask(137)
    train_random_selector.run(
        "../../data/processed/train.parquet",
        f"../dumps/random_selector/137.{RandomSelector.get_file_extension()}")

    print(f"src/dumps/random_selector/137.{RandomSelector.get_file_extension()}")