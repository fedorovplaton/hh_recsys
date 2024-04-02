from src.selectors.PopularSelector import PopularSelector
from src.tasks.AbstractTask import AbstractTask
import pandas as pd


class TrainPopularSelectorTask(AbstractTask):
    def __init__(self):
        super().__init__()

    def run(self, train_path: str, output_path: str) -> None:
        print("TrainPopularSelectorTask::run")

        train = pd.read_parquet(train_path)
        random_selector = PopularSelector()
        random_selector.train(train)
        random_selector.save_selector_params(output_path)


if __name__ == '__main__':
    train_random_selector = TrainPopularSelectorTask()
    train_random_selector.run(
        "../../data/processed/train.parquet",
        f"../dumps/popular_selector/most_freq.{PopularSelector.get_file_extension()}")

    print(f"src/dumps/popular_selector/most_freq.{PopularSelector.get_file_extension()}")
