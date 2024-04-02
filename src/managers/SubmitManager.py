import pandas as pd
from os.path import join as path_join
import polars as pl


class SubmitManager:
    @staticmethod
    def prepare_submit(recommendations_path: str):
        sample = pl.from_pandas(pd.read_parquet(recommendations_path))
        output_path = path_join(*(recommendations_path.split("/")[:-1] + ["sample.pq"]))

        print(recommendations_path)
        print(output_path)
        print(sample.head(2))

        sample.write_parquet(output_path)
