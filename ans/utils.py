import random

from IPython import display
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import torch


class MetricsHistory:

    def __init__(self, index_name: str = 'epoch') -> None:
        self.index_name = index_name
        self.data = []

    def append(self, **kwargs) -> None:
        self.data.append(kwargs)

    def df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        df.index.name = self.index_name
        return df

    def best_results(self, metric: str, mode: str) -> pd.DataFrame:
        df = self.df()
        get_best_idx = lambda col: (col.idxmax() if mode == 'max' else col.idxmin())
        return df.iloc[get_best_idx(df.get(metric))]

    def update(self, print_df: bool = True, **append) -> None:
        if append:
            self.data.append(append)
        display.clear_output(wait=True)
        if print_df:
            print(self.df())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
