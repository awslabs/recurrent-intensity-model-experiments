import pandas as pd, numpy as np
import functools, collections

from .rnn import RNN
from .hawkes import Hawkes
from .hawkes_poisson import HawkesPoisson
from .lightfm_bpr import LightFM_BPR


class Rand:
    def transform(self, D):
        return pd.DataFrame(
            np.random.rand(len(D.user_in_test), len(D.item_in_test)),
            index=D.user_in_test.index, columns=D.item_in_test.index)


class Pop:
    def __init__(self, user_rec=True, item_rec=True):
        self.user_rec = user_rec
        self.item_rec = item_rec

    def transform(self, D):
        user_scores = np.fmax(0.01, D.user_in_test['_hist_len']) \
            if self.user_rec else np.ones(len(D.user_in_test))

        item_scores = np.fmax(0.01, D.item_in_test['_hist_len']) \
            if self.item_rec else np.ones(len(D.item_in_test))

        return pd.DataFrame(
            np.outer(user_scores, item_scores),
            index=D.user_in_test.index, columns=D.item_in_test.index)


class EMA:
    def __init__(self, horizon):
        self.horizon = horizon

    def transform(self, D):
        fn = lambda ts: np.exp(- (ts[-1] - np.array(ts[:-1])) / self.horizon).sum()
        user_scores = list(map(fn, D.user_in_test['_timestamps'].values))

        return pd.DataFrame(
            np.outer(user_scores, np.ones(len(D.item_in_test))),
            index=D.user_in_test.index, columns=D.item_in_test.index)
