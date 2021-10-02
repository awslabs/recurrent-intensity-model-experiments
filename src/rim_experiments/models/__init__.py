import pandas as pd, numpy as np
import functools, collections

from .rnn import RNN
from .hawkes import Hawkes
from .hawkes_poisson import HawkesPoisson
from .lightfm_bpr import LightFM_BPR
from .implicit import ALS, LogisticMF

from rim_experiments.util import ExponentiatedLowRankDataFrame


class Rand:
    def transform(self, D):
        """ return a constant of one """
        shape = (len(D.user_in_test), len(D.item_df))
        return ExponentiatedLowRankDataFrame(
            np.zeros(shape[0])[:, None], np.zeros(shape[1])[:, None], 1,
            index=D.user_in_test.index, columns=D.item_df.index)


class Pop:
    def __init__(self, user_rec=True, item_rec=True):
        self.user_rec = user_rec
        self.item_rec = item_rec

    def transform(self, D):
        """ user_score * item_score = (user_log_bias + item_log_bias).exp() """
        user_scores = np.fmax(0.01, D.user_in_test['_hist_len']) \
            if self.user_rec else np.ones(len(D.user_in_test))

        item_scores = np.fmax(0.01, D.item_df['_hist_len']) \
            if self.item_rec else np.ones(len(D.item_df))

        ind_logits = np.vstack([np.log(user_scores), np.ones(len(user_scores))]).T
        col_logits = np.vstack([np.ones(len(item_scores)), np.log(item_scores)]).T

        return ExponentiatedLowRankDataFrame(
            ind_logits, col_logits, 1,
            index=D.user_in_test.index, columns=D.item_df.index)


class EMA:
    def __init__(self, horizon):
        self.horizon = horizon

    def transform(self, D):
        fn = lambda ts: np.exp(- (ts[-1] - np.array(ts[:-1])) / self.horizon).sum()
        user_scores = list(map(fn, D.user_in_test['_timestamps'].values))

        return ExponentiatedLowRankDataFrame(
            np.log(user_scores)[:, None], np.ones(len(D.item_df))[:, None], 1,
            index=D.user_in_test.index, columns=D.item_df.index)
