import pandas as pd, numpy as np
import functools, collections, warnings

from .rnn import RNN
from .transformer import Transformer
from .hawkes import Hawkes
from .hawkes_poisson import HawkesPoisson
from .lightfm_bpr import LightFM_BPR
from .bpr import BPR
from .graph_conv import GraphConv
try:
    from .implicit import ALS, LogisticMF
except ImportError:
    ALS = LogisticMF = None
    warnings.warn("Implicit package not properly installed.")

from rime.util import LowRankDataFrame, RandScore


class Rand:
    def transform(self, D):
        """ return a constant of one """
        shape = (len(D.user_in_test), len(D.item_in_test))
        return RandScore(np.arange(shape[0]), np.arange(shape[1]))


class Pop:
    def __init__(self, user_rec=True, item_rec=True):
        self.user_rec = user_rec
        self.item_rec = item_rec

    def fit(self, D):
        self.item_scores = np.fmax(0.01, D.item_df['_hist_len'])
        return self

    def transform(self, D):
        """ user_score * item_score = (user_log_bias + item_log_bias).exp() """
        user_scores = np.fmax(0.01, D.user_in_test['_hist_len']) \
            if self.user_rec else np.ones(len(D.user_in_test))

        item_scores = self.item_scores.reindex(D.item_in_test.index, fill_value=0.01) \
            if self.item_rec else np.ones(len(D.item_in_test))

        ind_logits = np.vstack([np.log(user_scores), np.ones(len(user_scores))]).T
        col_logits = np.vstack([np.ones(len(item_scores)), np.log(item_scores)]).T

        S = LowRankDataFrame(
            ind_logits, col_logits,
            index=D.user_in_test.index, columns=D.item_in_test.index, act='exp')
        return S + RandScore.like(S) * 0.01


class EMA:
    def __init__(self, horizon):
        self.horizon = horizon

    def transform(self, D):
        fn = lambda ts: np.exp(- (ts[-1] - np.array(ts[:-1])) / self.horizon).sum()
        user_scores = list(map(fn, D.user_in_test['_timestamps'].values))

        return LowRankDataFrame(
            np.log(user_scores)[:, None], np.ones(len(D.item_in_test))[:, None],
            index=D.user_in_test.index, columns=D.item_in_test.index, act='exp')
