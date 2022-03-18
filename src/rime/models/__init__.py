import numpy as np
import warnings

from .rnn import RNN
from .transformer import Transformer
from .hawkes import Hawkes
from .hawkes_poisson import HawkesPoisson
from .lightfm_bpr import LightFM_BPR
from .bpr import BPR
from .graph_conv import GraphConv
from .lda import LDA
try:
    from .implicit import ALS, LogisticMF
except ImportError:
    ALS = LogisticMF = None
    warnings.warn("Implicit package not properly installed.")

from rime.util import LazyDenseMatrix, RandScore


class Rand:
    def transform(self, D):
        """ return a constant of one """
        shape = (len(D.user_in_test), len(D.item_in_test))
        return RandScore.create(shape)


class Pop:
    def __init__(self, user_rec=True, item_rec=True,
                 item_pseudo=0.01, user_pseudo=0.01, tie_break_noise=0.01):
        self.user_rec = user_rec
        self.item_rec = item_rec
        self.item_pseudo = item_pseudo
        self.user_pseudo = user_pseudo
        self.tie_break_noise = tie_break_noise

    def fit(self, D):
        self.item_scores = D.item_df['_hist_len']
        return self

    def transform(self, D):
        """ user_score * item_score = (user_log_bias + item_log_bias).exp() """
        user_scores = self.user_pseudo + D.user_in_test['_hist_len'] \
            if self.user_rec else np.ones(len(D.user_in_test))

        item_scores = self.item_scores.reindex(D.item_in_test.index, fill_value=0) \
            if self.item_rec else np.ones(len(D.item_in_test))

        ind_logits = np.transpose([
            np.log(user_scores + self.user_pseudo), np.ones(len(user_scores))])
        col_logits = np.transpose([
            np.ones(len(item_scores)), np.log(item_scores + self.item_pseudo)])

        S = (LazyDenseMatrix(ind_logits) @ LazyDenseMatrix(col_logits).T).exp()
        return S + RandScore.create(S.shape) * self.tie_break_noise


class EMA:
    def __init__(self, horizon):
        self.horizon = horizon

    def transform(self, D):
        def fn(ts, T):
            ttl = (T - np.array(ts)) / self.horizon
            return np.exp(-ttl).sum()
        user_scores = D.user_in_test.apply(
            lambda x: fn(x['_hist_ts'], x['TEST_START_TIME']), axis=1)
        item_ones = np.ones(len(D.item_in_test))
        return (LazyDenseMatrix(np.log(user_scores)[:, None]) @
                LazyDenseMatrix(item_ones[:, None]).T).exp()
