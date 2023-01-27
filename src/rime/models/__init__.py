import numpy as np
import warnings

from .rnn import RNN
from .transformer import Transformer
from rime.util import LazyDenseMatrix, RandScore, MissingModel

try:
    from .hawkes import Hawkes
except ImportError as e:
    Hawkes = MissingModel("Hawkes", e)

try:
    from .hawkes_poisson import HawkesPoisson
except ImportError as e:
    HawkesPoisson = MissingModel("HawkesPoisson", e)

try:
    from .lightfm_bpr import LightFM_BPR
except ImportError as e:
    LightFM_BPR = MissingModel("LightFM_BPR", e)

try:
    from .bpr import BPR
except ImportError as e:
    BPR = MissingModel("BPR", e)

from .zero_shot import TF_IDF, BayesLM, ItemKNN

try:
    from .graph_conv import GraphConv
except ImportError as e:
    GraphConv = MissingModel("GraphConv", e)
try:
    from .lda import LDA
except ImportError as e:
    LDA = MissingModel("LDA", e)
try:
    from .implicit import ALS, LogisticMF
except ImportError as e:
    ALS = MissingModel("ALS", e)
    LogisticMF = MissingModel("LogisticMF", e)


class Rand:
    def transform(self, D):
        """ return a constant of one """
        shape = (len(D.user_in_test), len(D.item_in_test))
        return RandScore.create(shape)


class Pop:
    def __init__(self, user_rec=True, item_rec=True,
                 item_pseudo=0.01, user_pseudo=0.01, tie_breaker=0.01):
        self.user_rec = user_rec
        self.item_rec = item_rec
        self.item_pseudo = item_pseudo
        self.user_pseudo = user_pseudo
        self.tie_breaker = tie_breaker

    def fit(self, D):
        self.item_scores = D.item_df['_hist_len']
        return self

    def transform(self, D):
        """ user_score * item_score = (user_log_bias + item_log_bias).exp() """
        user_scores = self.user_pseudo + D.user_in_test['_hist_len'].values \
            if self.user_rec else np.ones(len(D.user_in_test))

        item_scores = self.item_pseudo + self.item_scores.reindex(
                D.item_in_test.index, fill_value=0).values \
            if self.item_rec else np.ones(len(D.item_in_test))

        S = LazyDenseMatrix(user_scores[:, None]) * \
            LazyDenseMatrix(item_scores[None, :])  # mtpp implies *
        if self.tie_breaker > 0:
            S = S + RandScore.create(S.shape) * self.tie_breaker
        return S


class EMA:
    def __init__(self, horizon):
        self.horizon = horizon

    def transform(self, D):
        def fn(ts, T):
            ttl = (T - np.array(ts)) / self.horizon
            return np.exp(-ttl).sum()
        user_scores = D.user_in_test.apply(
            lambda x: fn(x['_hist_ts'], x['TEST_START_TIME']), axis=1)
        item_zeros = np.zeros(len(D.item_in_test))
        return LazyDenseMatrix(user_scores[:, None]) + LazyDenseMatrix(item_zeros[None, :])
