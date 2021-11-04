import numpy as np, pandas as pd
import scipy.optimize
import functools
from ..util import LowRankDataFrame

class HawkesPoisson:
    """ intensity is additive function over non-negative states """
    def __init__(self, hawkes_model):
        self.hawkes_model = hawkes_model

    def fit(self, V):
        self.V = V
        H = self.hawkes_model.transform(V, state_only=True)
        X = np.vstack(H.values)
        Y = np.ravel(V.target_csr.sum(axis=1))

        self.coeffs = scipy.optimize.minimize(
            loss, np.zeros(X.shape[1]), (X, Y), options={"disp": True}
        ) #, method='Nelder-Mead')
        print(f"fit loss {loss(self.coeffs.x, X, Y, 0)}")
        return self

    @functools.lru_cache(1)
    def transform(self, D):
        H = self.hawkes_model.transform(D, state_only=True)
        X = np.vstack(H.values)
        intensity = np.vstack(H.values) @ np.log(1 + np.exp(self.coeffs.x))

        if hasattr(D, "target_csr"):
            Y = np.ravel(D.target_csr.sum(axis=1))
            print(f"transform loss {loss(self.coeffs.x, X, Y, 0)}")

        return LowRankDataFrame(
            np.log(intensity)[:, None], np.ones(len(D.item_in_test))[:, None],
            index=H.index, columns=D.item_in_test.index, act='exp')


def loss(x, H, Y, alpha=1e-3):
    w = np.log(1 + np.exp(x))
    Lamb = H @ w
    loglik = Y * np.log(1e-10 + Lamb) - Lamb
    return -loglik.mean() + alpha/2*(w**2).sum()
