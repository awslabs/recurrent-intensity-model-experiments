import numpy as np, pandas as pd
import torch, dataclasses, functools
from typing import Dict, List


@dataclasses.dataclass(repr=False)
class ScoreExpression:
    def eval(self, device=None):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    def has_nan(self):
        raise NotImplementedError

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return np.prod(self.shape)


@dataclasses.dataclass(repr=False)
class LogLowRankValues(ScoreExpression):
    """ a log-low-rank matrix """
    ind_logits: List[list]
    col_logits: List[list]
    sign: float

    def eval(self, device=None):
        if device is None:
            return np.exp(self.ind_logits @ self.col_logits.T) * self.sign
        else:
            ind_logits = torch.as_tensor(self.ind_logits, device=device)
            col_logits = torch.as_tensor(self.col_logits, device=device)
            return (ind_logits @ col_logits.T).exp() * self.sign


    def has_nan(self):
        return np.isnan(self.ind_logits).any() or np.isnan(self.col_logits).any()


    @property
    def shape(self):
        return (len(self.ind_logits), len(self.col_logits))

    @property
    def num_hidden(self):
        return self.ind_logits.shape[1]


    def __getitem__(self, key):
        if np.isscalar(key):
            return _LogVectorMatrixValues(self.ind_logits[key], self.col_logits, self.sign)
        else:
            return self.__class__(self.ind_logits[key], self.col_logits, self.sign)


    @property
    def T(self):
        return self.__class__(self.col_logits, self.ind_logits, self.sign)


    def __mul__(self, other):
        if np.isscalar(other):
            log_abs = np.log(np.fabs(other))
            return self.__class__(
                np.pad(self.ind_logits, ((0,0), (0,1)), constant_values=log_abs),
                np.pad(self.col_logits, ((0,0), (0,1)), constant_values=1),
                self.sign * np.sign(other))
        else:
            warnings.warn(f"falling back to dense to multiply with {other.__class__}")
            return self.eval() * other


    def iter_batches(self, device=None, batch_size=1000):
        for i in range(0, len(self), batch_size):
            key = slice(i, min(i+batch_size, len(self)))
            yield key, self[key]


    def gpu_max(self, *args, **kw):
        m = float("-inf")
        for _, s in self.iter_batches(*args, **kw):
            m = max(m, s.eval().max())
        return m.item() if hasattr(m, "item") else m


@dataclasses.dataclass(repr=False)
class _LogVectorMatrixValues(ScoreExpression):
    vec_logits: list
    col_logits: List[list]
    sign: float

    @staticmethod
    def collate_fn(batch):
        ind_logits = []
        col_logits = batch[0].col_logits
        sign = batch[0].sign

        for elm in batch:
            ind_logits.append(elm.vec_logits)
            assert elm.col_logits is col_logits
            assert elm.sign == sign

        return LogLowRankValues(
            np.asarray(ind_logits), col_logits, sign,
        )


@dataclasses.dataclass(repr=False)
class LogLowRankDataFrame(LogLowRankValues):
    """ extends log-low-rank matrix with reindexing functions """
    index: list
    columns: list

    def __post_init__(self):
        assert self.shape == (len(self.index), len(self.columns)), \
            "values shape inconsistent with index, columns"


    @property
    def values(self):
        return LogLowRankValues(self.ind_logits, self.col_logits, self.sign)

    @property
    def T(self):
        return self.__class__(self.col_logits, self.ind_logits, self.sign,
            self.columns, self.index)


    def reindex(self, index, axis=0, fill_value=float("nan")):
        if axis==1:
            return self.T.reindex(index, fill_value=fill_value).T

        ind_logits = np.pad(self.ind_logits, ((0,1), (0,1)), constant_values=0)
        ind_logits[-1, -1] = np.log(fill_value)
        col_logits = np.pad(self.col_logits, ((0,0), (0,1)), constant_values=1)

        new_ind = pd.Series(
            np.arange(len(self)), index=self.index
            ).reindex(index, fill_value=-1).values

        return self.__class__(
            ind_logits[new_ind], col_logits, self.sign, index, self.columns)


    def eval(self):
        data = super().eval()
        return pd.DataFrame(data, index=self.index, columns=self.columns)


    def __mul__(self, other):
        """ left-join with other dataframes """
        if isinstance(other, self.__class__):
            other = other.reindex(self.index, fill_value=0) \
                         .reindex(self.columns, axis=1, fill_value=0)

            ind_logits = np.hstack([self.ind_logits, other.ind_logits])
            col_logits = np.hstack([self.col_logits, other.col_logits])
            return self.__class__(ind_logits, col_logits, self.sign * other.sign,
                self.index, self.columns)
        else:
            warnings.warn(f"falling back to dense to multiply with {other.__class__}")
            return self.eval() * other
