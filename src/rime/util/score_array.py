import numpy as np, pandas as pd
import torch, dataclasses, functools, warnings, operator, builtins, numbers
from typing import Dict, List
from torch.utils.data import DataLoader
import scipy.sparse as sps

def get_batch_size(shape, frac=0.1):
    """ round to similar batch sizes """
    n_users, n_items = shape
    if torch.cuda.device_count():
        total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        total_memory = 16e9
    max_batch_size = total_memory / 8 / n_items * frac
    n_batches = int(n_users / max_batch_size) + 1
    return int(np.ceil(n_users / n_batches))


def matrix_reindex(csr, old_index, new_index, axis, fill_value=0):
    """ pandas.reindex functionality on sparse or dense matrices """
    if axis==1:
        return matrix_reindex(csr.T, old_index, new_index, 0, fill_value).T.copy()
    assert csr.shape[0] == len(old_index), "shape must match between csr and old_index"

    if sps.issparse(csr):
        csr = sps.vstack([csr, csr[:1] * 0 + fill_value], "csr")
        csr.eliminate_zeros()
    else:
        csr = np.vstack([csr, csr[:1] * 0 + fill_value])

    iloc = pd.Series(
            np.arange(len(old_index)), index=old_index
            ).reindex(new_index, fill_value=-1).values
    return csr[iloc].copy()


def sps_to_torch(x, device):
    """ convert scipy.sparse to torch.sparse """
    coo = x.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device)


class LazyScoreBase:
    """ Lazy element-wise A*B+C for sparse and low-rank matrices.

    The base class wraps over scalar, scipy.sparse, and numpy dense.
    Methods to overload include: eval, T, __getitem__, collate_fn.
    Method `reindex` is only supported in the derived LowRankDataFrame subclass.
    """

    def __init__(self, c):
        self._type = 'scalar' if isinstance(c, numbers.Number) else \
                     'sparse' if sps.issparse(c) else \
                     'dense' if np.ndim(c) == 2 else \
                     None
        assert self._type is not None, f"type {type(c)} is not supported"
        self.c = c if self._type != 'sparse' else c.tocsr()
        self.shape = c.shape if self._type != 'scalar' else None

    # methods to overload

    def eval(self, device):
        """ LazyScoreBase -> scalar, numpy (device is None), or torch (device) """
        if self._type == 'scalar':
            return self.c
        elif self._type == 'sparse':
            return self.c.toarray() if device is None else \
                   sps_to_torch(self.c, device).to_dense()
        elif self._type == 'dense':
            return self.c if device is None else \
                   torch.as_tensor(self.c, device=device)

    @property
    def T(self):
        """ LazyScoreBase -> LazyScoreBase(transposed) """
        cT = self.c if self._type == 'scalar' else self.c.T
        return self.__class__(cT)

    def __getitem__(self, key):
        """ LazyScoreBase -> LazyScoreBase(sub-rows); used in pytorch dataloader """
        if np.isscalar(key):
            key = [key]

        if self._type == 'scalar':
            return self.__class__(self.c)
        else:
            return self.__class__(self.c[key])

    def collate_fn(self, D):
        """ List[LazyScoreBase] -> LazyScoreBase; used in pytorch dataloader """
        C = [d.c for d in D]
        if self._type == 'scalar':
            return self.__class__(C[0])
        elif self._type == 'sparse':
            return self.__class__(sps.vstack(C))
        elif self._type == 'dense':
            return self.__class__(np.vstack(C))

    # methods to inherit

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def batch_size(self):
        return get_batch_size(self.shape)

    def _wrap_and_check(self, other):
        if not isinstance(other, LazyScoreBase):
            other = LazyScoreBase(other)
        if self.shape is not None and other.shape is not None:
            assert np.allclose(self.shape, other.shape), "shape must be compatible"
        return other

    def __add__(self, other):
        other = self._wrap_and_check(other)
        return LazyScoreExpression(operator.add, [self, other])

    def __mul__(self, other):
        other = self._wrap_and_check(other)
        return LazyScoreExpression(operator.mul, [self, other])


class LazyScoreExpression(LazyScoreBase):
    """ Tree representation of score expression until final eval """
    def __init__(self, op, children):
        self.op = op
        self.children = children
        for c in children:
            assert isinstance(c, LazyScoreBase), f"please wrap {c} in LazyScoreBase"
        self.shape = children[0].shape

    def eval(self, device=None):
        children = [c.eval(device) for c in self.children]
        return self.op(*children)

    @property
    def T(self):
        children = [c.T for c in self.children]
        return self.__class__(self.op, children)

    def __getitem__(self, key):
        children = [c[key] for c in self.children]
        return self.__class__(self.op, children)

    @classmethod
    def collate_fn(cls, batch):
        first = batch[0]
        data = zip(*[b.children for b in batch])
        children = [c.collate_fn(D) for c, D in zip(first.children, data)]
        return cls(first.op, children)


@dataclasses.dataclass(repr=False)
class LowRankDataFrame(LazyScoreBase):
    """ mimics a pandas dataframe with exponentiated low-rank structures
    """
    ind_logits: List[list]
    col_logits: List[list]
    index: list
    columns: list
    act: str
    ind_default: list = None
    col_default: list = None

    def __post_init__(self):
        if self.ind_default is None:
            self.ind_default = np.zeros_like(self.ind_logits[0])
        if self.col_default is None:
            self.col_default = np.zeros_like(self.col_logits[0])

        assert self.ind_logits.shape[1] == self.col_logits.shape[1], "check hidden"
        assert self.ind_logits.shape[0] == len(self.index), "check index"
        assert self.col_logits.shape[0] == len(self.columns), "check columns"
        assert self.act in ['exp', 'sigmoid'], "requires nonnegative act to solve cvx"

    def eval(self, device=None):
        if device is None:
            z = self.ind_logits @ self.col_logits.T
            assert not np.isnan(z).any(), "low rank score must be valid"

            if self.act == 'exp':
                return np.exp(z)
            elif self.act == 'sigmoid':
                return 1./(1+np.exp(-z))
        else:
            ind_logits = torch.as_tensor(self.ind_logits, device=device)
            col_logits = torch.as_tensor(self.col_logits, device=device)
            z = ind_logits @ col_logits.T
            assert not torch.isnan(z).any(), "low rank score must be valid"

            if self.act == 'exp':
                return z.exp()
            elif self.act == 'sigmoid':
                return z.sigmoid()

    @property
    def shape(self):
        return (len(self.ind_logits), len(self.col_logits))

    def __getitem__(self, key):
        if np.isscalar(key):
            key = [key]
        return self.__class__(self.ind_logits[key], self.col_logits,
            np.asarray(self.index)[key], self.columns, self.act,
            self.ind_default, self.col_default)

    @property
    def T(self):
        return self.__class__(self.col_logits, self.ind_logits,
            self.columns, self.index, self.act, self.col_default, self.ind_default)

    @classmethod
    def collate_fn(cls, batch):
        first = batch[0]
        ind_logits = []
        index = []

        for elm in batch:
            ind_logits.append(elm.ind_logits)
            index.extend(elm.index)

        return cls(np.vstack(ind_logits), first.col_logits, index,
            first.columns, first.act, first.ind_default, first.col_default)

    # new method only for this class

    def reindex(self, index, axis=0, fill_value=float("nan")):
        """ reindex with new hidden dim to express fill_value(0) as act(-inf * 1) """
        if axis==1:
            return self.T.reindex(index, fill_value=fill_value).T

        ind_logits = np.vstack([self.ind_logits, self.ind_default])
        ind_logits = np.hstack([ind_logits, np.zeros_like(ind_logits[:, :1])])
        ind_default = np.hstack([self.ind_default, np.zeros_like(self.ind_default[:1])])

        if fill_value == 0:
            ind_logits[-1, -1] = float("-inf")    # common for exp and sigmoid
        elif self.act == 'exp':
            ind_logits[-1, -1] = np.log(fill_value)
        elif self.act == 'sigmoid':
            ind_logits[-1, -1] = np.log(fill_value) - np.log(1-fill_value)

        col_logits = np.hstack([self.col_logits, np.ones_like(self.col_logits[:, :1])])
        col_default = np.hstack([self.col_default, np.ones_like(self.col_default[:1])])

        new_ind = pd.Series(
            np.arange(len(self)), index=self.index
            ).reindex(index, fill_value=-1).values

        return self.__class__(
            ind_logits[new_ind], col_logits, index, self.columns, self.act,
            ind_default, col_default)


def score_op(S, op, device=None):
    """ aggregation operations (e.g., max, min) across entire matrix """
    out = None
    for batch in DataLoader(S, S.batch_size, collate_fn=S.collate_fn):
        val = batch.eval(device)
        new = getattr(val, op)()
        out = new if out is None else getattr(builtins, op)(out, new)
    return out
