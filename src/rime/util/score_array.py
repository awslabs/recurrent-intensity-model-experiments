import numpy as np, pandas as pd
import torch, dataclasses, functools, warnings, operator, builtins
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

def df_to_coo(df):
    try:
        return df.sparse.to_coo()
    except KeyError:
        warnings.warn("pandas bug: https://github.com/pandas-dev/pandas/issues/25270")
        df = df.copy()
        df.index = list(range(len(df.index)))
        df.columns = list(range(len(df.columns)))
        return df.sparse.to_coo()

def sps_to_torch(x, device):
    coo = x.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device)

def auto_eval(c, device):
    if isinstance(c, ScoreExpression):
        return c.eval(device)
    elif np.isscalar(c):
        return c
    elif sps.issparse(c):
        return c.toarray() if device is None else sps_to_torch(c, device).to_dense()
    elif isinstance(c, pd.DataFrame):
        return auto_eval(df_to_coo(c))
    else:
        raise NotImplementedError(str(c))

def auto_values(c):
    if isinstance(c, ScoreExpression):
        return c.values
    elif np.isscalar(c) or sps.issparse(c):
        return c
    elif isinstance(c, pd.DataFrame):
        return df_to_coo(c).tocsr()
    else:
        raise NotImplementedError(str(c))

def auto_getitem(c, key):
    if isinstance(c, ScoreExpression) or sps.issparse(c):
        return c[key]
    elif np.isscalar(c):
        return c
    elif isinstance(c, pd.DataFrame):
        warnings.warn("dataframe getitem is inefficient; plz call values property first.")
        return c.iloc[key]
    else:
        raise NotImplementedError(str(c))

def auto_collate(c, D):
    if isinstance(c, ScoreExpression):
        return c.collate_fn(D)
    elif np.isscalar(c):
        return D[0]
    elif sps.issparse(c):
        return sps.vstack(D)
    elif isinstance(c, pd.DataFrame):
        warnings.warn('dataframe concat is inefficient; plz call values property first.')
        return pd.concat(D)
    else:
        raise NotImplementedError(str(c))


class ScoreExpression:
    """ Lazy evaluation for sparse and low-rank DataFrame by mini-batches.
    The base class also serves as tree root node after "+" and "*" operations;
    and interfaces with scalars and DataFrames during eval, slicing, and collate_fn.
    """
    def __init__(self, op, children):
        self.op = op
        self.children = children
        self.index = children[0].index
        self.columns = children[0].columns
        self.shape = children[0].shape

    def eval(self, device=None):
        children = [auto_eval(c, device) for c in self.children]
        return self.op(*children)

    @property
    def values(self):
        children = [auto_values(c) for c in self.children]
        return self.__class__(self.op, children)

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def batch_size(self):
        return get_batch_size(self.shape)

    def _check_index_columns(self, other):
        if self.index is not None and hasattr(other, 'index'):
            assert (self.index == other.index).all(), \
                "please reindex index before binary operations"
        if self.columns is not None and hasattr(other, 'columns'):
            assert (self.columns == other.columns).all(), \
                "please reindex columns before binary operations"

    def __add__(self, other):
        self._check_index_columns(other)
        return ScoreExpression(operator.add, [self, other])

    def __mul__(self, other):
        self._check_index_columns(other)
        return ScoreExpression(operator.mul, [self, other])

    @property
    def T(self):
        children = [c if np.isscalar(c) else c.T for c in self.children]
        return self.__class__(self.op, children)

    def __getitem__(self, key):
        """ used in pytorch dataloader. ignores index / columns """
        if np.isscalar(key):
            key = [key]
        children = [auto_getitem(c, key) for c in self.children]
        return self.__class__(self.op, children)

    @classmethod
    def collate_fn(cls, batch):
        self = batch[0]
        op = self.op
        data = zip(*[b.children for b in batch])
        children = [auto_collate(c, D) for c, D in zip(self.children, data)]
        return cls(op, children)


@dataclasses.dataclass(repr=False)
class LowRankDataFrame(ScoreExpression):
    """ mimics a pandas dataframe with exponentiated low-rank structures
    """
    ind_logits: List[list]
    col_logits: List[list]
    index: list
    columns: list
    act: str

    def __post_init__(self):
        assert self.ind_logits.shape[1] == self.col_logits.shape[1], "check hidden"
        assert self.ind_logits.shape[0] == len(self.index), "check index"
        assert self.col_logits.shape[0] == len(self.columns), "check columns"
        assert self.act in ['exp', 'sigmoid'], "requires nonnegative act to solve cvx"

    def eval(self, device=None):
        if device is None:
            z = self.ind_logits @ self.col_logits.T

            if self.act == 'exp':
                return np.exp(z)
            elif self.act == 'sigmoid':
                return 1./(1+np.exp(-z))
        else:
            ind_logits = torch.as_tensor(self.ind_logits, device=device)
            col_logits = torch.as_tensor(self.col_logits, device=device)
            z = ind_logits @ col_logits.T

            if self.act == 'exp':
                return z.exp()
            elif self.act == 'sigmoid':
                return z.sigmoid()
    @property
    def values(self):
        return self

    @property
    def shape(self):
        return (len(self.ind_logits), len(self.col_logits))

    def __getitem__(self, key):
        if np.isscalar(key):
            key = [key]
        return self.__class__(self.ind_logits[key], self.col_logits,
            self.index[key], self.columns, self.act)

    @property
    def T(self):
        return self.__class__(self.col_logits, self.ind_logits,
            self.columns, self.index, self.act)

    @classmethod
    def collate_fn(cls, batch):
        ind_logits = []
        col_logits = batch[0].col_logits
        index = []
        columns = batch[0].columns
        act = batch[0].act

        for elm in batch:
            ind_logits.append(elm.ind_logits)
            index.extend(elm.index)

        return cls(np.vstack(ind_logits), col_logits, index, columns, act)

    def reindex(self, index, axis=0, fill_value=float("nan")):
        if axis==1:
            return self.T.reindex(index, fill_value=fill_value).T

        ind_logits = np.pad(self.ind_logits, ((0,1), (0,1)), constant_values=0)
        with np.errstate(divide='ignore'): # 0 -> -inf
            ind_logits[-1, -1] = np.log(fill_value)
        col_logits = np.pad(self.col_logits, ((0,0), (0,1)), constant_values=1)

        new_ind = pd.Series(
            np.arange(len(self)), index=self.index
            ).reindex(index, fill_value=-1).values

        return self.__class__(
            ind_logits[new_ind], col_logits, index, self.columns, self.act)


def score_op(S, op, device=None):
    out = None
    for batch in DataLoader(S, S.batch_size, collate_fn=S.collate_fn):
        val = batch.eval(device)
        new = getattr(val, op)()
        out = new if out is None else getattr(builtins, op)(out, new)
    return out
