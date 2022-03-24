import numpy as np, pandas as pd
import torch, dataclasses, warnings, operator, builtins, numbers, os, itertools, functools
from typing import List, ClassVar
from torch.utils.data import DataLoader
import scipy.sparse as sps


def get_batch_size(shape, frac=float(os.environ.get("BATCH_SIZE_FRAC", 0.1))):
    """ round to similar batch sizes """
    n_users, n_items = shape
    if torch.cuda.device_count():
        total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        total_memory = 16e9
    max_batch_size = total_memory / 8 / n_items * frac
    n_batches = int(n_users / max_batch_size) + 1
    return int(np.ceil(n_users / n_batches))


def find_iloc(old_index, new_index, allow_missing=False):
    iloc = pd.Series(
        np.arange(len(old_index)), index=old_index
    ).reindex(new_index, fill_value=-1).values
    if not allow_missing and -1 in iloc:
        raise IndexError("missing indices detected in a disallowed case")
    return iloc


def matrix_reindex(csr, old_index, new_index, axis, fill_value=0):
    """ pandas.reindex functionality on sparse or dense matrices as well as 1d arrays """
    if axis == 1:
        return matrix_reindex(csr.T, old_index, new_index, 0, fill_value).T.copy()
    assert axis == 0, "axis must be 0 or 1"
    assert csr.shape[0] == len(old_index), "shape must match between csr and old_index"

    if isinstance(csr, LazyScoreBase):
        pass  # does not support extrapolation
    elif sps.issparse(csr):
        csr = sps.vstack([csr, csr[:1] * 0 + fill_value], "csr")
        csr.eliminate_zeros()
    else:
        csr = np.concatenate([csr, csr[:1] * 0 + fill_value], axis=0)

    iloc = find_iloc(old_index, new_index, allow_missing=True)
    return csr[iloc]


def sps_to_torch(x, device):
    """ convert scipy.sparse to torch.sparse """
    coo = x.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device)


def auto_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def auto_tensor(x, device=None):
    if device is None:
        device = auto_device()
    if hasattr(x, "as_tensor"):
        return x.as_tensor(device)
    elif sps.issparse(x):
        return sps_to_torch(x, device).to_dense()
    else:
        return torch.as_tensor(x, device)


class LazyScoreBase:
    """ low-rank, sparse, and other expressions with deferred evaluations after subsampling

    * Support expressions like (a @ b.T + c).exp() * d + e
    * Support operations T, __getitem__, collate_fn
    * Lazy evaluation until as_tensor(device) or numpy()
    Drop support for reindex because it is nontrivial to find fill_values through activation.
    """
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f"<{type(self).__name__} {self.shape}>"

    def numpy(self):
        raise NotImplementedError

    def as_tensor(self, device=None):
        raise NotImplementedError

    @property
    def T(self):
        raise NotImplementedError

    def __getitem__(self, key):
        """ slice by rows for pytorch dataloaders """
        raise NotImplementedError

    @classmethod
    def collate_fn(cls, D):
        """ collate by rows for pytorch dataloaders """
        raise NotImplementedError

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def batch_size(self):
        return get_batch_size(self.shape)

    def __matmul__(self, other):
        return MatMulExpression(self, other)

    def __add__(self, other):
        return ElementWiseExpression(operator.add, [self, other])

    def __mul__(self, other):
        return ElementWiseExpression(operator.mul, [self, other])

    def __sub__(self, other):
        return ElementWiseExpression(operator.sub, [self, other])

    def __truediv__(self, other):
        return ElementWiseExpression(operator.truediv, [self, other])

    def exp(self):
        return ElementWiseExpression(torch.exp, [self])

    def softplus(self):
        return ElementWiseExpression(torch.nn.functional.softplus, [self])

    def sigmoid(self):
        return ElementWiseExpression(torch.sigmoid, [self])


def auto_cast_lazy_score(other):
    if other is None:
        return None  # prior_score=None -> None
    elif isinstance(other, LazyScoreBase):
        return other
    elif sps.issparse(other):
        return LazySparseMatrix(other)
    elif isinstance(other, pd.DataFrame):
        return LazyDenseMatrix(other.values)
    elif torch.is_tensor(other):
        return LazyDenseMatrix(other.detach().cpu().numpy())
    else:  # scalar or numpy arrays of 0, 1, 2 dimensions
        return LazyDenseMatrix(other)


class LazySparseMatrix(LazyScoreBase):
    def __init__(self, c):
        self.c = c.tocsr()
        self.shape = c.shape

    def numpy(self):
        return self.c.toarray()

    def as_tensor(self, device=None):
        return sps_to_torch(self.c, device).to_dense()

    @property
    def T(self):
        return self.__class__(self.c.T)

    def __getitem__(self, key):
        if np.isscalar(key):
            slc = slice(self.c.indptr[key], self.c.indptr[key + 1])
            _dict = {
                "values": self.c.data[slc],
                "keys": self.c.indices[slc],
                "shape": self.c.shape[1],
            }
            return _LazySparseDictFast(_dict)
        else:
            return self.__class__(self.c[key])

    @classmethod
    def collate_fn(cls, D):
        return cls(sps.vstack([d.c for d in D]))


class _LazySparseDictFast(LazyScoreBase):
    def __init__(self, c):
        self.c = c
        self.shape = (1, self.c['shape'])

    @classmethod
    def collate_fn(cls, D):
        C = [d.c for d in D]
        csr = sps.csr_matrix((
            np.hstack([c['values'] for c in C]),  # data
            np.hstack([c['keys'] for c in C]),  # indices
            np.hstack([[0], np.cumsum([len(c['keys']) for c in C])]),  # indptr
        ), shape=(len(C), C[0]['shape']))
        return LazySparseMatrix(csr)


class LazyDenseMatrix(LazyScoreBase):
    """ cast scalar and arrays to 2-d arrays """
    def __init__(self, c):
        self.c = np.array(c, ndmin=2)
        self.shape = self.c.shape

    def numpy(self):
        return self.c

    def as_tensor(self, device=None):
        return torch.as_tensor(self.c, device=device)

    @property
    def T(self):
        return self.__class__(self.c.T)

    def __getitem__(self, key):
        """ take modulos for broadcasting purposes """
        if isinstance(key, slice):
            key = range(key.stop)[key]
        key = np.array(key, ndmin=1) % self.shape[0]
        return self.__class__(self.c[key])

    @classmethod
    def collate_fn(cls, D):
        return cls(np.vstack([d.c for d in D]))


class LazyExpressionBase:
    def __init__(self, op, children):
        self.op = op
        self.children = [auto_cast_lazy_score(c) for c in children]

    def traverse(self):
        builder = ""
        for i, c in enumerate(self.children):
            if hasattr(c, "traverse"):
                builder = builder + f"({c.traverse()})"
            else:
                builder = builder + f"{c}"
            if i == 0:
                builder = builder + f" {self.op.__name__} "
        return builder

    def __repr__(self):
        return self.traverse()

    def as_tensor(self, device=None):
        children = [c.as_tensor(device) for c in self.children]
        return self.op(*children)

    def numpy(self):
        try:
            return self.op(*[c.numpy() for c in self.children])
        except Exception:  # torch-only operation
            return self.op(*[c.as_tensor() for c in self.children]).numpy()


class ElementWiseExpression(LazyExpressionBase, LazyScoreBase):
    """ Tree representation of an element-wise expression; auto-broadcast """
    def __init__(self, op, children):
        super().__init__(op, children)
        shape = np.transpose([c.shape for c in self.children])
        self.shape = (max(shape[0]), max(shape[1]))  # consider broadcast

    @property
    def T(self):
        children = [c.T for c in self.children]
        return self.__class__(self.op, children)

    def __getitem__(self, key):
        children = [c[key] for c in self.children]
        return self.__class__(self.op, children)

    @classmethod
    def collate_fn(cls, batch):
        op, template = batch[0].op, batch[0].children
        data = zip(*[b.children for b in batch])
        children = [c.collate_fn(D) for c, D in zip(template, data)]
        return cls(op, children)


class MatMulExpression(LazyExpressionBase, LazyScoreBase):
    def __init__(self, left, right):
        assert left.shape[1] == right.shape[0], \
            f"matmul shape check fail: {left.shape} vs {right.shape}"
        super().__init__(operator.matmul, [left, right])
        self.left = self.children[0]
        self.right = self.children[1]
        self.shape = (self.left.shape[0], self.right.shape[1])

    @property
    def T(self):
        return self.__class__(self.right.T, self.left.T)

    def __getitem__(self, key):
        return self.__class__(self.left[key], self.right)

    @classmethod
    def collate_fn(cls, batch):
        c = batch[0].left.__class__
        left = c.collate_fn([b.left for b in batch])
        return cls(left, batch[0].right)


@dataclasses.dataclass(repr=False)
class RandScore(LazyScoreBase):
    """ add random noise to break ties """
    row_seeds: list  # np.array for fast indexing
    col_seeds: list  # np.array for fast indexing
    _DEFAULT_MAX_SEED: ClassVar[int] = 10000

    @property
    def shape(self):
        return (len(self.row_seeds), len(self.col_seeds))

    @classmethod
    def create(cls, shape):
        return cls(np.random.choice(cls._DEFAULT_MAX_SEED) + np.arange(shape[0]),
                   np.random.choice(cls._DEFAULT_MAX_SEED) + np.arange(shape[1]))

    def numpy(self):
        return np.vstack([
            np.random.RandomState(int(s)).rand(len(self.col_seeds))
            for s in self.row_seeds])

    def as_tensor(self, device=None):
        rows = []
        for s in self.row_seeds:
            generator = torch.Generator(device).manual_seed(int(s))
            new = torch.rand(len(self.col_seeds), device=device, generator=generator)
            rows.append(new)
        return torch.vstack(rows)

    @property
    def T(self):
        warnings.warn("transpose changes rand seed; only for evaluate_user_rec")
        return self.__class__(self.col_seeds, self.row_seeds)

    def __getitem__(self, key):
        if np.isscalar(key):
            key = [key]
        row_seeds = self.row_seeds[key]
        return self.__class__(row_seeds, self.col_seeds)

    @classmethod
    def collate_fn(cls, batch):
        return cls(np.hstack([b.row_seeds for b in batch]), batch[0].col_seeds)


def batch_op_iter(S, op, device=None):
    if isinstance(op, str):
        op = getattr(torch, op)  # max, min, sum
    for i in range(0, len(S), S.batch_size):
        batch = S[i:min(len(S), i + S.batch_size)]
        val = batch.as_tensor(device)
        yield op(val)


def score_op(S, op, device=None, reduce_fn=None):
    """ aggregation operations (e.g., max, min, sum) across entire matrix """
    if reduce_fn is None:
        reduce_fn = {"max": max, "min": min, "sum": operator.add}[op]
    iterable = batch_op_iter(S, op, device)
    return functools.reduce(reduce_fn, iterable)
