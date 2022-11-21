# linprog
"""
```
max vec(p) @ vec(s)
s.t.
p @ 1 = 1
p = pT
```

Vectorized conditions
```
vec(p)[i * n + j] = p[i, j]
sum_j p[i, j] = vec(p) @ [0...0,1...1,0...0] = vec(p) @ (I * 1)
comm_mat @ vec(p) = vec(p)
```

Example:
>>> score_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
>>> score_mat = np.triu(score_mat, 1)
>>> score_mat = score_mat + score_mat.T
>>> assignments = LinProg(score_mat).fit(score_mat).transform(score_mat)
>>> print(np.round(assignments, 2))
array([[0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.]])
"""

import warnings
import numpy as np, scipy.optimize
import scipy.sparse as sps
from rime.util import auto_device, auto_cast_lazy_score, score_op


def comm_mat(m, n):
    """ wikipedia """
    # determine permutation applied by K
    w = np.arange(m * n).reshape((m, n), order='F').T.ravel(order='F')

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m * n)[w, :]


def triu_to_mat(n):
    """
    # upper
    array([[0, 1, 2],
           [1, 3, 4],
           [2, 4, 5]])
    """
    upper = np.cumsum(np.triu(np.ones((n, n), dtype=int))).reshape((n, n)) - 1
    upper[np.tril_indices(n, -1)] = 0
    upper = upper + np.triu(upper, 1).T

    i = np.arange(upper.size)
    j = upper.ravel()
    triu_to_mat = sps.coo_matrix(
        (np.ones(upper.size), (i, j)),
        shape=(upper.size, upper.max() + 1),
    )
    return triu_to_mat


def setup_linprog(scores, use_triu=True, noise_ratio=1e-3):
    if hasattr(scores, 'numpy'):
        scores = scores.numpy()
    scores = np.triu(scores, 1) + np.tril(scores, -1)
    if not np.allclose(scores, scores.T):
        warnings.warn("unsymmetric score matrix!")

    n = scores.shape[0]
    obj_max = np.ravel(scores)
    obj_min = -(obj_max + np.random.rand(*obj_max.shape) * noise_ratio)
    A_eq = sps.kron(sps.eye(n), np.ones((n, 1))).T.tocsr()
    b_eq = np.ones(n)
    if use_triu:
        cov = triu_to_mat(n)
        return obj_min @ cov, A_eq @ cov, b_eq
    else:
        A_sym = comm_mat(n, n) - np.eye(n * n)
        b_sym = np.zeros(n * n)
        return obj_min, np.vstack([A_eq.toarray(), A_sym]), np.hstack([b_eq, b_sym])


class LinProg:
    def __init__(self, score_mat, device=auto_device(), use_triu=True, decimals=2, solver_options={}):

        self.use_triu = use_triu
        self.decimals = decimals
        self.solver_options = solver_options
        score_mat = auto_cast_lazy_score(score_mat)
        self.score_max = float(score_op(score_mat, "max", device))
        self.score_min = float(score_op(score_mat, "min", device))
        print(f"entering LinProg score range=({self.score_min}, {self.score_max})")

    def transform(self, score_mat):
        if not hasattr(self, 'solution'):
            self.fit(score_mat)

        if self.use_triu:
            out = np.zeros(score_mat.shape)
            out[np.triu_indices(out.shape[0])] = self.solution.x
            out = out + np.triu(out, 1).T
        else:
            out = self.solution.x.reshape(score_mat.shape)

        if self.decimals is not None:
            out = np.round(out, self.decimals)
        if np.trace(out) > 0:
            warnings.warn(f"some users are unmatched. portion={np.trace(out)}")
        return out

    def fit(self, score_mat):
        obj_min, A_eq, b_eq = setup_linprog(score_mat, use_triu=self.use_triu)
        print(f'solving linprog with shapes {obj_min.shape}, {A_eq.shape}, {b_eq.shape}')
        self.solution = scipy.optimize.linprog(
            obj_min, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1),
            options={'tol': 1e-7, 'disp': True, **self.solver_options})
        return self
