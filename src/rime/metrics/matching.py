import time, numba, functools
import numpy as np, pandas as pd, scipy as sp
from ..util import timed, _argsort


@numba.jit(nopython=True)
def _assign_sorted_numba(argsort_i, argsort_j, k_vec, c_vec, blocked):
    assigned = []
    for i, j in zip(argsort_i, argsort_j):
        if k_vec[i] > 0 and c_vec[j] > 0 and (i,j) not in blocked:
            k_vec[i] -= 1
            c_vec[j] -= 1
            assigned.append((i,j))
    return assigned


@timed("_assign_sorted")
def _assign_sorted(shape, k, c, argsort_ij, blocked={(-1, -1)}):
    k_vec = np.broadcast_to(k, (shape[0],)).astype(int)
    c_vec = np.broadcast_to(c, (shape[1],)).astype(int)
    blocked = set(blocked) if len(blocked) else {(-1, -1)}

    assigned = _assign_sorted_numba(*argsort_ij, k_vec, c_vec, blocked)
    i, j = np.asarray(assigned).reshape((-1, 2)).T
    csr = sp.sparse.coo_matrix((np.ones(len(i)), (i,j)), shape=shape).tocsr()
    return (csr, assigned)


def assign_mtch(score_mat, topk, C,
    argsort_ij=None, constraint_type='ub', device="cpu"):

    n_users, n_items = score_mat.shape

    if argsort_ij is None:
        argsort_ij = _argsort(score_mat, device=device)

    if constraint_type == 'ub':
        assigned_csr, _ = _assign_sorted((n_users, n_items), topk, C, argsort_ij)
    else: # lb
        if np.isscalar(C):
            min_total_recs = min(n_users * topk, C * n_items)
            min_k = min(topk, np.ceil(min_total_recs / n_users).astype(int))
            min_C = min(C,    np.ceil(min_total_recs / n_items).astype(int))
        else:
            assert np.broadcast_to(topk, (n_users,)).sum() > C.sum(), \
                "relative only on item_rec"
            min_k = np.round(C.sum() / n_users).astype(int)
            min_C = C
        min_csr, blocked = _assign_sorted((n_users, n_items), min_k, min_C, argsort_ij)

        if topk > min_k:
            k_vec = topk - np.ravel(min_csr.sum(axis=1))
            c_vec = n_users
        else:
            k_vec = n_items
            c_vec = C - np.ravel(min_csr.sum(axis=0))

        top_off, _ = _assign_sorted((n_users, n_items), k_vec, c_vec, argsort_ij, blocked)
        assigned_csr = min_csr + top_off

    return assigned_csr
