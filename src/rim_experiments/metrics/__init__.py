import numpy as np, pandas as pd, scipy as sp, warnings
from scipy.sparse import issparse
from ..util import perplexity, _assign_topk
from .matching import assign_mtch
from .cvx import CVX


def _multiply(x, y):
    return x.multiply(y) if issparse(x) else y.multiply(x) if issparse(y) else x*y


def evaluate_assigned(target_csr, assigned_csr, score_mat=None, axis=None):
    """ compare targets and recommendation assignments on user-item matrix
    """
    hit = _multiply(target_csr, assigned_csr)

    out = {
        'prec':     hit.sum() / assigned_csr.sum(),
        'item_cov': (assigned_csr.sum(axis=0)>0).mean(),  # 1 by n_items
        'item_ppl': perplexity(assigned_csr.sum(axis=0)),
        'user_cov': (assigned_csr.sum(axis=1)>0).mean(),  # n_users by 1
        'user_ppl': perplexity(assigned_csr.sum(axis=1)),
    }

    if score_mat is not None:
        if hasattr(score_mat, "iter_batches"):
            obj_sum = np.sum([
                _multiply(assigned_csr[key], s.eval()).sum()
                for key, s in score_mat.iter_batches()
                ])
        else:
            obj_sum = _multiply(assigned_csr, score_mat).sum()

    if axis is not None:
        hit_axis = np.ravel(hit.sum(axis=axis))
        ideal = np.ravel(target_csr.sum(axis=axis))
        out['recall'] = (hit_axis / np.fmax(1, ideal)).mean()

    return out


def evaluate_item_rec(target_csr, score_mat, topk, **kw):
    assigned_csr = _assign_topk(score_mat, topk, **kw)
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=1)


def evaluate_user_rec(target_csr, score_mat, C, **kw):
    assigned_csr = _assign_topk(score_mat.T, C, **kw).T
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=0)


def evaluate_mtch(target_csr, score_mat, topk, C, cvx=False, valid_mat=None, **kw):
    if cvx:
        self = CVX(valid_mat, topk, C, **kw)
        assigned_csr = self.fit(valid_mat).transform(score_mat)
        if assigned_csr.sum() == 0:
            warnings.warn("cvx should not return empty assignments unless in Rand")
    else:
        assigned_csr = assign_mtch(score_mat, topk, C, **kw)
    out = evaluate_assigned(target_csr, assigned_csr, score_mat)
    print('evaluate_mtch prec@{topk}={prec:.1e} item_ppl@{C}={item_ppl:.1e}'.format(
        **out, **locals()))
    return out
