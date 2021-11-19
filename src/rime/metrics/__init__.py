import numpy as np, pandas as pd, scipy as sp, warnings
from scipy.sparse import issparse
from torch.utils.data import DataLoader
from ..util import perplexity, _assign_topk
from .matching import assign_mtch
from .cvx import CVX


def _multiply(x, y):
    return x.multiply(y) if issparse(x) else y.multiply(x) if issparse(y) else x*y


def _multiply_sum_by_batches(x, s):
    bsz = s.batch_size
    return np.sum([
        _multiply(x[i*bsz : min((i+1)*bsz, x.shape[0])], batch.eval()).sum()
        for i, batch in enumerate(DataLoader(s, bsz, collate_fn=s.collate_fn))
        ])


def evaluate_assigned(target_csr, assigned_csr, score_mat=None, axis=None,
    min_total_recs=0):
    """ compare targets and recommendation assignments on user-item matrix
    """
    hit = _multiply(target_csr, assigned_csr)
    min_total_recs = max(min_total_recs, assigned_csr.sum())

    out = {
        'prec':     hit.sum() / min_total_recs,
        'recs/user': assigned_csr.sum() / assigned_csr.shape[0],
        'item_cov': (assigned_csr.sum(axis=0)>0).mean(),  # 1 by n_items
        'item_ppl': perplexity(assigned_csr.sum(axis=0)),
        'user_cov': (assigned_csr.sum(axis=1)>0).mean(),  # n_users by 1
        'user_ppl': perplexity(assigned_csr.sum(axis=1)),
    }

    if score_mat is not None:
        if hasattr(score_mat, "collate_fn"):
            obj_sum = _multiply_sum_by_batches(assigned_csr, score_mat)
        else:
            obj_sum = _multiply(assigned_csr, score_mat).sum()
        out['obj_mean'] = float(obj_sum / min_total_recs)

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


def evaluate_mtch(target_csr, score_mat, topk, C, cvx=False, valid_mat=None,
    relative=False, item_prior=None, constraint_type='ub', **kw):
    if relative:
        C = (C * np.asarray(item_prior) / np.mean(item_prior))

    if cvx:
        self = CVX(valid_mat, topk, C, constraint_type, **kw)
        assigned_csr = self.fit(valid_mat).transform(score_mat)
        if assigned_csr.sum() == 0:
            warnings.warn("cvx should not return empty assignments unless in Rand")
    else:
        assigned_csr = assign_mtch(score_mat, topk, C, constraint_type, **kw)

    if constraint_type == 'ub':
        min_total_recs = np.mean(C) * score_mat.shape[1]
    else:
        min_total_recs = topk * score_mat.shape[0]
    out = evaluate_assigned(target_csr, assigned_csr, score_mat,
        min_total_recs=min_total_recs)

    print('evaluate_mtch prec@{topk}={prec:.1e} item_ppl@{mean_C}={item_ppl:.1e}'.format(
        **out, mean_C=np.mean(C), **locals()))
    return out
