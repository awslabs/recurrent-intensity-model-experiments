import numpy as np, pandas as pd, scipy as sp
from ..util import perplexity, _assign_topk
from .matching import assign_mtch


def evaluate_assigned(target_csr, assigned_csr, score_mat=None, axis=None):
    """ compare targets and recommendation assignments on user-item matrix
    """
    hit = getattr(assigned_csr, "multiply", assigned_csr.__mul__)(target_csr)

    out = {
        'prec':     hit.sum() / assigned_csr.sum(),
        'item_cov': (assigned_csr.sum(axis=0)>0).mean(),  # 1 by n_items
        'item_ppl': perplexity(assigned_csr.sum(axis=0)),
        'user_cov': (assigned_csr.sum(axis=1)>0).mean(),  # n_users by 1
        'user_ppl': perplexity(assigned_csr.sum(axis=1)),
    }

    if score_mat is not None:
        obj_sum = getattr(assigned_csr, "multiply", assigned_csr.__mul__)(score_mat).sum()
        out['obj'] = obj_sum / assigned_csr.sum()

    if axis is not None:
        hit_axis = np.ravel(hit.sum(axis=axis))
        ideal = np.ravel(target_csr.sum(axis=axis))
        out['recall'] = (hit_axis / np.fmax(1, ideal)).mean()

    return out


def evaluate_item_rec(target_csr, score_mat, topk):
    assigned_csr = _assign_topk(score_mat, topk)
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=1)


def evaluate_user_rec(target_csr, score_mat, C):
    assigned_csr = _assign_topk(score_mat.T, C).T
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=0)


def evaluate_mtch(target_csr, score_mat, *args, **kw):
    assigned_csr = assign_mtch(target_csr, score_mat, *args, **kw)
    out = evaluate_assigned(target_csr, assigned_csr, score_mat)
    print('evaluate_mtch prec={prec:.1e} item_ppl={item_ppl:.1e}'.format(**out))
    return out
