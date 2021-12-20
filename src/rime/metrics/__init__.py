import numpy as np, pandas as pd, warnings
from scipy.sparse import issparse
from torch.utils.data import DataLoader
from ..util import perplexity, _assign_topk, empty_cache_on_exit, score_op
from .matching import assign_mtch
from .cvx import CVX


def _multiply(x, y):
    return x.multiply(y) if issparse(x) else y.multiply(x) if issparse(y) else x * y


@empty_cache_on_exit
def evaluate_assigned(target_csr, assigned_csr, score_mat=None, axis=None,
                      min_total_recs=0, device=None):
    """ compare targets and recommendation assignments on user-item matrix
    """
    hit = _multiply(target_csr, assigned_csr)
    min_total_recs = max(min_total_recs, assigned_csr.sum())

    out = {
        'prec': hit.sum() / min_total_recs,
        'recs/user': assigned_csr.sum() / assigned_csr.shape[0],
        'item_cov': (assigned_csr.sum(axis=0) > 0).mean(),  # 1 by n_items
        'item_ppl': perplexity(assigned_csr.sum(axis=0)),
        'user_cov': (assigned_csr.sum(axis=1) > 0).mean(),  # n_users by 1
        'user_ppl': perplexity(assigned_csr.sum(axis=1)),
    }

    if score_mat is not None:
        if hasattr(score_mat, "collate_fn"):
            obj_sum = float(score_op(score_mat * assigned_csr, "sum", device))
        else:
            obj_sum = _multiply(assigned_csr, score_mat).sum()
        out['obj_mean'] = float(obj_sum / min_total_recs)

    if axis is not None:
        hit_axis = np.ravel(hit.sum(axis=axis))
        ideal = np.ravel(target_csr.sum(axis=axis))
        out['recall'] = (hit_axis / np.fmax(1, ideal)).mean()

    return out


def evaluate_item_rec(target_csr, score_mat, topk, device=None, **kw):
    assigned_csr = _assign_topk(score_mat, topk, device=device, **kw)
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=1, device=device)


def evaluate_user_rec(target_csr, score_mat, C, device=None, **kw):
    assigned_csr = _assign_topk(score_mat.T, C, device=device, **kw).T
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=0, device=device)


def evaluate_mtch(target_csr, score_mat, topk, C, cvx=False, valid_mat=None,
                  relative=False, item_prior=None, constraint_type='ub', device=None, **kw):
    if relative:
        C = (C * np.asarray(item_prior) / np.mean(item_prior))

    if cvx:
        self = CVX(valid_mat, topk, C, constraint_type, device=device, **kw)
        assigned_csr = self.fit(valid_mat).transform(score_mat)
        if assigned_csr.sum() == 0:
            warnings.warn("cvx should not return empty assignments unless in Rand")
    else:
        assigned_csr = assign_mtch(score_mat, topk, C, constraint_type, device=device, **kw)

    if constraint_type == 'ub':
        min_total_recs = np.mean(C) * score_mat.shape[1]
    else:
        min_total_recs = topk * score_mat.shape[0]
    out = evaluate_assigned(target_csr, assigned_csr, score_mat,
                            min_total_recs=min_total_recs, device=device)

    for pct in ['25%', '50%', '75%']:
        out[f'recs/item_{pct}'] = pd.Series(np.ravel(assigned_csr.sum(0))).describe()[pct]

    print('evaluate_mtch prec@{topk}={prec:.1e} item_ppl@{mean_C}={item_ppl:.1e}'.format(
        **out, mean_C=np.mean(C), **locals()))
    return out
