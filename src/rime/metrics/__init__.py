import numpy as np, pandas as pd, scipy.sparse as sps, warnings, torch, operator
from torch.utils.data import DataLoader
from ..util import perplexity, _assign_topk, empty_cache_on_exit, score_op, LazyScoreBase
from .matching import assign_mtch
from .cvx import CVX


def _multiply(x, y):
    """ lazy, sparse, or numpy array """
    if isinstance(x, LazyScoreBase):
        return x * y
    elif isinstance(y, LazyScoreBase):
        return y * x
    elif sps.issparse(x):
        return x.multiply(y)
    elif sps.issparse(y):
        return y.multiply(x)
    else:  # numpy dense
        return x * y


def _sum(x, axis, device):
    if isinstance(x, LazyScoreBase):
        if axis is None:
            return float(score_op(x, "sum", device))
        elif axis == 0:
            return score_op(x, lambda x: torch.sum(x, axis), device, operator.add).cpu().numpy()
        else:  # axis == 1
            return score_op(x, lambda x: torch.sum(x, axis), device,
                            lambda x, y: torch.hstack([x, y])).cpu().numpy()
    else:
        return x.sum(axis)


@empty_cache_on_exit
def evaluate_assigned(target_csr, assigned_csr, score_mat=None, axis=None,
                      min_total_recs=0, device="cpu"):
    """ compare targets and recommendation assignments on user-item matrix

    target_csr: sparse or numpy array
    assigned_csr: sparse, LazyScoreBase, or numpy array
    score_mat: LazyScoreBase or numpy array
    axis: [None, 0, 1]
    """
    hit_axis = _sum(_multiply(target_csr, assigned_csr), axis, device)
    assigned_sum_0 = _sum(assigned_csr, 0, device)
    assigned_sum_1 = _sum(assigned_csr, 1, device)
    min_total_recs = max(min_total_recs, assigned_sum_0.sum())

    out = {
        'prec': np.sum(hit_axis) / min_total_recs,
        'recs/user': assigned_sum_1.mean(),
        'item_cov': (assigned_sum_0 > 0).mean(),  # 1 by n_items
        'item_ppl': perplexity(assigned_sum_0),
        'user_cov': (assigned_sum_1 > 0).mean(),  # n_users by 1
        'user_ppl': perplexity(assigned_sum_1),
    }

    if score_mat is not None:
        obj_sum = _sum(_multiply(score_mat, assigned_csr), None, device)
        out['obj_mean'] = float(obj_sum / min_total_recs)

    if axis is not None:
        ideal = np.ravel(target_csr.sum(axis=axis))
        out['recall'] = (hit_axis / np.fmax(1, ideal)).mean()

    return out


def evaluate_item_rec(target_csr, score_mat, topk, device="cpu", **kw):
    assigned_csr = _assign_topk(score_mat, topk, device=device, **kw)
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=1, device=device)


def evaluate_user_rec(target_csr, score_mat, C, device="cpu", **kw):
    assigned_csr = _assign_topk(score_mat.T, C, device=device, **kw).T
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=0, device=device)


def evaluate_mtch(target_csr, score_mat, topk, C, cvx=False, valid_mat=None,
                  relative=False, item_prior=None, constraint_type='ub', device="cpu", **kw):
    if relative:
        C = (C * np.asarray(item_prior) / np.mean(item_prior))

    if cvx:
        n_users, n_items = valid_mat.shape
        kw['alpha_ub'] = topk / n_items
        kw['beta_' + constraint_type] = C / n_users
        self = CVX(valid_mat, device=device, **kw)
        assigned_csr = self.fit(valid_mat).transform(score_mat)
    else:
        assigned_csr = assign_mtch(score_mat, topk, C, constraint_type, device=device, **kw)

    assigned_sum_0 = _sum(assigned_csr, 0, device)
    if assigned_sum_0.sum() == 0:
        warnings.warn("mtch should not return empty assignments unless in Rand")

    if constraint_type == 'ub':
        min_total_recs = np.mean(C) * score_mat.shape[1]
    else:
        min_total_recs = topk * score_mat.shape[0]
    out = evaluate_assigned(target_csr, assigned_csr, score_mat,
                            min_total_recs=min_total_recs, device=device)

    for pct in ['25%', '50%', '75%']:
        out[f'recs/item_{pct}'] = pd.Series(np.ravel(assigned_sum_0)).describe()[pct]

    print('evaluate_mtch prec@{topk}={prec:.1e} item_ppl@{mean_C}={item_ppl:.1e}'.format(
        **out, mean_C=np.mean(C), **locals()))
    return out
