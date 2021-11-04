import pandas as pd, numpy as np, scipy as sp
import functools, collections, time, contextlib, os, torch, gc, warnings
from torch.utils.data import DataLoader
from datetime import datetime
from pytorch_lightning import LightningModule
from backports.cached_property import cached_property
from .score_array import *


class timed(contextlib.ContextDecorator):
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.tic = time.time()
        print("entering", self.name)

    def __exit__(self, *args, **kw):
        print("exiting", self.name, "time {:.1f}s".format(time.time() - self.tic))


def warn_nan_output(func):
    @functools.wraps(func)
    def wrapped(*args, **kw):
        out = func(*args, **kw)
        values = getattr(out, "values", out)

        if hasattr(values, "has_nan"):
            has_nan = values.has_nan()
        else:
            has_nan = np.isnan(values).any()

        if has_nan:
            warnings.warn(f"{func.__name__} output contains NaN", stacklevel=2)
        return out
    return wrapped


def _empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _get_cuda_objs():
    objs = []
    for obj in gc.get_objects():
        try:
            flag = torch.is_tensor(obj) # or \
                # (hasattr(obj, 'data') and torch.is_tensor(obj.data))
        except Exception:
            flag = False
        if flag and torch.device(obj.device) != torch.device("cpu"):
            objs.append(obj)
    return objs


def empty_cache_on_exit(func):
    @functools.wraps(func)
    def wrapped(*args, **kw):
        _empty_cache()
        start_list = _get_cuda_objs()

        tic = time.time()
        out = func(*args, **kw)
        print(func.__name__, "time {:.1f}s".format(time.time() - tic))

        _empty_cache()
        end_list = _get_cuda_objs()
        for obj in set(end_list) - set(start_list):
            print(func.__name__, "memory leak",
                type(obj), obj.size(), obj.device, flush=True)

        del start_list
        del end_list
        _empty_cache()
        return out
    return wrapped


def perplexity(x):
    x = np.ravel(x) / x.sum()
    return float(np.exp(- x @ np.log(np.where(x>0, x, 1e-10))))


@empty_cache_on_exit
def _assign_topk(S, k, tie_breaker=1e-10, device="cpu"):
    indices = []
    if hasattr(S, "collate_fn"):
        batches = DataLoader(S, batch_size=S.batch_size, collate_fn=S.collate_fn)
    else:
        batches = [S]
    for s in batches:
        if hasattr(s, "eval"):
            s = s.eval(device)
        else:
            s = torch.tensor(s, device=device)
        if tie_breaker:
            s = s + torch.rand(*s.shape, device=device) * tie_breaker
        s_topk = s.topk(k).indices.cpu().numpy()
        indices.append(s_topk)
    indices = np.vstack(indices)

    return sp.sparse.csr_matrix((
        np.ones(indices.size),
        np.ravel(indices),
        np.arange(0, indices.size+1, indices.shape[1]),
    ), shape=S.shape)


@empty_cache_on_exit
def _argsort(S, tie_breaker=1e-10, device="cpu"):
    print(f"_argsort {S.size:,} scores on device {device}; ", end="")
    if hasattr(S, "batch_size") and S.batch_size < S.shape[0]:
        warnings.warn(f"switching numpy.argsort due to {S.batch_size}<{S.shape[0]}")
        device = None

    if hasattr(S, "eval"):
        S = S.eval(device)

    shape = S.shape

    if device is None:
        if tie_breaker>0:
            S = S + np.random.rand(*S.shape) * tie_breaker
        S = -S.reshape(-1)
        _empty_cache()
        argsort_ind = np.argsort(S)
    else:
        S = torch.as_tensor(S, device=device)
        if tie_breaker>0:
            S = S + torch.rand(*S.shape, device=device) * tie_breaker
        S = -S.reshape(-1)
        _empty_cache()
        argsort_ind = torch.argsort(S).cpu().numpy()

    return np.unravel_index(argsort_ind, shape)


def extract_user_item(event_df):
    user_df = event_df.groupby('USER_ID').agg(
        _Tmin=("TIMESTAMP", min), _Tmax=("TIMESTAMP", max)
    )
    item_df = event_df.groupby('ITEM_ID').size().to_frame("_siz")
    return (user_df, item_df)


def groupby_collect(series):
    """
    >>> groupby_collect(pd.Series([1,2,3,4,5], index=[1,1,2,3,3])).to_dict()
    {1: [1, 2], 2: [3], 3: [4, 5]}
    """
    last_i = None
    for i in series.index.values:
        if last_i is not None and last_i>i:
            warnings.warn("unsorted input to groupby_collect may be inefficient")
            series = series.sort_index(kind='mergesort')
            break
        last_i = i

    splits = np.where(
        np.array(series.index.values[1:]) != np.array(series.index.values[:-1])
        )[0] + 1

    return pd.Series(
        [x.tolist() for x in np.split(series.values, splits)],
        index=series.index.values[np.hstack([[0], splits])])


def create_matrix(event_df, user_index, item_index, return_type='csr'):
    """ create matrix and prune unknown indices """
    user2ind = {k:i for i,k in enumerate(user_index)}
    item2ind = {k:i for i,k in enumerate(item_index)}
    event_df = event_df[
        event_df['USER_ID'].isin(set(user_index)) &
        event_df['ITEM_ID'].isin(set(item_index))
    ]
    data = np.ones(len(event_df))
    i = [user2ind[k] for k in event_df['USER_ID']]
    j = [item2ind[k] for k in event_df['ITEM_ID']]
    shape = (len(user_index), len(item_index))
    csr = sp.sparse.coo_matrix((data, (i,j)), shape=shape).tocsr()
    if return_type == 'csr':
        return csr
    elif return_type == 'df':
        return pd.DataFrame.sparse.from_spmatrix(csr, user_index, item_index)


def split_by_time(user_df, test_start, valid_start):
    user_df = user_df.copy()
    user_df['TEST_START_TIME'] = test_start
    valid_df = user_df.copy()
    valid_df['TEST_START_TIME'] = valid_start
    return (user_df, valid_df)


def split_by_user(user_df, in_groupA, test_start, relative=True):
    """ test_start=+inf: training user without test window; -inf: user to exclude """
    if relative:
        test_start = user_df['_Tmin'] + test_start
    train_df = user_df.copy()
    train_df['TEST_START_TIME'] = np.where(in_groupA, float("inf"), test_start)
    valid_df = user_df.copy()
    valid_df['TEST_START_TIME'] = np.where(in_groupA, test_start, float("-inf"))
    return (train_df, valid_df)


def sample_groupA(user_df, frac=0.5, seed=888):
    return user_df.index.isin(
        user_df.sample(frac=frac, random_state=seed).index
    )


def filter_min_len(event_df, min_user_len, min_item_len):
    """ CAVEAT: use in conjunction with dataclass filter to avoid future-leaking bias """
    users = event_df.groupby('USER_ID').size()
    items = event_df.groupby('ITEM_ID').size()
    return event_df[
        event_df['USER_ID'].isin(users[users>=min_user_len].index) &
        event_df['ITEM_ID'].isin(items[items>=min_item_len].index)
    ]


class _LitValidated(LightningModule):
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        if isinstance(loss, collections.abc.Mapping) and 'loss' in loss:
            loss = loss['loss']
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.val_loss = torch.stack(outputs).mean()
