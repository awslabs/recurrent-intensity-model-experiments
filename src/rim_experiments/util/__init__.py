import pandas as pd, numpy as np, scipy as sp
import functools, collections, time, contextlib, os, torch, gc, warnings
from datetime import datetime
from pytorch_lightning import LightningModule

if torch.cuda.is_available():
    from pynvml import *
    nvmlInit()

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


cached_property = lambda foo: property(functools.lru_cache()(foo))


@empty_cache_on_exit
def _assign_topk(S, k, tie_breaker=1e-10, device="cpu", batch_size=10000):
    def fn(s):
        if hasattr(s, "eval"):
            s = s.eval(device)
        else:
            s = torch.tensor(s, device=device)
        if tie_breaker:
            s = s + torch.rand(*s.shape, device=device) * tie_breaker
        return s.topk(k).indices.cpu().numpy()

    indices = np.vstack([
        fn(S[i:min(i+batch_size, S.shape[0])])
        for i in range(0, S.shape[0], batch_size)
    ])
    return sp.sparse.csr_matrix((
        np.ones(indices.size),
        np.ravel(indices),
        np.arange(0, indices.size+1, indices.shape[1]),
    ), shape=S.shape)


@empty_cache_on_exit
def _argsort(S, tie_breaker=1e-10, device="cpu"):
    print(f"_argsort {S.size:,} scores on device {device}; ", end="")
    if hasattr(S, "eval"):
        S_torch = S.eval(device)
    else:
        S_torch = torch.tensor(S, device=device)
    if tie_breaker>0:
        S_torch = S_torch + torch.rand(*S.shape, device=device) * tie_breaker
    argsort_ind = torch.argsort(-S_torch.reshape(-1)).cpu().numpy()
    return np.unravel_index(argsort_ind, S.shape)


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
            series = series.sort_index(kind='mergesort')
            break
        last_i = i

    splits = np.where(
        np.array(series.index.values[1:]) != np.array(series.index.values[:-1])
        )[0] + 1

    return pd.Series(
        [x.tolist() for x in np.split(series.values, splits)],
        index=series.index.values[np.hstack([[0], splits])])


def create_matrix(event_df, user_index, item_index, return_type):
    user2ind = {k:i for i,k in enumerate(user_index)}
    item2ind = {k:i for i,k in enumerate(item_index)}
    event_df = event_df[
        event_df['USER_ID'].isin(user_index) &
        event_df['ITEM_ID'].isin(item_index)
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


def df_to_coo(df):
    try:
        return df.sparse.to_coo()
    except KeyError:
        warnings.warn("pandas bug: https://github.com/pandas-dev/pandas/issues/25270")
        df = df.copy()
        df.index = list(range(len(df.index)))
        df.columns = list(range(len(df.columns)))
        return df.sparse.to_coo()


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


def get_batch_size(shape):
    """ round to similar batch sizes """
    n_users, n_items = shape
    if torch.cuda.device_count():
        total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        total_memory = 16e9
    max_batch_size = total_memory / 8 / 10 / n_items
    n_batches = int(n_users / max_batch_size) + 1
    return int(np.ceil(n_users / n_batches))


def get_best_gpus():
    if "BEST_GPU" in os.environ:
        return [int(os.environ['BEST_GPU'])]
    memory_free = [
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
        for i in range(torch.cuda.device_count())
    ]
    print("best gpu", np.argmax(memory_free))
    return [np.argmax(memory_free)]
