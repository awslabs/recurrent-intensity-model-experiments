import pandas as pd, numpy as np, scipy as sp
import scipy.sparse as sps
import functools, collections, warnings, dataclasses, argparse
from ..util import create_matrix, cached_property, perplexity, \
           timed, groupby_collect, matrix_reindex, get_batch_size, fill_factory_inplace


def _check_index(event_df, user_df, item_df):
    assert not user_df.index.has_duplicates, "simplify to one test window per user"
    assert not item_df.index.has_duplicates, "assume one entry per item"
    assert event_df['USER_ID'].isin(user_df.index).all(), \
                                "user_df must include all users in event_df"
    assert event_df['ITEM_ID'].isin(item_df.index).all(), \
                                "item_df must include all items in event_df"


def _check_more_inputs(event_df, user_df, item_df):
    assert user_df['TEST_START_TIME'].notnull().all(), \
                    "user_df must include TEST_START_TIME for all users"

    with timed("checking whether the events are sorted via necessary conditions"):
        user_time = event_df[['USER_ID','TIMESTAMP']].values
        if not (user_time[1:] >= user_time[:-1]).any(axis=1).all():
            warnings.warn("please sort events in [user, time] for best efficiency.")

    with timed("checking for repeated user-item events"):
        nunique = len(set(event_df.set_index(['USER_ID', 'ITEM_ID']).index))
        if nunique < len(event_df):
            warnings.warn(f"user-item repeat rate {len(event_df) / nunique - 1:%}")


def _mark_holdout(event_df, user_df, horizon):
    """ mark _holdout=1 on test [start, end); mark _holdout=2 on post-test events
    training-only (Group-A) users should have TEST_START_TIME=+inf
    """
    event_df = event_df.join(user_df[['TEST_START_TIME']], on='USER_ID')
    event_df['_holdout'] = (
        event_df['TIMESTAMP'] >= event_df['TEST_START_TIME']
    ).astype(int) + (
        event_df['TIMESTAMP'] >= event_df['TEST_START_TIME'] + horizon
    ).astype(int)

    post_test = (event_df['_holdout']==2).mean()
    if post_test>0:
        warnings.warn("Post-test events with _holdout=2 should be ignored; "
            f"they account for {post_test:.1%} of all events")
    del event_df['TEST_START_TIME']
    return event_df


def _augment_user_hist(user_df, event_df):
    """ extract user histories from event_df before the respective TEST_START_TIME;
        append columns: _hist_items, _hist_ts, _timestamps, _hist_len, _hist_span
    """
    @timed("groupby, collect, reindex")
    def fn(col_name):
        hist = groupby_collect(
            event_df[event_df['_holdout']==0].set_index('USER_ID')[col_name]
            )
        return hist.reindex(user_df.index).apply(
            lambda x: x if isinstance(x, collections.abc.Iterable) else [])

    user_df = user_df.join(fn("ITEM_ID").to_frame("_hist_items")) \
                     .join(fn("TIMESTAMP").to_frame("_hist_ts"))

    user_df['_timestamps'] = user_df.apply(
        lambda x: x['_hist_ts'] + [x['TEST_START_TIME']], axis=1)

    user_df['_hist_len'] = user_df['_hist_items'].apply(len)
    user_df['_hist_span'] = user_df['_timestamps'].apply(lambda x: x[-1] - x[0])
    return user_df


def _augment_item_hist(item_df, event_df):
    """ augment history inferred from training set """
    return item_df.join(
        event_df[event_df['_holdout']==0]
        .groupby('ITEM_ID').size().to_frame('_hist_len')
    ).fillna({'_hist_len': 0})


@dataclasses.dataclass(eq=False)
class Dataset:
    """ A dataset with target_csr from test users and items, reference to training data,
    optional horizon and mask for evaluation purposes.
    The class can be mocked as::

        D = argparse.Namespace(
            target_csr=..., user_in_test=..., item_in_test=...,
            training_data=argparse.Namespace(event_df=..., user_df=..., item_df=...),
            ...)
    """
    target_csr: sps.spmatrix        # index=USER_ID, column=ITEM_ID
    user_in_test: pd.DataFrame      # index=USER_ID
    item_in_test: pd.DataFrame      # index=ITEM_ID
    training_data: argparse.Namespace # mock this class with the first four attributes
    horizon: float = float("inf")
    prior_score: pd.DataFrame = None    # index=USER_ID, column=ITEM_ID

    def __post_init__(self):
        assert self.target_csr.shape == (len(self.user_in_test), len(self.item_in_test)), \
            "target shape must match with test user/item lengths"

        if "_hist_items" not in self.user_in_test:
            warnings.warn(f"{self} not applicable for sequence models.")
        if "_timestamps" not in self.user_in_test:
            warnings.warn(f"{self} not applicable for temporal models.")

        _check_index(self.training_data.event_df,
            self.training_data.user_df, self.training_data.item_df)

        if self.prior_score is not None:
            assert (self.prior_score.shape == self.target_csr.shape), \
                        "prior_score shape must match with test target_csr"

        self.default_user_rec_top_c = int(np.ceil(len(self.user_in_test) / 100))
        self.default_item_rec_top_k = int(np.ceil(len(self.item_in_test) / 100))
        self.user_ppl_baseline = perplexity(self.user_in_test['_hist_len'])
        self.item_ppl_baseline = perplexity(self.item_in_test['_hist_len'])

    def __hash__(self):
        return id(self)

    def get_stats(self):
        return {
            'user_df': {
                '# test users': len(self.user_in_test),
                '# train users': len(self.training_data.user_df),
                'avg hist len': self.user_in_test['_hist_len'].mean(),
                'avg hist span': self.user_in_test['_hist_span'].mean(),
                'horizon': self.horizon,
                'avg target items': self.target_csr.sum(axis=1).mean(),
            },
            'item_df': {
                '# test items': len(self.item_in_test),
                '# train items': len(self.training_data.item_df),
                'avg hist len': self.item_in_test['_hist_len'].mean(),
                'avg target users': self.target_csr.sum(axis=0).mean(),
            },
            'event_df': {
                '# train events': len(self.training_data.event_df),
                '# test events': self.target_csr.sum(),
                'default_user_rec_top_c': self.default_user_rec_top_c,
                'default_item_rec_top_k': self.default_item_rec_top_k,
                "user_ppl_baseline": self.user_ppl_baseline,
                "item_ppl_baseline": self.item_ppl_baseline,
            },
        }

    def print_stats(self, verbose=True):
        print(pd.DataFrame(self.get_stats()).T.stack().apply('{:.2f}'.format))
        if verbose:
            print(self.user_in_test.sample().iloc[0])
            print(self.item_in_test.sample().iloc[0])

    def reindex(self, index, axis):
        if axis==0:
            old_index = self.user_in_test.index
            user_in_test = self.user_in_test.reindex(index, fill_value=0)
            item_in_test = self.item_in_test

            if not set(index) < set(old_index):
                fill_factory_inplace(user_in_test, [i not in old_index for i in index], {
                    "_hist_items": list, '_hist_ts': list,
                    'TEST_START_TIME': lambda: float('inf'),
                    "_timestamps": lambda: [float('inf')]
                    })
        else:
            old_index = self.item_in_test.index
            user_in_test = self.user_in_test
            item_in_test = self.item_in_test.reindex(index, fill_value=0)

        target_csr = matrix_reindex(
            self.target_csr, old_index, index, axis, fill_value=0)

        if self.prior_score is None:
            prior_score = None
        else:
            prior_score = matrix_reindex(
                self.prior_score, old_index, index, axis, fill_value=0)

        return self.__class__(target_csr, user_in_test, item_in_test,
            self.training_data, self.horizon, prior_score)


def create_dataset(event_df, user_df, item_df, horizon=float("inf"),
    min_user_len=1, min_item_len=1, prior_score=None, exclude_train=False):
    """ Create a labeled dataset from 3 related tables and additional configurations.

    :parameter event_df: [USER_ID, ITEM_ID, TIMESTAMP]
    :parameter user_df: [USER_ID (index), TEST_START_TIME]
    :parameter item_df: [ITEM_ID (index)]
    :parameter horizon: extract test window from TIMESTAMP, TEST_START_TIME, and horizon
    :parameter min_user_len: filter out test users with empty histories to avoid biases
    :parameter min_item_len: filter out test items with empty histories to avoid biases
    :parameter prior_score: add a prior score to boost/penalize certain user-item pairs
        in prediction
    :parameter exclude_train: exclude training events from predictions and targets

    Infer target labels from TEST_START_TIME (per user) and horizon.
    Filter test users/items by _hist_len.
    """
    _check_index(event_df, user_df, item_df)
    _check_more_inputs(event_df, user_df, item_df)

    print("augmenting and data tables")
    event_df = _mark_holdout(event_df, user_df, horizon)
    user_df = _augment_user_hist(user_df, event_df)
    item_df = _augment_item_hist(item_df, event_df)

    print("marking and cleaning test data")
    user_in_test = user_df[
        (user_df['_hist_len']>=min_user_len) &
        (user_df['TEST_START_TIME']<float("inf")) # training-only users have inf start time
    ].copy()
    item_in_test = item_df[
        item_df['_hist_len']>=min_item_len
    ].copy()
    target_csr = create_matrix(
        event_df[event_df['_holdout']==1].copy(),
        user_in_test.index, item_in_test.index
    )
    training_data = argparse.Namespace(
        event_df=event_df[event_df['_holdout']==0].copy(),
        user_df=user_df, item_df=item_df
    )

    if exclude_train:
        print("optionally excluding training events in predictions and targets")
        assert prior_score is None, "double configuration for prior score"

        exclude_csr = create_matrix(
            event_df[event_df['_holdout']==0].copy(),
            user_in_test.index, item_in_test.index
        ).astype(bool)
        prior_score = exclude_csr * -1e10    # clip -inf to avoid nan

        mask_csr = target_csr.astype(bool) > exclude_csr.astype(bool)
        target_csr = target_csr.multiply(mask_csr)
        target_csr.eliminate_zeros()

    D = Dataset(target_csr, user_in_test, item_in_test, training_data,
        horizon, prior_score)
    print("Dataset created!")
    return D
