import pandas as pd, numpy as np
import scipy.sparse as sps
import warnings, dataclasses, argparse
from ..util import (create_matrix, perplexity, timed, groupby_collect,
                    matrix_reindex, fill_factory_inplace)


def _check_index(event_df, user_df, item_df, allow_user_duplicates=False):
    assert not user_df.index.has_duplicates or allow_user_duplicates, \
        "allow one row per user for autoregressive training"
    assert not item_df.index.has_duplicates, "assume one entry per item"
    assert event_df['USER_ID'].isin(user_df.index).all(), \
        "user_df must include all users in event_df"
    assert event_df['ITEM_ID'].isin(item_df.index).all(), \
        "item_df must include all items in event_df"


def _check_more_inputs(event_df, user_df, item_df):
    assert user_df['TEST_START_TIME'].notnull().all(), \
        "user_df must include TEST_START_TIME for all users"

    with timed("checking whether the events are sorted via necessary conditions"):
        user_time = event_df[['USER_ID', 'TIMESTAMP']].values
        if not (user_time[1:] >= user_time[:-1]).any(axis=1).all():
            warnings.warn("please sort events in [user, time] for best efficiency.")

    with timed("checking for repeated user-item events"):
        nunique = len(set(event_df.set_index(['USER_ID', 'ITEM_ID']).index))
        if nunique < len(event_df):
            warnings.warn(f"user-item repeat rate {len(event_df) / nunique - 1:%}")


def _mark_and_trim_holdout(event_df, user_df, horizon):
    """ mark _holdout=1 on test [start, end); remove post-test events
    training-only (Group-A) users should have TEST_START_TIME=+inf
    """
    event_df = event_df.join(user_df[['TEST_START_TIME']], on='USER_ID')
    event_df['_holdout'] = (
        event_df['TIMESTAMP'] >= event_df['TEST_START_TIME']
    ).astype(int) + (
        event_df['TIMESTAMP'] >= event_df['TEST_START_TIME'] + horizon
    ).astype(int)
    del event_df['TEST_START_TIME']

    if (event_df['_holdout'] == 2).any():
        warnings.warn("Removing trailing events after the test horizons; "
                      "number of total events decreases "
                      f"by {(event_df['_holdout'] == 2).mean():.1%} "
                      f"to {(event_df['_holdout'] < 2).sum():,}.")
        event_df = event_df[event_df['_holdout'] < 2].copy()

    return event_df


def _reindex_user_hist(user_df, index, factory={
        "_hist_items": list,
        "_hist_len": lambda: 0,
        "_hist_ts": lambda: [],
        "TEST_START_TIME": lambda: float("inf"),
}):
    missing = [i not in user_df.index for i in index]
    user_df = user_df.reindex(index)
    if any(missing):
        fill_factory_inplace(user_df, missing, factory)
    return user_df


def _augment_user_hist(user_df, event_df):
    """ extract user histories from event_df before the respective TEST_START_TIME;
        append columns: _hist_items, _hist_ts, _hist_len
    """
    @timed("groupby, collect, reindex")
    def fn(col_name):
        hist = groupby_collect(
            event_df[event_df['_holdout'] == 0].set_index('USER_ID')[col_name]
        )
        return _reindex_user_hist(hist, user_df.index, {None: list})

    user_df = user_df.join(fn("ITEM_ID").to_frame("_hist_items")) \
                     .join(fn("TIMESTAMP").to_frame("_hist_ts"))

    user_df['_hist_len'] = user_df['_hist_items'].apply(len)
    return user_df


def _augment_item_hist(item_df, event_df):
    """ augment history inferred from training set """
    return item_df.join(
        event_df[event_df['_holdout'] == 0]
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
    training_data: argparse.Namespace  # mock this class with the first four attributes
    horizon: float = float("inf")
    prior_score: pd.DataFrame = None    # index=USER_ID, column=ITEM_ID
    default_user_rec_top_c: int = None
    default_item_rec_top_k: int = None

    def __post_init__(self):
        assert self.target_csr.shape == (len(self.user_in_test), len(self.item_in_test)), \
            "target shape must match with test user/item lengths"

        if "_hist_items" not in self.user_in_test:
            warnings.warn(f"{self} not applicable for sequence models.")
        if "TEST_START_TIME" not in self.user_in_test or "_hist_ts" not in self.user_in_test:
            warnings.warn(f"{self} not applicable for temporal models.")

        if hasattr(self.training_data, "event_df"):
            _check_index(self.training_data.event_df,
                         self.training_data.user_df, self.training_data.item_df)

        if self.prior_score is not None:
            assert (self.prior_score.shape == self.target_csr.shape), \
                "prior_score shape must match with test target_csr"

        if self.default_user_rec_top_c is None:
            self.default_user_rec_top_c = int(np.ceil(len(self.user_in_test) / 100))
        if self.default_item_rec_top_k is None:
            self.default_item_rec_top_k = int(np.ceil(len(self.item_in_test) / 100))
        self.user_ppl_baseline = perplexity(self.user_in_test['_hist_len'])
        self.item_ppl_baseline = perplexity(self.item_in_test['_hist_len'])

    def __hash__(self):
        return id(self)

    def get_stats(self):
        if "TEST_START_TIME" in self.user_in_test and "_hist_ts" in self.user_in_test:
            avg_hist_span = self.user_in_test[  # test users with finite history
                (self.user_in_test["TEST_START_TIME"] < np.inf) &
                (self.user_in_test["_hist_ts"].apply(len) > 0)
            ].apply(
                lambda x: x["TEST_START_TIME"] - x["_hist_ts"][0], axis=1
            ).mean()
        else:
            avg_hist_span = float("nan")

        return {
            'user_df': {
                '# test users': len(self.user_in_test),
                '# train users': len(self.training_data.user_df),
                'avg hist len': self.user_in_test['_hist_len'].mean(),
                'avg hist span': avg_hist_span,
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
        if axis == 0:
            old_index = self.user_in_test.index
            user_in_test = _reindex_user_hist(self.user_in_test, index)
            item_in_test = self.item_in_test

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
                   min_user_len=1, min_item_len=1, prior_score=None, exclude_train=False,
                   test_incl_users_with_posinf_test_time=False,
                   test_incl_users_with_neginf_test_time=True,
                   ):
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
    event_df = _mark_and_trim_holdout(event_df, user_df, horizon)
    user_df = _augment_user_hist(user_df, event_df)
    item_df = _augment_item_hist(item_df, event_df)

    print("marking and cleaning test data")
    user_in_test = user_df[
        (user_df['_hist_len'] >= min_user_len) &
        ((user_df['TEST_START_TIME'] < np.inf) | test_incl_users_with_posinf_test_time) &
        ((user_df['TEST_START_TIME'] > -np.inf) | test_incl_users_with_neginf_test_time)
    ].copy()
    item_in_test = item_df[
        item_df['_hist_len'] >= min_item_len
    ].copy()
    target_csr = create_matrix(
        event_df[event_df['_holdout'] == 1].copy(),
        user_in_test.index, item_in_test.index
    )
    training_data = argparse.Namespace(
        event_df=event_df[event_df['_holdout'] == 0].copy(),
        user_df=user_df, item_df=item_df
    )

    if exclude_train:
        print("optionally excluding training events in predictions and targets")
        assert prior_score is None, "double configuration for prior score"

        exclude_csr = create_matrix(
            event_df[event_df['_holdout'] == 0].copy(),
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
