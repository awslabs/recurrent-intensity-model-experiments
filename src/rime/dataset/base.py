import pandas as pd, numpy as np
import scipy.sparse as sps
import warnings, dataclasses, argparse
from typing import List
from ..util import (create_matrix, perplexity, timed, groupby_unexplode, indices2csr,
                    matrix_reindex, fill_factory_inplace, LazyScoreBase)


def _check_inputs(event_df, user_df, item_df):
    assert user_df['TEST_START_TIME'].notnull().all(), \
        "user_df must include TEST_START_TIME for all users"
    assert not user_df.set_index('TEST_START_TIME', append=True).index.has_duplicates, \
        "allow one entry per user-time instance"
    assert not item_df.index.has_duplicates, "allow one entry per item"

    if event_df is not None:
        assert event_df['USER_ID'].isin(user_df.index).all(), \
            "user_df must include all users in event_df"
        assert event_df['ITEM_ID'].isin(item_df.index).all(), \
            "item_df must include all items in event_df"

    with timed("checking for repeated user-item events"):
        nunique = len(set(event_df.set_index(['USER_ID', 'ITEM_ID']).index))
        if nunique < len(event_df):
            warnings.warn(f"user-item repeat rate {len(event_df) / nunique - 1:%}")


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


@dataclasses.dataclass
class Dataset:
    """ A dataset with target_csr from given users and items, with aggregated histories and
    optional horizon and prior_score for evaluation purposes.
    """
    target_csr: sps.spmatrix  # index=USER_ID, column=ITEM_ID
    user_df: pd.DataFrame     # index=USER_ID, columns=[TEST_START_TIME, _hist_len, _hist_ts]
    item_df: pd.DataFrame     # index=ITEM_ID, columns=[_hist_len]
    horizon: float = float("inf")
    prior_score: pd.DataFrame = None    # index=USER_ID, column=ITEM_ID
    default_user_rec_top_c: int = None
    default_item_rec_top_k: int = None

    def __post_init__(self):
        assert self.target_csr.shape == (len(self.user_df), len(self.item_df)), \
            "target shape must match with test user/item lengths"

        if self.prior_score is not None:
            assert (self.prior_score.shape == self.target_csr.shape), \
                "prior_score shape must match with test target_csr"

        if self.default_user_rec_top_c is None:
            self.default_user_rec_top_c = int(np.ceil(len(self.user_df) / 100))
        if self.default_item_rec_top_k is None:
            self.default_item_rec_top_k = int(np.ceil(len(self.item_df) / 100))
        self.user_ppl_baseline = perplexity(self.user_df['_hist_len'])
        self.item_ppl_baseline = perplexity(self.item_df['_hist_len'])

    def get_stats(self):
        if "TEST_START_TIME" in self.user_df and "_hist_ts" in self.user_df:
            avg_hist_span = self.user_df[  # test users with finite history
                (self.user_df["TEST_START_TIME"] < np.inf) &
                (self.user_df["_hist_ts"].apply(len) > 0)
            ].apply(
                lambda x: x["TEST_START_TIME"] - x["_hist_ts"][0], axis=1
            ).mean()
        else:
            avg_hist_span = float("nan")

        return {
            'user_df': {
                '# users': len(self.user_df),
                'avg hist len': self.user_df['_hist_len'].mean(),
                'avg hist span': avg_hist_span,
                'horizon': self.horizon,
                'avg target items': self.target_csr.sum(axis=1).mean(),
            },
            'item_df': {
                '# items': len(self.item_df),
                'avg hist len': self.item_df['_hist_len'].mean(),
                'avg target users': self.target_csr.sum(axis=0).mean(),
            },
            'event_df': {
                '# train events': self.user_df['_hist_len'].sum(),
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
            print(self.user_df.sample().iloc[0])
            print(self.item_df.sample().iloc[0])

    def reindex(self, index, axis):
        if axis == 0:
            old_index = self.user_df.index
            user_df = _reindex_user_hist(self.user_df, index)
            item_df = self.item_df

        else:
            old_index = self.item_df.index
            user_df = self.user_df
            item_df = self.item_df.reindex(index, fill_value=0)

        target_csr = matrix_reindex(
            self.target_csr, old_index, index, axis, fill_value=0)

        if self.prior_score is None:
            prior_score = None
        elif isinstance(self.prior_score, LazyScoreBase):
            prior_score = self.prior_score.reindex(
                index, axis, fill_value=0, old_index=old_index)
        else:
            prior_score = matrix_reindex(
                self.prior_score, old_index, index, axis, fill_value=0)

        return self.__class__(target_csr, user_df, item_df, self.horizon, prior_score,
                              self.default_user_rec_top_c, self.default_item_rec_top_k)


def create_dataset(event_df, user_df, item_df, horizon=float("inf"),
                   min_user_len=1, min_item_len=1, prior_score=None, exclude_train=False,
                   test_incl_users_with_posinf_test_time=False,
                   test_incl_users_with_neginf_test_time=True):
    """ Create a labeled dataset from 3 related tables and additional configurations.

    :parameter event_df: [USER_ID, ITEM_ID, TIMESTAMP]
    :parameter user_df: [USER_ID (index), TEST_START_TIME]
    :parameter item_df: [ITEM_ID (index)]
    :parameter horizon: extract test window from TIMESTAMP, TEST_START_TIME, and horizon
    :parameter min_user_len: filter out users with empty histories to avoid anti-causal biases
    :parameter min_item_len: filter out items with empty histories to avoid anti-causal biases
    :parameter prior_score: add a prior score to boost/penalize certain user-item pairs
        in prediction
    :parameter exclude_train: exclude training events from predictions and targets

    Infer target labels from TEST_START_TIME (per user) and horizon.
    Filter users/items by _hist_len.
    """
    _check_inputs(event_df, user_df, item_df)

    with timed("creating user_explode to handle user multi-time indices"):
        user_explode = user_df.join(event_df.set_index('USER_ID'), how='inner') \
            .set_index('TEST_START_TIME', append=True) \
            .join(pd.Series(np.arange(len(item_df)), item_df.index).to_frame('_j'), on='ITEM_ID')
    user_time_index = user_df.set_index("TEST_START_TIME", append=True).index

    with timed("generating user histories"):
        user_df = user_df.copy()
        hist_explode = user_explode[user_explode['TIMESTAMP'] < user_explode.index.get_level_values(1)]
        user_df['_hist_items'] = groupby_unexplode(hist_explode['ITEM_ID'], user_time_index).values
        user_df['_hist_ts'] = groupby_unexplode(hist_explode['TIMESTAMP'], user_time_index).values
        user_df['_hist_len'] = user_df['_hist_items'].apply(len)

    print("generating item histories")
    item_df = item_df.copy()
    item_df['_hist_len'] = event_df[
        event_df['TIMESTAMP'] <
        user_df.groupby(level=0)['TEST_START_TIME'].min()  # count by user first-test time
               .reindex(event_df['USER_ID'], fill_value=float('-inf')).values
    ].groupby('ITEM_ID').size().reindex(item_df.index, fill_value=0)

    print("generating targets")
    target_explode = user_explode[
        (user_explode['TIMESTAMP'] >= user_explode.index.get_level_values(1)) &
        (user_explode['TIMESTAMP'] < user_explode.index.get_level_values(1) + horizon)
    ]
    target_csr = indices2csr(groupby_unexplode(target_explode['_j'], user_time_index).values,
                             shape1=len(item_df))

    if exclude_train:
        print("optionally excluding training events in predictions and targets")
        assert prior_score is None, "double configuration for prior score"

        exclude_csr = indices2csr(groupby_unexplode(hist_explode['_j'], user_time_index).values,
                                  shape1=len(item_df))
        prior_score = exclude_csr * -1e10    # clip -inf to avoid nan

        mask_csr = target_csr.astype(bool) > exclude_csr.astype(bool)
        target_csr = target_csr.multiply(mask_csr)

    print("filtering users and items; notice that the user-history data may contain extra items")
    user_in_test_bool = (
        (user_df['_hist_len'] >= min_user_len) &
        ((user_df['TEST_START_TIME'] < np.inf) | test_incl_users_with_posinf_test_time) &
        ((user_df['TEST_START_TIME'] > -np.inf) | test_incl_users_with_neginf_test_time)
    ).values.astype(bool)
    item_in_test_bool = (item_df['_hist_len'] >= min_item_len).values.astype(bool)

    D = Dataset(target_csr[user_in_test_bool][:, item_in_test_bool],
                user_df[user_in_test_bool].copy(),
                item_df[item_in_test_bool].copy(),
                horizon,
                None if prior_score is None else
                    prior_score[user_in_test_bool][:, item_in_test_bool])
    print("Dataset created!")
    return D


def create_temporal_splits(event_df, user_df, item_df, TEST_START_TIME,
                           horizon, validating_horizon, num_V_extra=0, **kw):
    testing_data = create_dataset(event_df,
                                  user_df.assign(TEST_START_TIME=TEST_START_TIME),
                                  item_df, horizon, **kw)
    validating_datasets = [create_dataset(
        event_df,
        user_df.assign(TEST_START_TIME=TEST_START_TIME - validating_horizon * (k + 1)),
        item_df, validating_horizon, **kw)
        for k in range(num_V_extra + 1)
    ]
    return testing_data, *validating_datasets


def create_user_splits(event_df, user_df, item_df, test_start_time_rel, horizon,
                       test_user_ids,):
    pass
