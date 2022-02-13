import pandas as pd, numpy as np
import scipy.sparse as sps
import warnings, dataclasses, argparse
from ..util import (perplexity, timed, groupby_unexplode, indices2csr,
                    matrix_reindex, fill_factory_inplace, LazyScoreBase)


def _check_inputs(event_df, user_df, item_df):
    assert user_df['TEST_START_TIME'].notnull().all(), \
        "user_df must include TEST_START_TIME for all users"
    assert not user_df.set_index('TEST_START_TIME', append=True).index.has_duplicates, \
        "allow one entry per user-time instance"
    assert not item_df.index.has_duplicates, "allow one entry per item"

    assert event_df['USER_ID'].isin(user_df.index).all(), \
        "user_df must include all users in event_df"
    assert event_df['ITEM_ID'].isin(item_df.index).all(), \
        "item_df must include all items in event_df"
    assert (event_df['TIMESTAMP'] >= 0).all(), \
        "assume TIMESTAMP >= 0 for correct user padding and splitting"

    with timed("checking for repeated user-item events"):
        nunique = len(set(event_df.set_index(['USER_ID', 'ITEM_ID']).index))
        if nunique < len(event_df):
            warnings.warn(f"user-item repeat rate {len(event_df) / nunique - 1:%}")


def _reindex_user_hist(user_df, index, factory={
        "_hist_items": list,
        "_hist_len": lambda: 0,
        "_hist_ts": lambda: [],
        "TEST_START_TIME": lambda: 0,
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

    The class also contains or implies training user/item_df based on the historical part
    of the user/item_in_test. If explicitly specified, the user/item_df could be different from
    (often larger than) user/item_in_test.
    """
    target_csr: sps.spmatrix    # index=USER_ID, column=ITEM_ID
    user_in_test: pd.DataFrame  # index=USER_ID, columns=[TEST_START_TIME, _hist_len, _hist_ts]
    item_in_test: pd.DataFrame  # index=ITEM_ID, columns=[_hist_len]
    horizon: float = float("inf")
    prior_score: pd.DataFrame = None    # index=USER_ID, column=ITEM_ID
    default_user_rec_top_c: int = None
    default_item_rec_top_k: int = None
    user_df: pd.DataFrame = None  # default to the history part in test users for sequence models
    item_df: pd.DataFrame = None  # could have different indices than test users/items

    def __post_init__(self):
        assert self.target_csr.shape == (len(self.user_in_test), len(self.item_in_test)), \
            "target shape must match with test user/item lengths"

        if self.prior_score is not None:
            assert (self.prior_score.shape == self.target_csr.shape), \
                "prior_score shape must match with test target_csr"

        if self.default_user_rec_top_c is None:
            self.default_user_rec_top_c = int(np.ceil(len(self.user_in_test) / 100))
        if self.default_item_rec_top_k is None:
            self.default_item_rec_top_k = int(np.ceil(len(self.item_in_test) / 100))
        self.user_ppl_baseline = perplexity(self.user_in_test['_hist_len'])
        self.item_ppl_baseline = perplexity(self.item_in_test['_hist_len'])

        if self.user_df is None:
            self.user_df = self.user_in_test.groupby(level=0, sort=False).first()
        if self.item_df is None:
            self.item_df = self.item_in_test

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
            'user_in_test': {
                '# test users': len(self.user_in_test),
                'avg hist len': self.user_in_test['_hist_len'].mean(),
                'avg hist span': avg_hist_span,
                'avg target len': self.target_csr.sum(axis=1).mean(),
            },
            'item_in_test': {
                '# test items': len(self.item_in_test),
                'avg hist len': self.item_in_test['_hist_len'].mean(),
                'avg target len': self.target_csr.sum(axis=0).mean(),
            },
            'event_df': {
                '# train events': self.user_df['_hist_len'].sum(),
                '# test events': self.target_csr.sum(),
                'horizon': self.horizon,
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
        elif isinstance(self.prior_score, LazyScoreBase):
            prior_score = self.prior_score.reindex(
                index, axis, fill_value=0, old_index=old_index)
        else:
            prior_score = matrix_reindex(
                self.prior_score, old_index, index, axis, fill_value=0)

        return self.__class__(target_csr, user_in_test, item_in_test, self.horizon, prior_score,
                              self.default_user_rec_top_c, self.default_item_rec_top_k,
                              self.user_df, self.item_df)


def create_dataset(event_df, user_df, item_df, horizon=float("inf"),
                   min_user_len=1, min_item_len=1, prior_score=None, exclude_train=False,
                   test_user_extra_filter=lambda x: x['TEST_START_TIME'] < float("inf")):
    """ Create a labeled dataset from 3 related tables and additional configurations.

    :parameter event_df: [USER_ID, ITEM_ID, TIMESTAMP]
    :parameter user_df: [USER_ID (index), TEST_START_TIME]
    :parameter item_df: [ITEM_ID (index)]
    :parameter horizon: extract test window from TIMESTAMP, TEST_START_TIME, and horizon
    :parameter min_user_len: filter out test users with empty histories to avoid anti-causal biases
    :parameter min_item_len: filter out test items with empty histories to avoid anti-causal biases
    :parameter prior_score: add a prior score to boost/penalize certain user-item pairs
        in prediction
    :parameter exclude_train: exclude training events from predictions and targets

    Infer target labels from TEST_START_TIME (per user) and horizon.
    Filter test users/items by _hist_len.
    """
    _check_inputs(event_df, user_df, item_df)

    with timed("creating user_time_index and user_explode"):
        user_time_index = user_df.set_index("TEST_START_TIME", append=True).index
        user_explode = user_df.join(event_df.set_index('USER_ID'), how='inner') \
            .set_index('TEST_START_TIME', append=True) \
            .join(pd.Series(np.arange(len(item_df)), item_df.index).to_frame('_j'), on='ITEM_ID')
        hist_explode = user_explode[
            user_explode['TIMESTAMP'] < user_explode.index.get_level_values(1)]
        target_explode = user_explode[
            (user_explode['TIMESTAMP'] >= user_explode.index.get_level_values(1)) &
            (user_explode['TIMESTAMP'] < user_explode.index.get_level_values(1) + horizon)]

    with timed("generating user histories"):
        _hist_splits = groupby_unexplode(hist_explode, user_time_index, 'splits')
        user_df = user_df.assign(
            _hist_items=[x.tolist() for x in np.split(hist_explode['ITEM_ID'].values, _hist_splits)],
            _hist_ts=[x.tolist() for x in np.split(hist_explode['TIMESTAMP'].values, _hist_splits)],
        ).assign(_hist_len=lambda x: x['_hist_items'].apply(len))

    training_user_df = user_df.groupby(level=0, sort=False).first()
    if len(training_user_df) < len(user_df):
        warnings.warn("Users with multiple TEST_START_TIME detected; "
                      "keeping the first user instance (in dataframe order) for training")

    with timed("generating item histories"):
        _item_size = event_df[
            event_df['TIMESTAMP'] < event_df[['USER_ID']].join(
                training_user_df[['TEST_START_TIME']], on='USER_ID')['TEST_START_TIME']
        ].groupby('ITEM_ID').size()
        item_df = item_df.assign(
            _hist_len=_item_size.reindex(item_df.index, fill_value=0).values)

    with timed("generating targets"):
        target_csr = indices2csr(groupby_unexplode(target_explode['_j'], user_time_index),
                                 shape1=len(item_df))

    if exclude_train:
        print("optionally excluding training events in predictions and targets")
        assert prior_score is None, "double configuration for prior score"

        exclude_csr = indices2csr(groupby_unexplode(hist_explode['_j'], user_time_index),
                                  shape1=len(item_df))
        prior_score = exclude_csr * -1e10    # clip -inf to avoid nan

        mask_csr = target_csr.astype(bool) > exclude_csr.astype(bool)
        target_csr = target_csr.multiply(mask_csr)

    print("filtering users and items; notice that the user-history data may contain extra items")
    user_in_test_bool = ((user_df['_hist_len'] >= min_user_len) &
                         user_df.apply(test_user_extra_filter, axis=1)).values.astype(bool)
    item_in_test_bool = (item_df['_hist_len'] >= min_item_len).values.astype(bool)

    D = Dataset(target_csr[user_in_test_bool][:, item_in_test_bool],
                user_df[user_in_test_bool].copy(),
                item_df[item_in_test_bool].copy(),
                horizon,
                prior_score=None if prior_score is None else
                            prior_score[user_in_test_bool][:, item_in_test_bool],
                user_df=training_user_df,
                item_df=item_df)
    print("Dataset created!")
    return D


def create_temporal_splits(event_df, user_df, item_df, TEST_START_TIME,
                           horizon, validating_horizon, num_V_extra=0, **kw):
    testing_data = create_dataset(event_df,
                                  user_df.assign(TEST_START_TIME=TEST_START_TIME),
                                  item_df, horizon, **kw)
    testing_data.print_stats()
    validating_datasets = [create_dataset(
        event_df,
        user_df.assign(TEST_START_TIME=TEST_START_TIME - validating_horizon * (k + 1)),
        item_df, validating_horizon, **kw)
        for k in range(num_V_extra + 1)
    ]
    return (testing_data, *validating_datasets)


def create_user_splits(event_df, user_df, item_df, test_start_rel, horizon, num_V_extra=0, **kw):
    assert '_in_GroupA' in user_df, "requires _in_GroupA"
    test_start_abs = user_df.get('_Tmin', 0) + test_start_rel
    D = create_dataset(
        event_df,
        user_df.assign(TEST_START_TIME=lambda x: np.where(
            x['_in_GroupA'], float("inf"), test_start_abs)),
        item_df, horizon, **kw)
    D.print_stats()
    V = create_dataset(
        event_df,
        user_df.assign(TEST_START_TIME=lambda x: np.where(
            x['_in_GroupA'], test_start_abs, 0)),
        item_df, horizon, **kw)
    if num_V_extra:
        V0 = create_dataset(
            event_df,
            user_df.assign(TEST_START_TIME=test_start_abs - horizon / 2),
            item_df, horizon / 2, **kw)
        return D, V, V0
    else:
        return D, V
