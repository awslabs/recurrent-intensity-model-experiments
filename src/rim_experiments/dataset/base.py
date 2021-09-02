import pandas as pd, numpy as np, scipy as sp
import functools, collections, warnings
from ..util import create_matrix, cached_property, perplexity, warn_nan_output


def _check_inputs(event_df, user_df, item_df):
    assert not user_df.index.has_duplicates, "assume one test window per user for simplicity"
    assert not item_df.index.has_duplicates, "assume one entry per item"
    assert event_df['USER_ID'].isin(user_df.index).all(), \
                                "user_df must include all users in event_df"
    assert event_df['ITEM_ID'].isin(item_df.index).all(), \
                                "item_df must include all items in event_df"
    if (event_df.groupby("USER_ID")["TIMESTAMP"].diff() < 0).any():
        warnings.warn("please sort events in time or [user, time] for temporal models.")
    duplication_rate = event_df.duplicated(["USER_ID", "ITEM_ID"]).mean()
    if duplication_rate>0:
        warnings.warn(f"duplicated user-item at {duplication_rate:%}", DeprecationWarning)


def _holdout_and_trim_events(event_df, user_df, horizon):
    """ mark _holdout=1 on test [start, end); remove post-test events
    training-only (Group-A) users should have TEST_START_TIME=+inf
    """
    test_start = user_df['TEST_START_TIME'].reindex(event_df['USER_ID']).values

    assert not np.isnan(test_start).any(), "user_df must include all users"
    event_df['_holdout'] = (test_start <= event_df['TIMESTAMP']).astype(bool)

    if horizon == float("inf"):
        warnings.warn("TPP models require finite horizon to train properly.")
    return event_df[event_df['TIMESTAMP'] < test_start + horizon].copy()


def _augment_user_hist(user_df, event_df):
    """ augment history length before test start time """
    def fn(col_name):
        hist = event_df[event_df['_holdout']==0] \
                .groupby('USER_ID')[col_name].apply(list) \
                .reindex(user_df.index)
        return hist.apply(lambda x: x if isinstance(x, collections.abc.Iterable) else [])

    user_df = user_df.join(
        fn("ITEM_ID").to_frame("_hist_items"), on='USER_ID'
    ).join(
        fn("TIMESTAMP").to_frame("_hist_ts"), on='USER_ID'
    )

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


class Dataset:
    """
    A dataset class contains 3 related tables and we will infer columns with underscored names
        event_df: [USER_ID, ITEM_ID, TIMESTAMP]; will infer [_holdout]
        user_df: [USER_ID, TEST_START_TIME]; will infer [_hist_items, _timestamps, _in_test]
        item_df: [ITEM_ID]; will infer [_hist_len, _in_test]
    """
    def __init__(self, event_df, user_df, item_df, horizon,
        min_user_len=1, min_item_len=1, drop_duplicates=True, print_stats=False):

        _check_inputs(event_df, user_df, item_df)
        self._raw_inputs = locals().copy()

        print("augmenting and trimming data")
        self.event_df = _holdout_and_trim_events(event_df, user_df, horizon)
        self.user_df = _augment_user_hist(user_df, self.event_df)
        self.item_df = _augment_item_hist(item_df, self.event_df)
        self.horizon = horizon

        print("marking and cleaning test data")
        self.user_df['_in_test'] = (
            (self.user_df['_hist_len']>=min_user_len) &
            (self.user_df['TEST_START_TIME']<float("inf")) # exclude Group-A users
        ).astype(bool)
        self.item_df['_in_test'] = (
            self.item_df['_hist_len']>=min_item_len
        ).astype(bool)
        self.target_df = create_matrix(
            self.event_df[self.event_df['_holdout']==1],
            self.user_in_test.index, self.item_in_test.index, "df"
        )

        print("inferring default parameters")
        self.default_user_rec_top_c = int(np.ceil(len(self.user_in_test) / 100))
        self.default_item_rec_top_k = int(np.ceil(len(self.item_in_test) / 100))

        if print_stats:
            print('dataset stats')
            print(pd.DataFrame(self.get_stats()).T.stack().apply('{:.2f}'.format))
            print(self.user_df.sample().iloc[0])
            print(self.item_df.sample().iloc[0])

    def get_stats(self):
        return {
            'user_df': {
                '# warm users': sum(self.user_df['_in_test']),
                '# cold users': sum(~self.user_df['_in_test']),
                'avg hist len': self.user_in_test['_hist_len'].mean(),
                'avg hist span': self.user_in_test['_hist_span'].mean(),
                'horizon': self.horizon,
                'avg target items': self.target_df.sum(axis=1).mean(),
            },
            'item_df': {
                '# warm items': sum(self.item_df['_in_test']),
                '# cold items': sum(~self.item_df['_in_test']),
                'avg hist len': self.item_in_test['_hist_len'].mean(),
                'avg target users': self.target_df.sum(axis=0).mean(),
            },
            'event_df': {
                '# train events': sum(self.event_df['_holdout']==0),
                '# test events': self.target_df.values.sum(),
                'default_user_rec_top_c': self.default_user_rec_top_c,
                'default_item_rec_top_k': self.default_item_rec_top_k,
                "user_ppl": perplexity(self.user_in_test['_hist_len']),
                "item_ppl": perplexity(self.item_in_test['_hist_len']),
            },
        }

    @property
    def user_in_test(self):
        return self.user_df[self.user_df['_in_test']]

    @property
    def item_in_test(self):
        return self.item_df[self.item_df['_in_test']]

    @warn_nan_output
    def transform(self, S):
        """ reindex the score matrix to match with test users and items """
        return S.reindex(self.user_in_test.index) \
            .reindex(self.item_in_test.index, axis=1)
