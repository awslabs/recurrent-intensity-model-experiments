import pandas as pd, numpy as np, scipy as sp
import functools, collections, warnings, dataclasses
from ..util import create_matrix, cached_property, perplexity, \
                   timed, warn_nan_output, groupby_collect, df_to_coo


def _check_index(event_df, user_df, item_df):
    assert not user_df.index.has_duplicates, "assume one test window per user for simplicity"
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
    """ augment history length before test start time """
    @timed("groupby, collect, reindex")
    def fn(col_name):
        hist = groupby_collect(
            event_df[event_df['_holdout']==0].set_index('USER_ID')[col_name]
            )
        return hist.reindex(user_df.index).apply(
            lambda x: x if isinstance(x, collections.abc.Iterable) else [])

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


@dataclasses.dataclass(eq=False)
class _SelfSupervisedDataset:
    """ a dataset with observed events and optional user histories and timestamps
    for self-supervised training
    """
    event_df: pd.DataFrame
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self):
        _check_index(self.event_df, self.user_df, self.item_df)
        if "_hist_items" not in self.user_df:
            warnings.warn(f"{self} not applicable for sequence models.")
        if "_timestamps" not in self.user_df:
            warnings.warn(f"{self} not applicable for temporal models.")

    def __hash__(self):
        return id(self)


@dataclasses.dataclass(eq=False)
class Dataset(_SelfSupervisedDataset):
    """ A labeled dataset from 3 related tables:
        event_df: [USER_ID, ITEM_ID, TIMESTAMP]
        user_df: [USER_ID (index), TEST_START_TIME]
        item_df: [ITEM_ID (index)]
    Infer target labels from TEST_START_TIME (per user) and horizon.
    Filter test users/items by _hist_len.
    """
    horizon: float = float("inf")
    min_user_len: int = 1
    min_item_len: int = 1
    _is_synthetic_data: bool = False

    def __post_init__(self):
        _check_index(self.event_df, self.user_df, self.item_df)
        _check_more_inputs(self.event_df, self.user_df, self.item_df)

        print("augmenting and data tables")
        self.event_df = _mark_holdout(self.event_df, self.user_df, self.horizon)
        self.user_df = _augment_user_hist(self.user_df, self.event_df)
        self.item_df = _augment_item_hist(self.item_df, self.event_df)

        print("marking and cleaning test data")
        self.user_in_test = self.user_df[
            (self.user_df['_hist_len']>=self.min_user_len) &
            (self.user_df['TEST_START_TIME']<float("inf")) # exclude Group-A users
        ].copy()
        self.item_in_test = self.item_df[
            self.item_df['_hist_len']>=self.min_item_len
        ].copy()
        self.target_df = create_matrix(
            self.event_df[self.event_df['_holdout']==1].copy(),
            self.user_in_test.index, self.item_in_test.index, "df"
        )
        self.training_data = _SelfSupervisedDataset(
            self.event_df[self.event_df['_holdout']==0].copy(),
            self.user_df, self.item_df
        )

        print("inferring default parameters")
        self.default_user_rec_top_c = int(np.ceil(len(self.user_in_test) / 100))
        self.default_item_rec_top_k = int(np.ceil(len(self.item_in_test) / 100))

        print("Dataset initialized!")

    def __hash__(self):
        return id(self)

    def get_stats(self):
        return {
            'user_df': {
                '# warm users': len(self.user_in_test),
                '# all users': len(self.user_df),
                'avg hist len': self.user_in_test['_hist_len'].mean(),
                'avg hist span': self.user_in_test['_hist_span'].mean(),
                'horizon': self.horizon,
                'avg target items': df_to_coo(self.target_df).sum(axis=1).mean(),
            },
            'item_df': {
                '# warm items': len(self.item_in_test),
                '# all items': len(self.item_df),
                'avg hist len': self.item_in_test['_hist_len'].mean(),
                'avg target users': df_to_coo(self.target_df).sum(axis=0).mean(),
            },
            'event_df': {
                '# train events': sum(self.event_df['_holdout']==0),
                '# test events': df_to_coo(self.target_df).sum(),
                'default_user_rec_top_c': self.default_user_rec_top_c,
                'default_item_rec_top_k': self.default_item_rec_top_k,
                "user_ppl": perplexity(self.user_in_test['_hist_len']),
                "item_ppl": perplexity(self.item_in_test['_hist_len']),
            },
        }

    def print_stats(self, verbose=True):
        print(pd.DataFrame(self.get_stats()).T.stack().apply('{:.2f}'.format))
        if verbose:
            print(self.user_df.sample().iloc[0])
            print(self.item_df.sample().iloc[0])

    @warn_nan_output
    def transform(self, S, user_index=None, fill_value=float("nan")):
        """ reindex the score matrix to match with test users and items """
        if user_index is None:
            user_index = self.user_in_test.index
        return S.reindex(user_index, fill_value=fill_value) \
                .reindex(self.item_in_test.index, fill_value=fill_value, axis=1)
