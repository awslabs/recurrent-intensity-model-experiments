import pandas as pd, numpy as np, scipy as sp
import functools, collections, warnings, dataclasses
from ..util import create_matrix, cached_property, perplexity, \
                   timed, groupby_collect, df_to_coo, get_batch_size


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
class TrainingData:
    """ a training set with observed events and optional user histories and timestamps
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
class Dataset:
    """ a dataset with target_df from test users and items, reference to training data,
    optional horizon and mask for evaluation purposes.
    """
    target_df: pd.DataFrame         # index=USER_ID, column=ITEM_ID
    user_in_test: pd.DataFrame      # index=USER_ID
    item_in_test: pd.DataFrame      # index=ITEM_ID
    training_data: TrainingData
    horizon: float = float("inf")
    prior_score: pd.DataFrame = None    # index=USER_ID, column=ITEM_ID
    _fill_prior_value: float = 0

    def __post_init__(self):
        assert (self.target_df.index == self.user_in_test.index).all(), \
                        "target index must match with user index"
        assert (self.target_df.columns == self.item_in_test.index).all(), \
                        "target columns must match with item index"
        if self.prior_score is not None:
            assert (self.prior_score.shape == self.target_df.shape), \
                        "prior_score shape must match with test target_df"

        self.default_user_rec_top_c = int(np.ceil(len(self.user_in_test) / 100))
        self.default_item_rec_top_k = int(np.ceil(len(self.item_in_test) / 100))

    def __hash__(self):
        return id(self)

    def get_stats(self):
        return {
            'user_df': {
                '# warm users': len(self.user_in_test),
                '# all users': len(self.training_data.user_df),
                'avg hist len': self.user_in_test['_hist_len'].mean(),
                'avg hist span': self.user_in_test['_hist_span'].mean(),
                'horizon': self.horizon,
                'avg target items': df_to_coo(self.target_df).sum(axis=1).mean(),
            },
            'item_df': {
                '# warm items': len(self.item_in_test),
                '# all items': len(self.training_data.item_df),
                'avg hist len': self.item_in_test['_hist_len'].mean(),
                'avg target users': df_to_coo(self.target_df).sum(axis=0).mean(),
            },
            'event_df': {
                '# train events': len(self.training_data.event_df),
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
            print(self.user_in_test.sample().iloc[0])
            print(self.item_in_test.sample().iloc[0])

    def reindex(self, index, axis=1):
        if axis==1:
            user_in_test = self.user_in_test
            item_in_test = self.item_in_test.reindex(index, fill_value=0)
        else:
            raise NotImplementedError("user reindexing is not used in cvx-online")

        target_df = self.target_df.reindex(index, axis=axis, fill_value=0)

        prior_score = None if self.prior_score is None else \
            self.prior_score.reindex(index, axis=axis, fill_value=_fill_prior_value)

        return self.__class__(target_df, user_in_test, item_in_test,
            self.training_data, self.horizon, prior_score, self._fill_prior_value)


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
        (user_df['TEST_START_TIME']<float("inf")) # exclude Group-A users
    ].copy()
    item_in_test = item_df[
        item_df['_hist_len']>=min_item_len
    ].copy()
    target_df = create_matrix(
        event_df[event_df['_holdout']==1].copy(),
        user_in_test.index, item_in_test.index, "df"
    )
    training_data = TrainingData(
        event_df[event_df['_holdout']==0].copy(),
        user_df, item_df
    )

    if exclude_train:
        print("optionally excluding training events in predictions and targets")
        assert prior_score is None, "double configuration for prior score"

        exclude_df = create_matrix(
            event_df[event_df['_holdout']==0].copy(),
            user_in_test.index, item_in_test.index, "df"
        ).astype(bool)
        prior_score = exclude_df * float("-inf")    # sparse 0 * -inf = -inf

        mask_csr = df_to_coo(target_df).astype(bool) > df_to_coo(exclude_df).astype(bool)
        target_df = pd.DataFrame.sparse.from_spmatrix(
            df_to_coo(target_df).multiply(mask_csr), target_df.index, target_df.columns)

    D = Dataset(target_df, user_in_test, item_in_test, training_data,
        horizon, prior_score)
    print("Dataset created!")
    return D
