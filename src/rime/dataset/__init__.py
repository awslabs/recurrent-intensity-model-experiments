import pandas as pd, numpy as np
import argparse
from ..util import extract_user_item, split_by_time, split_by_user, create_matrix
from .base import (create_dataset, Dataset, _mark_and_trim_holdout, _reindex_user_hist,
                   _augment_user_hist, _augment_item_hist)
from .prepare_netflix_data import prepare_netflix_data
from .prepare_ml_1m_data import prepare_ml_1m_data
from .prepare_yoochoose_data import prepare_yoochoose_data


def prepare_minimal_dataset():
    """ minimal dataset to enable main workflow in unit tests """
    event_df = pd.DataFrame([
        ["u1", "i1", 3],
        ["u2", "i2", 5],
        ["u3", "i3", 7],
        ["u3", "i4", 9],
    ], columns=["USER_ID", "ITEM_ID", "TIMESTAMP"])

    user_df = pd.Series({
        "u1": 4,
        "u2": float("inf"),  # +inf=training-only user, unless added after create_dataset
        "u3": 9,
    }).to_frame("TEST_START_TIME")

    item_df = pd.DataFrame(index=["i1", "i2", "i3", "i4"])

    horizon = 100

    # mark and trim _holdout by [TEST_START_TIME, TEST_START_TIME + horizon)
    # can be customized by setting _holdout as 0=training and 1=testing.
    event_df = _mark_and_trim_holdout(event_df, user_df, horizon)
    user_df = _augment_user_hist(user_df, event_df)  # add _hist_items, _hist_ts, _hist_len
    item_df = _augment_item_hist(item_df, event_df)  # add _hist_len

    training_data = argparse.Namespace(
        user_df=user_df, item_df=item_df, event_df=event_df
    )

    # Here is a walk-through of create_dataset function, except that the function
    # automatically includes users and items by min_user/item_len and TEST_START_TIME<inf,
    # whereas we manually choose them.
    # New users/items will get zero prediction scores; they are better included in
    # training data, albeit having empty lists of events.
    user_in_test = _reindex_user_hist(user_df[[
        '_hist_items', '_hist_len',  '_hist_ts', 'TEST_START_TIME',
    ]], ['u1', 'u3', 'oov_users_get_all_zero_scores'])

    item_in_test = item_df[['_hist_len']].reindex([
        'i1', 'i4', 'oov_items_get_all_zero_scores'
    ], fill_value=0)

    target_csr = create_matrix(event_df[event_df['_holdout'] == 1],
                               user_in_test.index, item_in_test.index, 'csr')

    # excluding seen user-item pairs leads to performance with matrix factorization methods
    prior_score = create_matrix(event_df[event_df['_holdout'] == 0],
                                user_in_test.index, item_in_test.index, 'csr') * -1e10

    # test targets should only include predictable user-item pairs
    target_csr = target_csr.multiply(target_csr.astype(bool) > (prior_score < 0))

    D = Dataset(
        user_in_test=user_in_test, item_in_test=item_in_test, target_csr=target_csr,
        horizon=horizon, prior_score=prior_score, training_data=training_data
    )
    D.print_stats()
    return (D, None)


def prepare_synthetic_data(split_fn_name, exclude_train=False,
                           num_users=300, num_items=200, num_events=10000):
    """ prepare synthetic data for end-to-end unit tests """
    event_df = pd.DataFrame({
        'USER_ID': np.random.choice(num_users, num_events),
        'ITEM_ID': np.random.choice(num_items, num_events),
        'TIMESTAMP': np.random.uniform(0, 5, num_events),
    }).sort_values(["USER_ID", "TIMESTAMP"])

    user_df, item_df = extract_user_item(event_df)

    if split_fn_name == 'split_by_time':
        user_df, valid_df = split_by_time(user_df, 4, 3)
    elif split_fn_name == 'split_by_user':
        user_df, valid_df = split_by_user(user_df, user_df.index % 2, 3)
    else:
        raise ValueError(f"unknown {split_fn_name}")

    D = create_dataset(event_df, user_df, item_df, 1, exclude_train=exclude_train)
    D._is_synthetic_data = True  # for hawkes_poisson verification purposes
    D.print_stats()
    V = create_dataset(event_df, valid_df, item_df, 1, exclude_train=exclude_train)
    return (D, V)


def prepare_simple_pattern():
    """ Transformer(D.training_data.item_df, max_epochs=100).fit(D.training_data) # flaky
    RNN(D.training_data.item_df, max_epochs=50).fit(D.training_data) # stable
    """
    event_df = pd.DataFrame({
        "USER_ID": 1,
        "ITEM_ID": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
        "TIMESTAMP": 1 + np.arange(12),
    })
    user_df, item_df = extract_user_item(event_df)
    user_df['TEST_START_TIME'] = 12
    D = create_dataset(event_df, user_df, item_df, 1)
    D.print_stats()
    return (D, None)
