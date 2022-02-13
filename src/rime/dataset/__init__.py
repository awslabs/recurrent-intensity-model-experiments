import pandas as pd, numpy as np
import argparse
from ..util import extract_user_item
from .base import create_dataset, Dataset, create_temporal_splits, create_user_splits
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

    D = create_dataset(event_df, user_df, item_df, 100,
                       min_user_len=0, min_item_len=0, exclude_train=True)
    D = D.reindex(D.user_in_test.index.tolist() + ['oov user gets zero scores in some methods'], axis=0) \
         .reindex(D.item_in_test.index.tolist() + ['oov item gets zero scores in most methods'], axis=1)

    return (D,)


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
        D, V = create_temporal_splits(event_df, user_df, item_df, 4, 1, 1)
    elif split_fn_name == 'split_by_user':
        D, V = create_user_splits(event_df, user_df.assign(_in_GroupA=user_df.index % 2),
                                  item_df, 3, 1)
    else:
        raise ValueError(f"unknown {split_fn_name}")

    D._is_synthetic_data = True  # for hawkes_poisson verification purposes
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
