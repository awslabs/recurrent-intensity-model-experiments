import pandas as pd, numpy as np, scipy.sparse as sps
import argparse
from ..util import extract_user_item, split_by_time, split_by_user
from .base import create_dataset, Dataset, _augment_user_hist, _augment_item_hist
from .prepare_netflix_data import prepare_netflix_data
from .prepare_ml_1m_data import prepare_ml_1m_data
from .prepare_yoochoose_data import prepare_yoochoose_data


def prepare_synthetic_data(split_fn_name, exclude_train=False,
    num_users=300, num_items=200, num_events=10000):
    """ prepare synthetic data for end-to-end unit tests """
    event_df = pd.DataFrame({
        'USER_ID': np.random.choice(num_users, num_events),
        'ITEM_ID': np.random.choice(num_items, num_events)+1, # pandas bug
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
    D._is_synthetic_data = True # for hawkes_poisson verification purposes
    D.print_stats()
    V = create_dataset(event_df, valid_df, item_df, 1, exclude_train=exclude_train)
    return (D, V)


def prepare_minimal_dataset():
    """ minimal dataset to enable main workflow in unit tests """
    T_split = 10

    event_df = pd.DataFrame([
        ["u1", "i1", 3, False],
        ["u2", "i2", 5, False],
        ["u3", "i3", 7, False],
        ["u3", "i4", 9, False],
        ], columns=["USER_ID", "ITEM_ID", "TIMESTAMP", "_holdout"])

    user_df, item_df = extract_user_item(event_df)
    user_df = user_df.reindex(["u1", "u2", "u3", "u4"]) # include all test users
    item_df = item_df.reindex(["i1", "i2", "i3", "i4", "i5"]) # include all test items

    user_df['TEST_START_TIME'] = T_split
    user_df = _augment_user_hist(user_df, event_df) # add user history information
    del user_df['TEST_START_TIME'] # the last in _timestamps is set to TEST_START_TIME

    item_df = _augment_item_hist(item_df, event_df) # add item statistics

    user_in_test = pd.DataFrame.from_dict({
        "u1": [["i1"],       [3, T_split]   ],
        "u3": [["i3", "i4"], [7, 9, T_split]],
        "u4": [[],           [T_split]      ],
        }, columns=["_hist_items", "_timestamps"], orient="index")
    user_in_test["_hist_len"] = user_in_test["_hist_items"].apply(len)
    user_in_test["_hist_span"] = user_in_test["_timestamps"].apply(np.ptp)

    item_in_test = pd.DataFrame.from_dict({
        "i1": [1],
        "i5": [0],
        }, columns=["_hist_len"], orient="index")

    target_csr = sps.csr_matrix((len(user_in_test), len(item_in_test)))

    D = Dataset(
        user_in_test=user_in_test, item_in_test=item_in_test, target_csr=target_csr,
        training_data=argparse.Namespace(
            user_df=user_df, item_df=item_df, event_df=event_df
        ),
    )
    return (D, None)
