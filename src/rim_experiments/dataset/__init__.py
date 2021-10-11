import pandas as pd, numpy as np

from ..util import extract_user_item, split_by_time, split_by_user
from .base import Dataset
from .prepare_netflix_data import prepare_netflix_data
from .prepare_ml_1m_data import prepare_ml_1m_data
from .prepare_yoochoose_data import prepare_yoochoose_data


def prepare_synthetic_data(split_fn_name,
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

    D = Dataset(event_df, user_df, item_df, horizon=1, print_stats=True,
        _is_synthetic_data=True)
    V = Dataset(event_df, valid_df, item_df, horizon=1)
    return (D, V)
