import pandas as pd
from datetime import datetime
from ..util import extract_user_item, split_by_time
from .base import create_dataset


def prepare_netflix_data(
    data_path = "data/Netflix/nf.parquet",
    train_begin = datetime(2005,1,1).timestamp(),
    valid_start = datetime(2005,6,1).timestamp(),
    test_start = datetime(2005,6,15).timestamp(),
    test_end = datetime(2005,6,29).timestamp(),
    user_mod = 10,
    item_mod = 1,
    exclude_train = False,
    ):
    event_df = pd.read_parquet(data_path)
    print(event_df.head())

    event_df = event_df[
        (event_df['TIMESTAMP'] >= train_begin) &
        (event_df['TIMESTAMP'] < test_end) &
        (event_df['USER_ID'].astype(int) % user_mod == 0) &
        (event_df['ITEM_ID'].apply(lambda x: int(x[:-4])) % item_mod == 0)
    ].sample(frac=1, random_state=0).sort_values('TIMESTAMP', kind='mergesort')
    print(f"{event_df.describe()}")

    user_df, item_df = extract_user_item(event_df)
    user_df, valid_df = split_by_time(user_df, test_start, valid_start)

    D = create_dataset(event_df, user_df, item_df, test_end-test_start,
        exclude_train=exclude_train)
    D.print_stats()
    V = create_dataset(event_df, valid_df, item_df, test_start-valid_start,
        exclude_train=exclude_train)
    return (D, V)
