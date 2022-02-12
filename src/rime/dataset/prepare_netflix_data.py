import os, pandas as pd
from datetime import datetime
from ..util import extract_user_item
from .base import create_temporal_splits


def prepare_netflix_data(
    data_path="data/Netflix/nf.parquet",
    train_begin=datetime(2005, 1, 1).timestamp(),
    valid_start=datetime(2005, 6, 1).timestamp(),
    test_start=datetime(2005, 6, 15).timestamp(),
    test_end=datetime(2005, 6, 29).timestamp(),
    user_mod=10,
    item_mod=1,
    num_V_extra=10,
    title_path=None,
    **kw
):
    event_df = pd.read_parquet(data_path)
    print(event_df.head())

    event_df = event_df[
        (event_df['TIMESTAMP'] >= train_begin) &
        (event_df['TIMESTAMP'] < test_end) &
        (event_df['USER_ID'].astype(int) % user_mod == 0) &
        (event_df['ITEM_ID'].apply(lambda x: int(x[:-4])) % item_mod == 0)
    ].sample(frac=1, random_state=0).sort_values(['USER_ID', 'TIMESTAMP'], kind='mergesort')
    print(f"{event_df.describe()}")

    user_df, item_df = extract_user_item(event_df)

    if title_path is None:
        title_path = os.path.join(os.path.dirname(data_path), 'movie_titles.csv')
    if os.path.exists(title_path):
        movie_titles = pd.read_csv(title_path, encoding='latin1',
                                   names=['_ITEM_ID_number', '_', 'TITLE'])
        movie_titles.index = movie_titles['_ITEM_ID_number'].apply(lambda x: "{:d}.txt".format(x))
        item_df = item_df.join(movie_titles[['TITLE']])
        assert item_df['TITLE'].notnull().all(), "movie titles should not be missing"

    return create_temporal_splits(
        event_df, user_df, item_df, test_start,
        horizon=test_end - test_start,
        validating_horizon=test_start - valid_start,
        num_V_extra=num_V_extra,
        **kw)
