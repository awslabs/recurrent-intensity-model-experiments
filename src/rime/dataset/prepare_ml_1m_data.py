import os, pandas as pd
from ..util import extract_user_item, sample_groupA
from .base import create_user_splits


def prepare_ml_1m_data(data_path="data/ml-1m/ratings.dat",
                       seed=0, second_half_only=True,
                       title_path=None,
                       **kw):

    event_df = pd.read_csv(
        data_path, sep="::", names=["USER_ID", "ITEM_ID", "_", "TIMESTAMP"]
    ).sample(frac=1, random_state=seed).sort_values("TIMESTAMP", kind="mergesort")

    if second_half_only:
        event_df = event_df[
            event_df.groupby("USER_ID")["TIMESTAMP"].rank(method="first", pct=True) >= 0.5]

    user_df, item_df = extract_user_item(event_df)

    if title_path is None:
        title_path = os.path.join(os.path.dirname(data_path), 'movies.dat')
    if os.path.exists(title_path):
        movies_titles = pd.read_csv(title_path, encoding='latin1', sep='::',
                                    names=['ITEM_ID', 'TITLE', '_']).set_index('ITEM_ID')
        item_df = item_df.assign(_preserve_order=lambda x: x.index) \
            .join(movies_titles[['TITLE']], on='_preserve_order').drop('_preserve_order', axis=1)
        assert item_df['TITLE'].notnull().all(), "movie titles should not be missing"

    in_GroupA = sample_groupA(user_df, seed=seed + 888)

    test_start_rel = (user_df['_Tmax'] - user_df['_Tmin']).quantile(0.5)
    horizon = test_start_rel * 1.0
    print({"test_start_rel": test_start_rel, "horizon": horizon})

    return create_user_splits(
        event_df,
        user_df.assign(_in_GroupA=in_GroupA),
        item_df, test_start_rel, horizon, num_V_extra=1, **kw)
