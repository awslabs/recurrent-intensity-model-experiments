import pandas as pd
from ..util import extract_user_item, filter_min_len, sample_groupA
from .base import create_user_splits


def prepare_yoochoose_data(
    data_path="data/yoochoose-data/yoochoose-combined.csv",
    seed=0, user_sample_frac=0.1, min_user_len=4, min_item_len=10,
    drop_duplicates=True, num_V_extra=1,
    **kw
):
    event_df = pd.read_csv(data_path).sort_values('TIMESTAMP', kind="mergesort")
    if drop_duplicates:
        event_df = event_df.drop_duplicates(['USER_ID', 'ITEM_ID'])
    event_df = _sample_by_user(event_df, user_sample_frac, seed + 10)
    event_df = filter_min_len(event_df, min_user_len, min_item_len)

    user_df, item_df = extract_user_item(event_df)
    in_GroupA = sample_groupA(user_df, seed=seed + 888)
    print(len(event_df), len(user_df), len(item_df))

    test_start_rel = (user_df['_Tmax'] - user_df['_Tmin']).quantile(0.5)
    horizon = test_start_rel * 1.0
    print({"test_start_rel": test_start_rel, "horizon": horizon})

    return create_user_splits(
        event_df,
        user_df.assign(_in_GroupA=in_GroupA),
        item_df, test_start_rel, horizon, num_V_extra=num_V_extra,
        min_user_len=min_user_len, min_item_len=min_item_len, **kw)


def _sample_by_user(event_df, frac, seed):
    users = event_df.groupby("USER_ID").size().sample(frac=frac, random_state=seed)
    return event_df[event_df["USER_ID"].isin(users.index)]
