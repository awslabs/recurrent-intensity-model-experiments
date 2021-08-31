Put a symlink to each data source we use in the experiments.
Additional details are shown in `rim_experiments.dataset.*`.

* [MovieLens ml-1m](https://grouplens.org/datasets/movielens/1m/). We create an intermediate cache at `./ml-1m/ratings.dat`.
```
>>> _load_sort_ml_1m_data(data_path)
         USER_ID  ITEM_ID   TIMESTAMP
1000138     6040      858   956703932
999873      6040      593   956703954
...          ...      ...         ...
```
name    |   raw     |  train  | warm / clean test
--------|----------:|--------:|--------:
users   |     6,040 |   6,040 |  3,020
items   |     3,706 |   3,706 |  3,669
events  | 1,000,209 | 762,016 | 37,597

* [Netflix](https://www.kaggle.com/netflix-inc/netflix-prize-data). We create an intermediate cache at `./Netflix/nf.parquet`.
```
>>> pd.read_parquet(data_path)
           USER_ID    ITEM_ID   TIMESTAMP
0          1488844      1.txt  1125964800
1           822109      1.txt  1115942400
...            ...        ...         ...
```
name    |   raw       |   train   | warm / clean test
--------|------------:|----------:|--------:
users   |     480,189 |    33,423 |  32,238
items   |      17,770 |    16,378 |  16,217
events  | 100,480,507 | 2,437,151 | 187,096

* [Yoochoose](https://www.kaggle.com/chadgostopp/recsys-challenge-2015). We create an intermediate cache at `./yoochoose-data/yoochoose-combined.csv`, which combines clicks and buys data.
```
>>> pd.read_csv(data_path)[["USER_ID", "ITEM_ID", "TIMESTAMP"]]
           USER_ID    ITEM_ID     TIMESTAMP
0                1  214536502  1.396868e+09
1                1  214536500  1.396868e+09
...            ...        ...           ...
```
name    | clicks + buys |   train   | warm / clean test
--------|--------------:|----------:|--------:
users   |     9,249,729 |   203,523 |  71,784
items   |        52,739 |    16,517 |  11,431
events  |    34,154,697 | 1,087,267 |  49,500
