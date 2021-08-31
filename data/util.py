import urllib.request, zipfile, os, io, pandas


def download_ml_1m(
    url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    ):
    with urllib.request.urlopen(url) as html:
        file_like = io.BytesIO(html.read())
        with zipfile.ZipFile(file_like) as z:
            z.extractall('.')


def extract_netflix(
    input_archive_split_by_movies="Netflix/archive.zip",
    output_file="Netflix/nf.parquet",
    ):
    user_item_time = []
    with zipfile.ZipFile(input_archive_split_by_movies) as z:
        print(z.namelist())
        for i in range(4):
            file_name = f"combined_data_{i+1}.txt"
            with z.open(file_name) as f:
                item_id = None
                for line in f:
                    line = line.decode().strip('\n')
                    if line.endswith(':'):
                        item_id = line.replace(':', '.txt')
                    else:
                        user_id, _, time = line.split(',')
                        user_item_time.append((user_id, item_id, time))
            print(f"done reading {file_name}")

    df = pandas.DataFrame(user_item_time, columns=['USER_ID', 'ITEM_ID', 'TIMESTAMP'])
    df['USER_ID'] = df['USER_ID'].astype(int)
    df['TIMESTAMP'] = df['TIMESTAMP'].values.astype("datetime64[s]").astype(int)
    print(df.info(verbose=True))
    df.to_parquet(output_file)
    return df


def combine_yoochoose(
    input_archive="yoochoose-data/archive.zip",
    output_file="yoochoose-data/yoochoose-combined.csv",
    ):
    with zipfile.ZipFile(input_archive) as z:
        print(z.namelist())
        clicks = pandas.read_csv(z.open("yoochoose-data/yoochoose-clicks.dat"),
            names=["USER_ID", "TIMESTAMP", "ITEM_ID", "_category"])
        buys = pandas.read_csv(z.open("yoochoose-data/yoochoose-buys.dat"),
            names=["USER_ID", "TIMESTAMP", "ITEM_ID", "_price", "_quantity"])

    df = pandas.concat([clicks.iloc[:, :3], buys.iloc[:, :3]])
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64') \
                        .values.astype('datetime64[ms]').astype(int) / 1e3
    df = df.sort_values('TIMESTAMP', kind='mergesort')
    print(df.info(verbose=True))
    df.to_csv(output_file)
    return df, clicks, buys
