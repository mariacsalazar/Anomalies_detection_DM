import pandas as pd
import read_write
from io import StringIO
import psycopg2
import time
import numpy as np

TOTAL_SPLITS = 640


def get_df_pages(df):
    '''Yields the pages according to the total splits'''
    total_splits = TOTAL_SPLITS
    page_size = int(len(df) / total_splits)
    sorted_df = df.sort_values('timestamps_utc')
    for i in range(total_splits):
        yield sorted_df.iloc[i * page_size:(i + 1) * page_size - 1].copy()


def upload_to_db_efficiently(df, table_name="train"):
    """
    Upload the stock price data to the TimescaleDB database as quickly and efficiently
    as possible by truncating (i.e. removing) the existing data and copying all-new data
    """

    with psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='postgres',
        user='postgres',
        password='password',
        connect_timeout=5
    ) as conn:
        with conn.cursor() as cursor:
            for small_df in get_df_pages(df):
                sio = StringIO()
                small_df.drop_duplicates(subset='mapped_veh_id', keep="last")
                small_df.timestamps_utc = pd.Timestamp.now() - pd.Timedelta(hours=1)
                small_df.timestamps_utc = small_df.timestamps_utc.dt.strftime(
                    '%Y-%m-%d %H:%M:%S.000')

                # Write the Pandas DataFrame as a CSV file to the buffer
                sio.write(small_df.to_csv(index=None, header=None))
                # Be sure to reset the position to the start of the stream
                sio.seek(0)
                cursor.copy_from(
                    file=sio,
                    table=table_name,
                    sep=",",
                    null="",
                    size=8192,
                    columns=df.columns
                )
                conn.commit()
                print("UPDATE SEND")
                time.sleep(1)  # sleep one second

    print("Finished uploading data")


def main():
    df = read_write.read_data(read_write.Filenames.small_subsample)
    df.columns = map(str.lower, df.columns)
    upload_to_db_efficiently(df)


if __name__ == '__main__':
    main()
