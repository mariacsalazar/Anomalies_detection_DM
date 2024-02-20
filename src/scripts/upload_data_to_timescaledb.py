import pandas as pd
import read_write
from io import StringIO
import psycopg2


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
            # create table
            cursor.execute('''CREATE TABLE train (
                original_index INT,
                mapped_veh_id INT,
                timestamps_UTC TIMESTAMP not null,
                lat FLOAT,
                lon FLOAT,
                RS_E_InAirTemp_PC1 FLOAT,
                RS_E_InAirTemp_PC2 FLOAT,
                RS_E_OilPress_PC1 FLOAT,
                RS_E_OilPress_PC2 FLOAT,
                RS_E_RPM_PC1 FLOAT,
                RS_E_RPM_PC2 FLOAT,
                RS_E_WatTemp_PC1 FLOAT,
                RS_E_WatTemp_PC2 FLOAT,
                RS_T_OilTemp_PC1 FLOAT,
                RS_T_OilTemp_PC2 FLOAT,
                track_or_stop INT,
                temp FLOAT,
                dwpt FLOAT,
                rhum FLOAT,
                prcp FLOAT,
                snow FLOAT,
                wdir FLOAT,
                wspd FLOAT,
                wpgt FLOAT,
                pres FLOAT,
                tsun FLOAT,
                coco FLOAT,
                k_anomaly INT,
                if_anomaly INT,
                svm_anomaly INT,
                anomalies_triggered INT
            );''')
            conn.commit()
            # Truncate the existing table (i.e. remove all existing rows)
            cursor.execute(f"TRUNCATE {table_name}")
            conn.commit()

            cursor.execute(
                'SELECT create_hypertable(\'train\', \'timestamps_utc\');')
            conn.commit()
            # Now insert the brand-new data
            # Initialize a string buffer
            sio = StringIO()
            # Write the Pandas DataFrame as a CSV file to the buffer
            sio.write(df.to_csv(index=None, header=None))
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
            print("DataFrame uploaded to TimescaleDB")


def main():
    df = read_write.read_data(read_write.Filenames.data_with_cluster)
    # df = df[df.timestamps_UTC > '2023-06-01']
    df.timestamps_UTC = df.timestamps_UTC.dt.strftime('%Y-%m-%d %H:%M:%S.000')
    df.columns = map(str.lower, df.columns)
    upload_to_db_efficiently(df)


if __name__ == '__main__':
    main()
