import read_write as io
import pandas as pd


def main():
    df = io.read_data(io.Filenames.data_with_cluster)
    mask = df.timestamps_UTC.dt.date.apply(str) == '2023-06-01'
    io.write_data(df[mask], io.Filenames.small_subsample)
    return


if __name__ == "__main__":
    main()
