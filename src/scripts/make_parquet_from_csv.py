import read_write as io
import pandas as pd


def main():
    df = io.read_data(
        io.Filenames.original_data,
        parquet=False)
    df['timestamps_UTC'] = pd.to_datetime(df.timestamps_UTC)
    df['mapped_veh_id'] = df.mapped_veh_id.apply(int)
    df.index = df.index.set_names(['original_index'])
    io.write_data(df.reset_index(), io.Filenames.original_data)
    return


if __name__ == "__main__":
    main()
