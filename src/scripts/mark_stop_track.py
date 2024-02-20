import read_write as io
import pandas as pd
from scipy import spatial
import geopandas as gpd
import clean_location
import numpy as np

MAX_DISTANCE_FOR_TRACK_STOP_DIVISION_X = 3000
MAX_DISTANCE_FOR_TRACK_STOP_DIVISION_Y = 3000


def divide_track_stop(data):
    '''Divide the data into track and stop category, 0 for track and 1 for stop

    returns a dataframe with a new column track_or_stop'''
    meters = clean_location.get_df_proyected_in_meters(data)
    geo_data = pd.concat([data, meters[["x", "y"]]], axis=1)
    geo_data['timestamps_UTC'] = pd.to_datetime(geo_data.timestamps_UTC)
    geo_data['date'] = geo_data['timestamps_UTC'].dt.date
    grouped_data = geo_data.groupby(['mapped_veh_id', 'date']).agg(
        min_x=('x', 'min'),
        max_x=('x', 'max'),
        min_y=('y', 'min'),
        max_y=('y', 'max')
    ).reset_index()
    grouped_data['track_or_stop'] = (
        (grouped_data['max_x'] -
         grouped_data['min_x'] < MAX_DISTANCE_FOR_TRACK_STOP_DIVISION_X) & (
            grouped_data['max_y'] -
            grouped_data['min_y'] < MAX_DISTANCE_FOR_TRACK_STOP_DIVISION_Y)).astype(int)
    geo_data = pd.merge(geo_data,
                        grouped_data[['mapped_veh_id',
                                      'date',
                                      'track_or_stop']],
                        on=['mapped_veh_id',
                            'date'],
                        how='left')
    geo_data['track_or_stop'] = geo_data['track_or_stop'].fillna(1).astype(int)
    data = geo_data.drop(columns=['date', 'x', 'y'])
    return data


def clean():
    data = io.read_data(io.Filenames.outliers_fixed)
    data = divide_track_stop(data)
    io.write_data(data, io.Filenames.data_enhanced)


def main():
    clean()
    return


if __name__ == "__main__":
    main()
