import read_write as io
import pandas as pd
from scipy import spatial
import geopandas as gpd
import numpy as np

MAX_DISTANCE_FOR_OUTLIER_DETECTION_IN_METERS = 3000


def get_df_proyected_in_meters(data: pd.DataFrame) -> pd.DataFrame:
    '''Gets the data into a EPSG:3857

    When calculating distances is important to project into a
    system that uses meters, then when we want to calculate distances
    they are significant to us.
    '''
    geo_data = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.lon, data.lat),
        crs="EPSG:4326"
    )
    geo_data['geometry'] = geo_data.geometry.to_crs('EPSG:3857')
    geo_data['x'] = geo_data.geometry.apply(lambda x: x.x)
    geo_data['y'] = geo_data.geometry.apply(lambda x: x.y)
    return geo_data[['x', 'y']]


def get_outliers_by_distance(data: pd.DataFrame):
    '''Given the original dataset return the mask for the dataset without outliers

    The MAX_DISTANCE_FOR_OUTLIER_DETECTION_IN_METERS gives the distance
    for the outlier detection, this function calculates the avg distance in
    meters from every point to its 4 nearest neighbours. Then, it marks them
    for deletion if the surpass the threshold set by
    MAX_DISTANCE_FOR_OUTLIER_DETECTION_IN_METERS.

    For calculating distances the function uses a kdtree from scipy spatial.
    '''
    def get_knn_distance(row):
        '''
        Auxiliary function to get nearest neighbours avg distance
        '''
        distance, _ = kdtree.query([row.x, row.y], k=5)
        return np.average(distance[1:])

    spatial_df = get_df_proyected_in_meters(data)
    kdtree = spatial.KDTree(spatial_df.values)
    spatial_df['distance'] = spatial_df.apply(get_knn_distance, axis=1)
    return spatial_df['distance'] < MAX_DISTANCE_FOR_OUTLIER_DETECTION_IN_METERS


def clean_position(data: pd.DataFrame, distance_cleaning=True) -> pd.DataFrame:
    '''Clean erroneus latitude and longitude

    returns a dataframe with the data cleaned
    '''
    if distance_cleaning:
        distance_outliers_mask = get_outliers_by_distance(data)
        data = data[distance_outliers_mask]

    return data


def clean():
    data = io.read_data(io.Filenames.original_data)
    data = clean_position(data, distance_cleaning=True)
    io.write_data(data, io.Filenames.data_cleaned)


def main():
    clean()
    return


if __name__ == "__main__":
    main()
