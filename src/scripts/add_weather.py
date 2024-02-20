import read_write as io
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
from meteostat import Hourly, Point
import time


NUMBER_OF_CLUSTERS = 10


def get_min_and_max_dates(df):
    mn, mx = df['timestamps_UTC'].min(), df['timestamps_UTC'].max()
    mn_datetime = datetime(mn.year, mn.month, mn.day)
    mx_datetime = datetime(mx.year, mx.month, mx.day)
    return (mn_datetime, mx_datetime)


def get_weather_information(cluster_centers, mn_datetime, mx_datetime):
    meteostat_points = [Point(cc[0], cc[1], None) for cc in cluster_centers]
    information_non_fetched = [
        Hourly(
            point,
            start=mn_datetime,
            end=mx_datetime) for point in meteostat_points]
    information = []
    print('=' * len(information_non_fetched))
    for info in information_non_fetched:
        information.append(info.fetch())
        # Wait for the api not to rate limit you
        time.sleep(5)
        print('=', end='')
    return information


def add_weather(data: pd.DataFrame):
    '''Adds weather data using kmeans approach.
    '''
    mn_datetime, mx_datetime = get_min_and_max_dates(data)
    # Train k means
    x_k_means = data[['lat', 'lon']].values
    kmeans = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        random_state=0,
        n_init="auto").fit(x_k_means)
    data['cluster_label'] = kmeans.labels_
    # Get weather data
    weather_information = get_weather_information(
        kmeans.cluster_centers_, mn_datetime, mx_datetime)
    small_index = weather_information[0].index
    aux_col = small_index.get_indexer(data['timestamps_UTC'], method='pad')
    big_weather = pd.concat(weather_information).reset_index(drop=True)
    big_weather.reset_index(inplace=True, names=['merge_key'])
    # Join the two tables
    data['merge_key'] = (
        kmeans.labels_ * len(weather_information[0])) + aux_col
    data = data.merge(
        big_weather,
        how='inner',
        left_on='merge_key',
        right_on='merge_key')
    # Clean up
    data.drop(['merge_key'], axis=1, inplace=True)
    data.drop(['cluster_label'], axis=1, inplace=True)
    return data


def main():
    data = io.read_data(io.Filenames.data_enhanced)
    data = add_weather(data)
    io.write_data(data, io.Filenames.data_with_weather)
    return


if __name__ == "__main__":
    main()
