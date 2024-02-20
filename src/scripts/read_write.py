import pandas as pd
from enum import Enum
from fastparquet import write
import os
import pickle


def directory_up(path: str, n: int):
    for _ in range(n):
        path = directory_up(path.rpartition("/")[0], 0)
    return path


root_path = os.path.dirname(os.path.realpath(__file__))
# Change working directory to root of the project.
os.chdir(directory_up(root_path, 2))


class Filenames(Enum):
    data_with_weather = 'data_with_weather'
    data_cleaned = 'data_cleaned'
    data_enhanced = 'data_enhanced'
    original_data = 'ar41_for_ulb'
    outliers_fixed = 'outliers_fixed'
    data_labeled = 'data_labeled'
    data_transformed_woe = 'data_transformed_woe'
    data_with_cluster = 'data_with_cluster'
    model_kmeans = 'model_kmeans'
    model_if = 'model_if'
    model_svm = 'model_svm'
    small_subsample = 'small_subsample'
    scaler = 'scaler'
    data_for_dashboard = 'data_for_dashboard'
    detect_anomaly = 'detect_anomaly'


def get_route_from_filename_csv(filename: Filenames) -> str:
    return './data/{}.csv'.format(filename.value)


def get_route_from_filename_parquet(filename: Filenames) -> str:
    return './data/{}.parq'.format(filename.value)


def get_route_from_filename_pickle(filename: Filenames) -> str:
    return './model/{}.pkl'.format(filename.value)


def write_data(data_frame: pd.DataFrame, filename: Filenames, parquet=True):
    '''If parquet is true, it writes to a parquet, otherwise, it writes to a csv
    '''
    if parquet:
        data_frame.to_parquet(
            get_route_from_filename_parquet(filename),
            engine='fastparquet')
        return
    data_frame.to_csv(get_route_from_filename_csv(filename), index=False)


def read_data(filename: Filenames, parquet=True):
    if parquet:
        return pd.read_parquet(
            get_route_from_filename_parquet(filename),
            engine='fastparquet')
    return pd.read_csv(
        get_route_from_filename_csv(filename),
        sep=';',
        index_col=0)


def write_pickle(trained_model, filename: Filenames):
    with open(get_route_from_filename_pickle(filename), 'wb') as file:
        pickle.dump(trained_model, file)


def read_pickle(filename: Filenames):
    with open(get_route_from_filename_pickle(filename), "rb") as file:
        return pickle.load(file)
