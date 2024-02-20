import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import read_write as io


def get_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Add cluster result to the database')

    # Define the arguments
    # Kmeans
    parser.add_argument(
        '-kmeans_k',
        type=int,
        default=5,
        help='The number of clusters in Kmeans')
    parser.add_argument(
        '-kmeans_percentile',
        type=int,
        default=95,
        help='Determine a threshold for anomaly detection based on percentile')
    # DBSCAN
    parser.add_argument(
        '-dbscan_eps',
        type=float,
        default=0.5,
        help='The eps in dbscan')
    parser.add_argument(
        '-dbscan_min_samples',
        type=int,
        default=20,
        help='The min_samples in dbscan')
    # Isolation forest
    parser.add_argument(
        '-if_contamination',
        type=float,
        default=0.05,
        help='The contamination in isolation forest, contamination is to specify the percentage of anomalies')
    # SVM
    parser.add_argument(
        '-svm_nu',
        type=float,
        default=0.05,
        help='The nu in SVM, nu is to specify the percentage of anomalies')
    # train
    parser.add_argument(
        '-train_num',
        type=int,
        default=181,
        help='Specify the train number')

    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def select_train(data, args):
    con = (data["mapped_veh_id"] == args.train_num)
    selected_data = data[con]
    new_data = selected_data.reset_index(drop=True)
    return new_data


def select_data(data):
    con_1 = (data["track_or_stop"] == 0)
    move = data[con_1]
    # random_state for reproducibility
    selected_data = move.sample(n=200000, random_state=42)
    return selected_data


def select_columns_standardize(data):
    x = data[['RS_E_InAirTemp_PC1',
              'RS_E_InAirTemp_PC2',
              'RS_E_OilPress_PC1',
              'RS_E_OilPress_PC2',
              'RS_E_RPM_PC1',
              'RS_E_RPM_PC2',
              'RS_E_WatTemp_PC1',
              'RS_E_WatTemp_PC2',
              'RS_T_OilTemp_PC1',
              'RS_T_OilTemp_PC2']]

    x = x.reset_index(drop=True)
    scaler = StandardScaler()
    scaler.fit(x)
    io.write_pickle(scaler, io.Filenames.scaler)
    X = scaler.transform(x)
    return X


def get_kmeans_result(X, data, args):
    # Assuming X is your feature matrix
    kmeans = KMeans(n_clusters=args.kmeans_k)
    kmeans.fit(X)
    io.write_pickle(kmeans, io.Filenames.model_kmeans)

    # Step 4: Get the cluster assignments for each data point
    cluster_assignments = kmeans.predict(X)

    # Step 5: Calculate the distances from each point to its cluster center
    distances_to_center = pairwise_distances_argmin_min(
        X, kmeans.cluster_centers_)[1]

    # Step 6: Determine a threshold for anomaly detection (e.g., based on a
    # percentile)
    threshold = np.percentile(
        distances_to_center,
        args.kmeans_percentile)  # Adjust as needed

    # Step 7: Identify anomalies based on the chosen threshold
    anomalies_indices = np.where(distances_to_center > threshold)[0]

    kmeans_anomaly_result = - \
        (data.index.isin(anomalies_indices).astype(int) * 2 - 1)
    return cluster_assignments, kmeans_anomaly_result


def get_dbscan_result(X, args):
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    dbscan.fit(X)
    db_cluster = dbscan.fit_predict(X)
    return db_cluster


def get_if_result(X, args):
    # Adjust contamination based on your dataset
    isolation_forest = IsolationForest(contamination=args.if_contamination)
    # if_labels = isolation_forest.fit_predict(X)
    if_model = isolation_forest.fit(X)
    io.write_pickle(if_model, io.Filenames.model_if)
    if_labels = if_model.predict(X)
    return if_labels


def get_svm_result(X, args):
    # Train One-Class SVM model
    svm_model = OneClassSVM(nu=args.svm_nu)  # Adjust nu based on your scenario
    svm_model.fit(X)
    io.write_model(svm_model, io.Filenames.model_svm)
    # Predictions: 1 for normal, -1 for anomaly
    svm_result = svm_model.predict(X)
    return svm_result


def main():
    args = get_args()
    data = io.read_data(io.Filenames.data_with_weather)
    selected_data = select_data(data)
    X = select_columns_standardize(selected_data)
    selected_data['k_cluster'], selected_data['k_anomaly'] = get_kmeans_result(
        X, selected_data, args)
    # selected_data['db_cluster'] = get_dbscan_result(X,args)
    selected_data['if_cluster'] = get_if_result(X, args)
    selected_data['svm_cluster'] = get_svm_result(X, args)
    io.write_data(selected_data, io.Filenames.data_with_cluster)
    return


if __name__ == "__main__":
    main()
