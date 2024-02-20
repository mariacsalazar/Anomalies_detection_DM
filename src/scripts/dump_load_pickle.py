import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import read_write as io
from joblib import Parallel, delayed

NUMBER_OF_CORES = 14


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

    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def parallelize_svm_prediction(scaled_data, model):
    number_of_columns = scaled_data.shape[1]

    parallel = Parallel(n_jobs=NUMBER_OF_CORES)
    results = parallel(delayed(model.predict)(
        scaled_data[i].reshape(-1, number_of_columns)) for i in range(scaled_data.shape[0]))

    return np.vstack(results).flatten()


class DetectAnomaly:
    def __init__(self, args):
        self.args = args

    def train_model(self, data):
        # select_data
        con_1 = (data["track_or_stop"] == 0)
        move = data[con_1]
        # random_state for reproducibility
        selected_data = move.sample(n=200_000, random_state=42)
        del data
        # select_column_standardize
        x = selected_data[['RS_E_InAirTemp_PC1',
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
        X = scaler.transform(x)
        self.scaler_model = scaler
        print("Scaler trained!")
        # kmeans
        kmeans = KMeans(n_clusters=self.args.kmeans_k)
        kmeans.fit(X)
        self.kmeans_model = kmeans
        print("Kmeans trained!")
        # if
        # Adjust contamination based on your dataset
        isolation_forest = IsolationForest(
            contamination=self.args.if_contamination)
        if_model = isolation_forest.fit(X)
        self.if_model = if_model
        print("Isolation forest trained!")
        # svm
        # Adjust nu based on your scenario
        svm_model = OneClassSVM(nu=self.args.svm_nu)
        svm_model.fit(X)
        self.svm_model = svm_model
        print("SVM trained!")

    def apply_model(self, data):
        # apply scaler
        X = self.scaler_model.transform((data[['RS_E_InAirTemp_PC1',
                                               'RS_E_InAirTemp_PC2',
                                               'RS_E_OilPress_PC1',
                                               'RS_E_OilPress_PC2',
                                               'RS_E_RPM_PC1',
                                               'RS_E_RPM_PC2',
                                               'RS_E_WatTemp_PC1',
                                               'RS_E_WatTemp_PC2',
                                               'RS_T_OilTemp_PC1',
                                               'RS_T_OilTemp_PC2']]).reset_index(drop=True))
        # Apply kmeans
        cluster_assignments = self.kmeans_model.predict(X)
        distances_to_center = pairwise_distances_argmin_min(
            X, self.kmeans_model.cluster_centers_)[1]
        threshold = np.percentile(
            distances_to_center,
            self.args.kmeans_percentile)
        anomalies_indices = np.where(distances_to_center > threshold)[0]
        kmeans_anomaly_result = - \
            (data.index.isin(anomalies_indices).astype(int) * 2 - 1)
        data['k_cluster'], data['k_anomaly'] = cluster_assignments, kmeans_anomaly_result
        print("Kmeans predicted!")
        # apply isolation forest
        data['if_anomaly'] = self.if_model.predict(X)
        print("Isolation forest predicted!")
        # apply svm
        data["svm_anomaly"] = parallelize_svm_prediction(X, self.svm_model)
        print("SVM predicted!")
        data['anomalies_triggered']= -((data.svm_anomaly-1)+(data.if_anomaly-1) +(data.k_anomaly-1))/2
        data.anomalies_triggered = data.anomalies_triggered.apply(int)
        # return data


def main():
    # train model
    args = get_args()
    data = io.read_data(io.Filenames.data_with_weather)
    detect_anomaly = DetectAnomaly(args)
    detect_anomaly.train_model(data)
    io.write_pickle(detect_anomaly, io.Filenames.detect_anomaly)
    # apply model
    detect_anomaly_loaded = io.read_pickle(io.Filenames.detect_anomaly)
    data = io.read_data(io.Filenames.data_with_weather)
    # randomly sample some to test
    detect_anomaly_loaded.apply_model(data)
    print(data['k_anomaly'].value_counts())
    print(data['if_anomaly'].value_counts())
    print(data['svm_anomaly'].value_counts())
    print(data.head())
    io.write_data(data.drop(['k_cluster'], axis=1), io.Filenames.data_with_cluster)


if __name__ == "__main__":
    main()
