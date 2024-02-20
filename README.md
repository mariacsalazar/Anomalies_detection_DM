# Data Mining Train Project

![AR41](https://upload.wikimedia.org/wikipedia/commons/c/c9/Foto_van_de_MW41_%282%29.png)
Repository for the data mining project at the ULB 2023-Fall

# Installation
The virtual environment assumes python 3.11. Create your virtual environment and activate it using:

```sh
# Creates the virtual environment
python3.11 -m venv train_env 
#activates the virtual environment
source train_env/bin/activate 
```
Install the dependencies used by using 

```sh
pip install -r requirements.txt
```

If you need to update the requirements, i.e. you added a new library, add it to the requirements using 
```sh
pip freeze > requirements.txt
```

To deactivate the virtual environment, open a new terminal or run the command 
```sh
deactivate
```
# Fix-Lint Autopep8
We use `autopep8` to fix the format of the code. Run the following command before pushing to the branch.
```sh
sh fix-lint.sh
```

# Data
Please add the file `ar41_for_ulb.csv` to the data folder so that all scripts are consistent.

## Order of running the scripts
Please run in order 
```sh
python src/scripts/make_parquet_from_csv.py
python src/scripts/clean_location.py
python src/scripts/clean_outliers.py
python src/scripts/mark_stop_track.py 
python src/scripts/add_weather.py
python src/scripts/dump_load_pickle.py -kmeans_k 5 -kmeans_percentile 95 -dbscan_eps 0.5 -dbscan_min_samples 20 -if_contamination 0.05 -svm_nu 0.05
python src/scripts/make_small_subsample.py
```

For reference about the meteorological data please consult this [link](https://dev.meteostat.net/formats.html#meteorological-parameters)

# Cluster
Columns explanation:

1. k_cluster: The cluster the point belongs to(0,1,2,3...) calculated by kmeans

1. k_anomaly: Whether it's an anomaly, detected by threshold(-1,1)

1. db_cluster: The cluster the point belongs to(-1,0,1,2...) calculated by dbscan

1. if_cluster: The cluster the point belongs to(-1,1) calculated by isolation forest

1. svm_cluster: The cluster the point belongs to(-1,1) calculated by svm one class

# Dashboard

For the dashboard please set up docker first using 
```
docker compose up -d
```

Afterwards upload the data to TimescaleDB using 
```sh
python src/scripts/upload_data_to_timescaledb.py 
```
It should take around 5 minutes to upload. 

Now you can go to `http://localhost:3000/dashboards` and a selection of dashboards should appear.
The user is `gabriel` and the password is `gabriel`.

To stream data you can use 
```sh
python src/scripts/stream_data.py
```