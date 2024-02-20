#!/bin/sh

source train_env/bin/activate

# echo TRANSFORMING DATA
# python src/scripts/make_parquet_from_csv.py
# echo CLEAN LOCATIO
# python src/scripts/clean_location.py
echo CLEAN OUTLIERS
python src/scripts/clean_outliers.py
echo MARK STOP AND TRACK
python src/scripts/mark_stop_track.py 
echo ADD WEATHER
python src/scripts/add_weather.py