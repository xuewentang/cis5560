# CIS5560: Machine Learning Methodology Project

This repository contains the project for the graduate course **CIS5560** on Machine Learning Methodology. The goal of this project is to utilize machine learning models to make predictions based on GeoTIFF files.

## Project Workflow

1. **Data Collection**:
    - The raw data is stored in the `dataset` folder and is obtained from Google Earth Engine.
    - The dataset includes:
      - California Shapefile.
      - Multi-band GeoTIFF file.

2. **Data Preprocessing**:
    - Use the script `main.py` to resample the raster file in the `dataset` folder.
    - The output is a CSV file named `reprojected_resampled_raster_with_indices.csv`, where the pixel resolution is adjusted to 200x200.

3. **Model Training**:
    - Python scripts in the `ML_py` folder are used to build and train four machine learning models based on the processed CSV file.
    - The goal is to identify the best model for predicting wildfires in California.

## Folder Structure
- `dataset/`: Contains raw data files (California Shapefile and GeoTIFF files).
- `ML_py/`: Contains Python scripts for machine learning model development and training.
- `main.py`: Script for data preprocessing and raster resampling.

## Objective
To leverage machine learning techniques for wildfire prediction in California using geospatial data.
