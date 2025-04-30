# Machine Learning Methodology Project

This repository contains the project for a graduate course on Machine Learning Methodology. The goal of this project is to utilize machine learning models to make predictions based on GeoTIFF file formats.

## Project Workflow

1. **Raw Datasets**:  
    The raw datasets are located in the `dataset` folder.

2. **GeoTIFF Processing**:  
    - The GeoTIFF data file is processed to generate a `reprojected_resampled_raster.tif` file.
    - The `reprojected_resampled_raster.tif` file is resampled and converted into a CSV format for further analysis.

3. **Model Training and Testing**:  
    - Python scripts in the `ML_py` folder are used to train and test machine learning models on the processed data.

## Learning Objectives

Through this project, we aim to:
- Understand how to process GeoTIFF file formats.
- Apply machine learning models to geospatial data.
- Predict outcomes based on geospatial features.

## Folder Structure

```
/dataset       - Contains raw GeoTIFF datasets
/ML_py         - Python scripts for training and testing ML models
raster.tif     - Processed GeoTIFF file
output.csv     - Resampled data in CSV format
```

## Requirements

- Python 3.x
- Required Python libraries (install via `requirements.txt` if provided)

## Usage

1. Process the GeoTIFF file to generate `raster.tif` and convert it to CSV format.
2. Run the Python scripts in the `ML_py` folder to train and test the machine learning models.

## Acknowledgments

This project is part of the graduate course on Machine Learning Methodology.
