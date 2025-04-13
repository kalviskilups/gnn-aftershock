# Aftershock Location Prediction

A machine learning system that predicts the location of aftershocks following a major seismic event using waveform data from seismic stations.

## Overview

This project analyzes seismic waveform data to predict the locations of aftershocks following a major earthquake. It extracts features from seismic waveforms, identifies mainshock and aftershock events, and trains a machine learning model to predict aftershock locations based on waveform characteristics and temporal patterns.

## Features

- Automated identification of mainshock and its aftershock sequence
- Extraction of key features from seismic waveforms
- Consolidation of multiple station recordings of the same event
- Time-aware feature engineering 
- Spatial prediction of aftershock locations using machine learning
- Comprehensive visualization and evaluation metrics

## Requirements

- Python 3.12+
- NumPy 1.26+
- Pandas 2.2+
- SciPy
- Scikit-learn 1.6+
- Matplotlib
- Seaborn
- SeisBench (for accessing the Iquique dataset)
- tqdm (for progress bars)

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm seisbench
```

## Dataset

This project uses the Iquique dataset, which contains seismic waveform data from the April 2014 Iquique earthquake in Chile and its aftershocks. The dataset is accessed through SeisBench, a machine learning benchmark platform for seismology.

**Note:** The Iquique dataset is approximately 5GB. Ensure you have sufficient storage and memory before downloading and using it.

## Usage

```bash
# Run the main script
python aftershock_prediction.py
```

By default, the script:
1. Downloads and loads the Iquique dataset
2. Processes up to 13,400 waveforms
3. Identifies the mainshock and aftershocks
4. Extracts features from the waveforms
5. Trains a RandomForest model to predict aftershock locations
6. Generates visualizations in the current directory

## Code Structure

- **Data Loading**: `load_aftershock_data_with_waveforms()` loads and preprocesses the seismic data
- **Event Identification**: `identify_mainshock_and_aftershocks()` identifies the main earthquake and its aftershocks
- **Feature Extraction**: `WaveformFeatureExtractor` class extracts features from raw waveform data
- **Data Preparation**: `prepare_ml_dataset()` and `safe_engineer_features()` prepare data for machine learning
- **Model Training**: `train_location_prediction_model_holdout()` trains and evaluates the prediction model
- **Visualization**: `visualize_results()` creates plots to evaluate model performance

## Output

The script generates several visualization files:
- `feature_importance_location.png`: Bar chart of feature importance
- `location_error_distribution.png`: Histogram of prediction errors
- `spatial_prediction.png`: Map showing actual vs. predicted locations
- `model_performance.png`: Summary of model performance metrics
- `error_vs_time.png`: Scatter plot of prediction errors vs. time since mainshock

Additionally, detailed logs are saved to the `logs/` directory.

## Performance

The model's performance is evaluated using:
- Mean Absolute Error (MAE)
- R² score
- Root Mean Square Error (RMSE)
- Median error in kilometers
- Mean error in kilometers

Typical performance on the Iquique dataset shows median location errors of ~50-60km.

## Limitations

- The model requires P and S wave arrivals for optimal feature extraction
- Prediction accuracy varies with distance from the mainshock
- Feature extraction is currently focused on a limited set of waveform characteristics

## Future Work

- Integration of additional waveform features
- Testing on other earthquake sequences
- Implementation of deep learning approaches
- Real-time prediction capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The SeisBench team for providing access to the Iquique dataset
- The scientific community for research on aftershock patterns