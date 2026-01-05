# Symptom Trajectories

End-to-end pipeline for building model-ready train/validation/test splits of time series data from Synthea synthetic data, and modeling mental health symptom trajectories.

## Features
- Uses the [Synthetic Health Synthetic Data Generator](https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running) to generate a homogenous, normalized dataset.
- Ingest Synthea CSV/CSV.GZ exports and builds longitudinal feature tables.
- Derive modeling-ready dataframes and train/val/test splits with configurable thresholds.
- Predict warning signals for treatment failure in depressed patients over the next 30 days by training a XGBoost, LSTM, or custom LSTM-Transformer fusion model (dubbed a Long Short Term Transformer, LSTT) on the time series data.

## Dependencies
- conda: `conda env create -f environment.yml`
- Pip alternative: run `pip install -e .` inside a venv

## Running the pipeline
1) Generate Synthea data into `data/raw-synthea` (see below)
2) Build features into `data/build` (see `config/default.yaml`)
  - `python -m symptom_trajectories.cli build-features`
3) Prepare splits/training tables
  - `python -m symptom_trajectories.cli prepare-splits`
  - train/val/test splits output into `data/modeling`
4) Train model
  - XG Boost: `python -m symptom_trajectories.run train-xgb [--config path]`
  - LSTM: `python -m symptom_trajectories.run train-lstm [--config path]`
  - fusion LSTM-transformer: `python -m symptom_trajectories.run train-lstt [--config path]`

#### Configuration
All commands pull from accompanying sections in `config/default.yaml` is loaded automatically. `config/colab.yaml` still being tested. To run with a different file (e.g., Colab paths, model derivations): `python -m symptom_trajectories.cli build-features --config config/colab.yaml`

## Generating Synthea data
Reference: 
1) Follow [instructions](https://github.com/synthetichealth/synthea/wiki/Developer-Setup-and-Running) for installing Java 11-17
2) `cd data/synthea/synthea`. Export properties can be edited in `src/main/resources/synthea.properties`
3) Run (powershell):
   - `.\gradlew run --args="-p 10000 -d ..\modules --exporter.csv.export=true"`
   - Adjust `-p` for population size based on your system
   - existing files are not overridden by default; make sure to delete files from previous runs in `data/synthea` before running again.

## List of commands
- `python -m symptom_trajectories.run build-features [--config path] [overrides...]`
  - Parses raw Synthea CSV/CSV.GZ into feature tables in `data/data-build` by default.
- `python -m symptom_trajectories.run prepare-splits [--config path] [overrides...]`
  - Builds model-ready tables and train/val/test splits in `data/data-train` by default.
- `python -m symptom_trajectories.run train-xgb [--config path]`
  - Train XGBoost classifier on the data
- `python -m symptom_trajectories.run train-lstm [--config path]`
  - Train XGBoost classifier on the data
- `python -m symptom_trajectories.run train-lstt [--config path]`
  - Train XGBoost classifier on the data