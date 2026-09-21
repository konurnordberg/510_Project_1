# Signals or Noise? NVIDIA Through Stocks and Stars

**Konur and Hadley | AIPI 510 | Duke University**

## Overview

We explore two questions:
- Do TSMC stock declines precede declines in NVIDIA’s stock?
- Do GeForce NOW customer reviews reveal sentiment changes around product announcements?

Our analysis includes data cleaning, feature engineering, visualizations, and before/after event comparisons.

## Data

Source: [samartalwar’s NVIDIA dataset on Kaggle, Version 4](https://www.kaggle.com/datasets/samartalwar/nvidia-360-stock-cloud-ux-and-semiconductor-macro).

The raw files are included in `data/`:
- `nvidia_daily_master_360.csv`: 4,280 days of market data.
- `geforcenow_app_reviews_raw.csv`: 12,008 reviews with ratings, text, timestamps, and supplied sentiment labels.

Announcement dates and source links are documented in `data/processed/event_study/event_catalogue.csv`.

## Reproduce the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/konurnordberg/510_Project_1.git
cd 510_Project_1
```

### 2. Install dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn kagglehub jupyterlab ipykernel
```

### 3. Launch Jupyter from the notebooks folder

```bash
cd notebooks
jupyter lab
```

### 4. Run the notebooks in order

1. `data_preprocessing.ipynb`
2. `q1_supply_chain_beta.ipynb`
3. `frustration_analysis.ipynb`

Run each notebook from top to bottom. **Skip the Kaggle download cell in preprocessing:** use the included CSVs to reproduce our original results.

Keep the working directory as `notebooks/` so the relative file paths work.

Results appear in the notebooks. The combined dataset and saved event-study outputs are stored under `data/processed/`.

## Limitations

Reviews do not represent all customers, supplied sentiment labels can be misleading, and before/after differences do not establish causation. This project is exploratory and does not establish a reliable trading strategy.
