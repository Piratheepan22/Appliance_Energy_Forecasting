# Appliance Energy Prediction with Deep Learning (TensorFlow)

**Author:** Piratheepan M. 

Multivariate time-series forecasting of household appliance energy use (Wh) from the Appliance Energy Prediction dataset (10-minute readings, January to May 2016). The main task predicts energy use **10 minutes ahead**. A second experiment predicts **1 hour ahead**.

## What the project does
1. Exploratory data analysis (trends, daily and weekly patterns, correlations, autocorrelation).
2. Preprocessing: missing-value check and imputation test, outlier analysis, capping of extreme sensor values, standardisation, time-ordered 80/20 split (plus a validation set taken from the end of the training period).
3. Feature engineering: calendar features, lags chosen from the partial autocorrelation, rolling statistics, interaction features, humidity-change features, a "typical usage at the target time" feature, and public holidays. An automatic test checks that no feature uses future data.
4. Feature selection: correlation filter, random-noise benchmark and Recursive Feature Elimination, judged on the validation set.
5. Baselines: persistence, moving average, Linear Regression, Random Forest and Gradient Boosting (also with a log-transformed target, so the comparison with the neural networks is fair).
6. Deep learning (TensorFlow/Keras): MLP, LSTM, GRU and CNN-LSTM compared; random search over window length, layers, units, dropout, L2, learning rate, batch size, optimiser, loss and target transformation; best candidates re-tested with new seeds; dropout sweep; early stopping.
7. Final model: five-seed ensembles of the best "raw target" CNN-LSTM network (main) and the best "log target" CNN-LSTM network (companion), blended with a weight chosen on the validation set (0.65 main, 0.35 companion). The blend gains only a small amount over the main network alone (about 0.6% to 1.2%), which is reported honestly in the report.
8. Evaluation with MAE, RMSE, MAPE and R-squared, plots of predictions and residuals, error analysis by hour and by consumption level, a before/after-tuning comparison, and a one-hour-ahead check.

## Repository structure
```
├── data/
│   ├── raw/                  <- put energy_data_set.csv here
│   └── processed/            <- feature table written by the notebook
├── notebooks/
│   └── Appliance_Energy_Forecasting.ipynb   <- the whole implementation
├── models/                   <- final_config.json, selected_features.json (+ trained models and scaler, see below)
├── reports/
│   ├── figures/              <- all plots (created by the notebook)
│   ├── tables/               <- all result tables as CSV (created by the notebook)
│   └── report.pdf            <- final report
├── requirements.txt
└── README.md
```
All code is in one notebook, organised in numbered sections with functions and docstrings (no separate `.py` files).

## How to run

### Option A: Google Colab (recommended)
1. Open `notebooks/Appliance_Energy_Forecasting.ipynb` in Google Colab (upload it, or open it from GitHub: *File > Open notebook > GitHub*).
2. Choose *Runtime > Change runtime type > T4 GPU*.
3. Run all cells (*Runtime > Run all*). When the notebook asks for the dataset, upload `energy_data_set.csv` (or place it in `data/raw/` first if you cloned the repository).
4. A full run trains about 60 small networks and takes roughly 60-90 minutes on a T4 GPU (estimate). For a quick check, set `QUICK_RUN = True` in section 0.2 (a few minutes, results are not meaningful).
5. At the end, `reports/results_bundle.zip` (all figures and tables) is created and downloaded.

### Option B: local machine
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook notebooks/Appliance_Energy_Forecasting.ipynb
```
Run the notebook from the repository root or from the `notebooks/` folder. A GPU is optional but makes training much faster.

## Dependencies
Python 3.9 or higher. The exact versions of the reported run are in `requirements.txt` (TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, Matplotlib 3.10.0, Seaborn 0.13.2, joblib 1.6.0).

## Models
The notebook saves the trained networks into `models/`: `trained_model.keras` (and `.h5`) is the first of the five runs of the main network, `ensemble/` holds all five runs of the main and of the companion network (the final model is their average, blended with the weight in `final_config.json`), and `feature_scaler.joblib` holds the scaler.

## Reproducibility
Random seeds are fixed (`SEED = 42`). Small differences between runs can still occur on GPUs. Library versions of the final run are written to `requirements.txt` and `reports/tables/run_summary.json`.

## Results
Test set: the last 20% of the time line, predicting the next 10 minutes.

| Model | MAE (Wh) | RMSE (Wh) | MAPE | R-squared |
|---|---|---|---|---|
| Persistence (same as 10 minutes ago) | 26.65 | 66.44 | 21.9% | 0.467 |
| Linear Regression | 27.28 | 59.73 | 25.2% | 0.569 |
| Random Forest | 25.68 | 58.12 | 23.1% | 0.592 |
| Gradient Boosting (log target) | 23.64 | 58.99 | 19.1% | 0.580 |
| CNN-LSTM main network (raw target), average of 5 runs | 23.43 | 56.84 | 20.0% | 0.610 |
| **Final model: blend of main and companion CNN-LSTM networks** | **23.15** | **56.52** | **19.6%** | **0.615** |

The final model has lower MAE and RMSE than every baseline (2.1% and 2.8% lower than the best baseline on each metric). The blend gains only about 0.6% to 1.2% over the main network alone, which is small compared with the run-to-run variation of a single run. The full tables and plots are in the report and in `reports/tables` and `reports/figures`.

## Data source
Appliance Energy Prediction dataset (Candanedo, Feldheim and Deramaix, 2017).
