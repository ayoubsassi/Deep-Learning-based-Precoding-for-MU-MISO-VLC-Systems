# VLC Beamforming with Masked Neural Networks

A VLC (Visible Light Communication) beamforming workflow built around three stages:

1. Generate per-user-count datasets with MATLAB CVX labels.
2. Build one balanced mixed dataset with zero-padded samples and masks.
3. Train a masked neural network and evaluate it against ZF and CVX-based OLP.

## Project Structure

```
├── README.md
├── requirements.txt
├── src/                    # reusable modules
│   ├── R_func.py           # Lambertian radiant intensity function R(t, m)
│   └── model_eval.py       # metrics helpers (MSE, MAE, RMSE, R²) and loss plots
├── scripts/                # pipeline steps, in order
│   ├── NEW_data_extraction_with_power.py   # 1. generate per-K datasets
│   ├── build_mixed_dataset.py              # 2. build mixed padded dataset
│   ├── train_mixed_masked.py               # 3. train the masked model
│   └── evaluate_dataset_rates_cvx.py       # 4. evaluate ZF vs OLP vs ML
├── matlab/
│   └── cvx_func_bisection_2.m              # MATLAB CVX solver
├── data/                   # datasets (.pkl) — gitignored, generated locally
├── models/                 # trained model + scaler — gitignored
└── results/                # plots and summaries — gitignored
```

> **Note:** `data/`, `models/`, and `results/` contents are **not included in this repository** (excluded via `.gitignore`). They are produced by running the pipeline.

> **Important:** the scripts still reference files by bare filenames (e.g. `3_users_data.pkl`), so they expect their inputs/outputs in the working directory. Until the paths are updated in the scripts, run each script from the directory containing the files it needs (or temporarily copy artifacts next to the script).

## Pipeline Scripts

- `scripts/NEW_data_extraction_with_power.py`
  Generates one dataset for a fixed number of users `K` by:
  - creating random user-position matrices,
  - computing the VLC channel matrix `H`,
  - solving for beamforming weights `W` with MATLAB CVX,
  - appending rows to a pickle DataFrame with columns `H`, `power`, `label`.

  Current hardcoded settings:
  - `K = 3`, output file `3_users_data.pkl`
  - transmitters `M = 6`
  - 16 power points from 15 dBm to 30 dBm

  To generate other datasets, edit `K`, `num_rows_u`, `file_path`, and the output filename, then rerun. If the target pickle already exists and is non-empty, the script appends more rows.

- `scripts/build_mixed_dataset.py`
  Samples 16,000 rows from each of the six single-user-count datasets (K = 1..6) and combines them into one padded dataset, `mixed_dataset_balanced_16k(123456).pkl` (96,000 rows), with columns:
  - `H`: `6 x 6` (zero-padded channel matrix)
  - `label`: `6 x 6` (zero-padded beamforming matrix)
  - `mask`: length-6 vector marking active users
  - `power`, `K`

- `scripts/train_mixed_masked.py`
  Trains a TensorFlow masked CNN on the mixed dataset. It creates train/validation/test splits, saves the test split, fits a `StandardScaler`, trains the model, and saves weights, full model, and training history.

  Important details:
  - input features: 37 values = flattened `H` (36) + `power` (1)
  - output: 36 values = flattened `6 x 6` beamforming matrix
  - mask is applied both in the model output and in the custom loss
  - existing model artifacts are deleted before saving new ones
  - settings: 10% test split, 20% validation split of the remaining pool, 1000 epochs, batch size 32, learning rate `1e-3`

- `scripts/evaluate_dataset_rates_cvx.py`
  Evaluates three methods on the saved test split:
  - `ZF` (zero-forcing)
  - `OLP` solved through MATLAB CVX
  - `ML` from the trained TensorFlow model

  Outputs: a text summary (`test_rates_summary_cvx.txt` by default), `aggregate_avg_rate.png`, and `aggregate_maxmin_rate.png`.

  Optional example:

  ```powershell
  python evaluate_dataset_rates_cvx.py --num-samples 50 --target-k 3 --output results\k3_eval.txt
  ```

## Generated Files

Per-user-count datasets (pandas DataFrames pickled, schema `H` / `power` / `label`), stored in `data/`:

| File | Rows | `H` and `label` shape |
|---|---|---|
| `1_users_data.pkl` | 16,000 | `6 x 1` |
| `2_users_data_19.2k.pkl` | 19,200 | `6 x 2` |
| `3_users_data.pkl` | 16,000 | `6 x 3` |
| `old_data_4users_40k.pkl` | 40,000 | `6 x 4` |
| `5_users_data.pkl` | 16,000 | `6 x 5` |
| `6_users_data.pkl` | 38,400 | `6 x 6` |

Training artifacts:

- `data/mixed_dataset_balanced_16k(123456).pkl` — mixed balanced training dataset, 96,000 rows
- `data/mixed_dataset_test_split.pkl` — saved test split (adds derived columns `H_vector`, `W_vector`, `mask_vector`, `input_vec`)
- `models/SS_scaler_mixed_masked_16k(123456).pkl` — `joblib`-saved `StandardScaler` fitted on 37 input features
- `models/mixed_masked_model_16k(123456).weights.h5` — best model weights
- `models/mixed_masked_model_16k(123456).h5` — full saved Keras model
- `mixed_masked_history.csv` — training history

## Setup

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

### MATLAB (required for generation and evaluation)

- MATLAB with the [CVX](http://cvxr.com/cvx/) toolbox installed
- MATLAB Engine for Python

Both Python scripts that call MATLAB contain this hardcoded line — **change the path to your own CVX installation**:

```python
eng.addpath(eng.genpath(r"C:\Users\ayoubsa\Desktop\cvx"), nargout=0)
```

## Workflow

Run the pipeline steps in order:

1. `python NEW_data_extraction_with_power.py` — once per user count `K` (edit settings between runs)
2. `python build_mixed_dataset.py`
3. `python train_mixed_masked.py`
4. `python evaluate_dataset_rates_cvx.py`
