# Deep Learning-based Precoding for MU-MISO VLC Systems

A masked neural network learns beamforming (precoding) matrices for multi-user MISO visible light communication systems, trained on optimal solutions from MATLAB CVX. The model handles a variable number of users (1–6) with a single network via zero-padding and masking, and is benchmarked against Zero-Forcing (ZF) and CVX-based optimal linear precoding (OLP).

## Structure

```
├── src/         # reusable modules (channel model, metrics)
├── scripts/     # pipeline steps (see Usage)
├── matlab/      # CVX solver called from Python
├── data/        # datasets (.pkl) — generated locally, not in repo
├── models/      # trained model + scaler — not in repo
└── results/     # evaluation outputs — not in repo
```

## Requirements

- Python dependencies: `pip install -r requirements.txt`
- MATLAB with [CVX](http://cvxr.com/cvx/) and MATLAB Engine for Python (needed for data generation and OLP evaluation)
- Update the hardcoded CVX path in the scripts that call MATLAB:

  ```python
  eng.addpath(eng.genpath(r"C:\path\to\your\cvx"), nargout=0)
  ```

## Usage

Run the pipeline in order:

```powershell
python NEW_data_extraction_with_power.py   # 1. generate dataset for one user count K (edit K, rerun per count)
python build_mixed_dataset.py              # 2. combine into one balanced, padded dataset
python train_mixed_masked.py               # 3. train the masked model
python evaluate_dataset_rates_cvx.py       # 4. evaluate ZF vs OLP vs ML, produce plots
```

The model takes the flattened channel matrix `H` plus transmit power as input (37 features) and outputs a `6 x 6` beamforming matrix, with a mask applied in both the output and the loss so inactive user columns are ignored.
