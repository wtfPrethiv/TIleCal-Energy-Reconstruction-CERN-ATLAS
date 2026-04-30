# TileCal Signal Energy Reconstruction - Ridge Regression Baseline

A linear ridge regression model for reconstructing Low-Gain (LG) channel signal energy from 7-sample ADC pulse windows recorded by the ATLAS Tile Calorimeter (TileCal), developed as part of **Google Summer of Code 2026** with **CERN-HSF**.

**Mentors:** Luca Fiorini, Fernando Carrió

---

## Project Overview

The ATLAS Tile Calorimeter digitizes detector signals at 40 MHz (one sample every 25 ns). For each bunch crossing, a window of **7 consecutive ADC samples** is recorded per channel. The goal of this project is to reconstruct the true deposited energy from these raw samples using a **fully linear algorithm** - a hard requirement for deployment in FPGA-based Level-0 hardware trigger systems, which demand nanosecond-scale latency.

TileCal uses a dual-gain readout:

- **High Gain (HG)** - sensitive to small energy deposits, saturates for large signals
- **Low Gain (LG)** - handles large energy deposits without saturation

This project targets **Low-Gain energy reconstruction** using the 7 normalized ADC samples as input features.

---

## How the Linear Algorithm Was Developed

### Why Ridge Regression?

The simplest linear model - ordinary least squares - learns weights that minimize MSE on the training set. However, in a pile-up environment (signals from neighbouring bunch crossings contaminating the in-time pulse), some samples in the pulse wings carry noisy, pile-up-dominated information. An unregularized model can overfit to these noisy samples by assigning them disproportionately large weights.

Ridge regression adds an L2 penalty on the weights:

```
L = (1/M) * sum(y - y_pred)^2 + lambda * ||w||^2
```

This shrinks weights toward zero, discouraging the model from relying heavily on any single noisy pulse sample. The result is a more stable set of filter coefficients that generalizes better across varying pile-up conditions.

The 7 learned weights are directly analogous to **Optimal Filter coefficients** - a fixed weighted sum implementable on an FPGA as a single multiply-accumulate (MAC) operation with no non-linear units.

### Implementation

The model is a single `nn.Linear(7, 1)` layer in PyTorch:

```python
y_pred = w[0]*x[0] + w[1]*x[1] + ... + w[6]*x[6] + b
```

L2 regularization is applied via AdamW's `weight_decay` parameter. This is equivalent to the closed-form ridge solution:

```
w_hat = (X^T X + lambda * I)^-1 X^T y
```

Training uses `X[:, 1, :]` (Low-Gain samples) as input and `y[:, 1]` as the reconstruction target.

---

## Repository Structure

```
CERN-ATLAS/
│
├── notebooks/
│   └── signal_reconstruction.ipynb  # full pipeline: load → train → evaluate → plot
│
├── data/
│   ├── train/                        # training shard files (.pt)
│   ├── val/                          # validation shard files (.pt)
│   ├── test/                         # test shard files (.pt)
│   └── y_stats.npz                   # normalization statistics (mean, std per channel)
│
├── report/
│   └── cern__ATLAS_Report.pdf        # project report
│
├── results/
│   ├── models/
│   │   └── final_ridge_regression.pth  # saved model weights
│   └── plots/                          # generated evaluation plots
│
├── src/
│   ├── evaluate.py                   # metrics and denormalization
│   ├── load_data.py                  # TileCalDataset and DataLoader setup
│   ├── models.py                     # LinearRegression model definition
│   ├── train.py                      # training loop with validation
│   └── utils.py                      # plotting utilities
│
├── main.py                           # end-to-end evaluation pipeline
├── README.md
└── requirements.txt
```

---

## Dataset Format

Each shard file (`.pt`) contains:

```python
{
  "X":    # shape [2048, 2, 7]  - normalized ADC samples (both gain channels)
  "y":    # shape [2048, 2]     - true signal energy targets
  "y_OF": # shape [2048, 2]     - Optimal Filter reference reconstruction
}
```

Channel indexing:

```python
X[:, 0, :]  # High-Gain samples
X[:, 1, :]  # Low-Gain samples  <- used as model input

y[:, 0]     # HG true energy
y[:, 1]     # LG true energy    <- used as regression target
```

Each shard contains **2048 events**. Dataset splits:

| Split      | Shards | Approximate Events |
|------------|--------|--------------------|
| Training   | 261    | ~534,528           |
| Validation | 56     | ~114,688           |
| Testing    | 56     | ~114,688           |

The 7 input samples span a timing window centred on the in-time bunch crossing n:

| Sample | Time Offset | Notes                       |
|--------|-------------|-----------------------------|
| 1      | -75 ns      |                             |
| 2      | -50 ns      | Out-of-time pile-up zone    |
| 3      | -25 ns      |                             |
| **4**  | **0 ns**    | **In-time peak (dominant)** |
| 5      | +25 ns      |                             |
| 6      | +50 ns      | Pulse tail / pile-up zone   |
| 7      | +75 ns      |                             |

---

## Installation

```bash
pip install torch numpy matplotlib scikit-learn
```

---

## Training

All training is performed in:

```
notebooks/signal_reconstruction.ipynb
```

### Hyperparameters

| Parameter        | Value |
|------------------|-------|
| Optimizer        | AdamW |
| Learning Rate    | 9e-3  |
| Regularization λ | 1e-3  |
| Epochs           | 120   |
| Batch Size       | 1024  |
| Loss Function    | MSE   |

Save trained weights after training:

```python
torch.save(model.state_dict(), "results/models/final_ridge_regression.pth")
```

---

## Running Evaluation

```bash
python main.py
```

This will:
1. Load the test dataset from `data/test/`
2. Load trained model weights from `results/models/`
3. Run inference on the test set
4. Denormalize predictions using `y_stats.npz`
5. Compute all metrics and print results
6. Save evaluation plots to `results/plots/`

---

## Results

### Metrics on Test Set

| Metric              | Value   |
|---------------------|---------|
| MSE                 | 0.0111  |
| MAE                 | 0.0302  |
| R² Score            | 0.9906  |
| Mean Relative Error | 0.4122  |
| RMS Relative Error  | 10.0534 |

MSE, MAE, and R² are computed on **normalized** predictions. Mean and RMS relative error are computed in **physical energy units** after denormalization.

### Understanding the Results

The R² of **0.9906** confirms the model reconstructs energy well overall - it explains over 99% of the variance in the Low-Gain target energy across the full test set.

The elevated RMS relative error (10.05) requires careful interpretation. Relative error is computed as `(E_pred - E_true) / E_true`, which is highly sensitive to low-energy events where the denominator is small. Even a small absolute prediction error produces a large relative deviation when the true energy is near zero.

In practice, a small fraction of low-energy events are heavily affected by pile-up - overlapping pulses from neighbouring bunch crossings obscure the true in-time signal, causing the model to predict near-zero energy. This produces large relative errors concentrated in the low-energy region, visible as a diagonal band in the 2D relative error plots. The gap between RMS (10.05) and mean (0.41) confirms this is driven by outliers, not systematic bias. This behaviour is consistent with MLP and CNN models trained on the same dataset.

---

## Generated Plots

All plots are saved to `results/plots/final_plots`:

| File                  | Description                                            |
|-----------------------|--------------------------------------------------------|
| `training.png`        | MSE loss vs epoch (train and validation)               |
| `acc.png`             | Predicted vs true energy scatter on validation set     |
| `Reative_err.png`     | Histogram of relative error distribution               |
| `scaled_rel2d.png`    | 2D relative error vs true energy (zoomed, E < 60 ADC) |
| `unscaled_rel_2d.png` | 2D relative error vs true energy (full energy range)   |
| `abs_error.png`       | Absolute error vs true energy (hexbin)                 |
| `det_pulse.png`       | Example detector pulse - HG and LG channels overlaid   |

---

## Technologies

- **PyTorch** - model implementation, training loop, dataset loading
- **NumPy** - numerical computation and denormalization
- **Matplotlib** - all visualizations
- **Scikit-learn** - MSE, MAE, R² metric computation

### GSoC 2026 Status (CERN-ATLAS)

- [x] Qualified GSoC 2026 Phase 1 (CERN-ATLAS)
- ~~Proposal under review (Google Summer of Code 2026)~~ [rejected]
