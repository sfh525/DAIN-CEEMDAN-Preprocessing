# CEEMDAN–WT–DAIN for Daily Stock Closing Price Prediction

A novel preprocessing pipeline for forecasting daily closing stock prices. The method combines **CEEMDAN** wave decomposition, **wavelet thresholding (WT)** denoising, and **Deep Adaptive Input Normalization (DAIN)**, then feeds the cleaned series into LSTM/GRU models.

This repository builds on the comparative LSTM/GRU optimizer study of [Makinde (2024)](http://arxiv.org/abs/2410.01843) as the **baseline**. That setup is improved and reused here as the **control**, alongside an **ablation study** that isolates the contribution of each preprocessing stage and the order of normalization.

---

## Motivation

Stock closing prices are non-stationary and noisy. Standard MinMax scaling alone often fails to handle regime shifts and high-frequency noise. This work proposes a staged pipeline:

1. **Decompose** the series into intrinsic mode functions (IMFs) with CEEMDAN  
2. **Denoise** each IMF with wavelet soft thresholding  
3. **Reconstruct** the signal and **adaptively normalize** with DAIN  

Forecasting is then performed with LSTM and GRU networks under Adam and Nesterov Accelerated Gradient (NAG).

---

## Method Overview

```
Raw Close  →  CEEMDAN  →  Wavelet Thresholding  →  Reconstruct  →  DAIN  →  LSTM / GRU
```

### CEEMDAN

Complete Ensemble Empirical Mode Decomposition with Adaptive Noise decomposes the closing-price series into IMFs plus a residual, separating multi-scale oscillatory components while reducing mode mixing.

- Implementation: `PyEMD.CEEMDAN`
- Settings: `trials=100`, `parallel=True`

### Wavelet Thresholding (WT)

Each IMF is denoised independently with a Daubechies wavelet and universal soft thresholding:

- Wavelet: `db4`
- Threshold: \(\sigma \sqrt{2 \log N}\), with \(\sigma\) estimated from the finest detail coefficients
- Mode: soft thresholding via `pywt`

Denoised IMFs are summed to reconstruct the cleaned series (per train/val/test split).

### DAIN (Deep Adaptive Input Normalization)

DAIN adaptively centers, scales, and gates the input using learned linear transforms over batch statistics (mean adaptation → adaptive standardization → sigmoid gating). Unlike fixed MinMax scaling, DAIN adjusts to distributional shifts between training and evaluation.

- Mode: `full`
- Separate forward/target layers for train vs. val/test (target path returns adaptive statistics for inverse transform)

### Forecasting Models

| Component | Setting |
|-----------|---------|
| Architectures | LSTM, GRU |
| Layers | 64 → Dropout(0.2) → 32 → Dropout(0.2) → Dense(16, ReLU) → Dense(1) |
| Optimizers | Adam (`lr=0.001`), NAG via SGD (`lr=0.01`, `momentum=0.9`, `nesterov=True`) |
| Loss | MSE |
| Epochs / batch | 50 / 32 |
| Lookback search | `{30, 60, 90, 120}` days (tuned on validation RMSE per model–optimizer pair) |

---

## Experimental Setup

| Item | Value |
|------|-------|
| Ticker | AAPL |
| Period | 2014-01-01 → 2024-08-01 |
| Target | Daily closing price |
| Split | 70% train / 15% val / 15% test (chronological) |
| Metrics | RMSE, MAE, R² (reported in original price scale) |

Data are downloaded with `yfinance`. Sequences for validation and test retain lookback context from earlier splits.

---

## Study Design

### 1. Baseline Comparative Study → Control

The control follows the comparative study of **Adam vs. NAG** on **LSTM vs. GRU** using stock market data by Makinde (2024), which we adopt as the experimental baseline and improve for this work (MinMax scaling only; no CEEMDAN/WT/DAIN). The same model, optimizer, and window-tuning protocol are reused across all ablation variants so differences are attributable to preprocessing.

> Makinde, A. (2024). *Optimizing Time Series Forecasting: A Comparative Study of Adam and Nesterov Accelerated Gradient on LSTM and GRU networks Using Stock Market data.* [arXiv:2410.01843](http://arxiv.org/abs/2410.01843)

**Notebook:** [`Control.ipynb`](Control.ipynb)

### 2. Ablation Study

The ablation varies *whether* and *when* decomposition/denoising and normalization are applied:

| # | Variant | Pipeline | Notebook |
|---|---------|----------|----------|
| 1 | **Control** | MinMax only | [`Control.ipynb`](Control.ipynb) |
| 2 | **CEEMDAN-WT-MinMax** | CEEMDAN → WT → MinMax *(default fixed normalization)* | [`CEEMDAN-WT-MinMax.ipynb`](CEEMDAN-WT-MinMax.ipynb) |
| 3 | **MinMax-CEEMDAN-WT** | MinMax → CEEMDAN → WT | [`MinMax-CEEMDAN-WT.ipynb`](MinMax-CEEMDAN-WT.ipynb) |
| 4 | **DAIN-CEEMDAN-WT** | DAIN → CEEMDAN → WT | [`DAIN-CEEMDAN-WT.ipynb`](DAIN-CEEMDAN-WT.ipynb) |
| 5 | **CEEMDAN-WT-DAIN** *(proposed)* | CEEMDAN → WT → DAIN | [`CEEMDAN-WT-DAIN.ipynb`](CEEMDAN-WT-DAIN.ipynb) |

A related DAIN-only run (no CEEMDAN/WT) is in [`DAIN.ipynb`](DAIN.ipynb).

**What the ablation tests**

- Value of CEEMDAN–WT vs. raw MinMax (Control vs. CEEMDAN-WT-MinMax)  
- Order of fixed scaling relative to decomposition (CEEMDAN-WT-MinMax vs. MinMax-CEEMDAN-WT)  
- Adaptive vs. fixed normalization, and order relative to CEEMDAN–WT (DAIN-CEEMDAN-WT vs. CEEMDAN-WT-DAIN)

---

## Results Summary

Test-set metrics from the executed notebooks (AAPL, original price scale):

### Control (MinMax only)

| Model & Optimizer | RMSE | MAE | R² |
|-------------------|------|-----|-----|
| LSTM Adam | 23.4734 | 22.8194 | −1.5720 |
| LSTM NAG | 24.0611 | 22.0872 | −1.5750 |
| GRU Adam | 18.0387 | 17.5436 | −0.3901 |
| GRU NAG | **13.9353** | **12.8611** | **0.3419** |

### CEEMDAN-WT-MinMax

| Model & Optimizer | RMSE | MAE | R² |
|-------------------|------|-----|-----|
| LSTM Adam | 40.1362 | 39.4245 | −4.5133 |
| LSTM NAG | 22.6643 | 20.9190 | −0.7580 |
| GRU Adam | 11.3346 | 10.7384 | 0.3906 |
| GRU NAG | **9.9029** | **8.9952** | **0.5749** |

### MinMax-CEEMDAN-WT

| Model & Optimizer | RMSE | MAE | R² |
|-------------------|------|-----|-----|
| LSTM Adam | 33.6341 | 32.9721 | −4.1015 |
| LSTM NAG | 39.3774 | 37.6874 | −5.7160 |
| GRU Adam | 32.0151 | 31.5638 | −2.5061 |
| GRU NAG | **15.9109** | **14.9277** | **−0.1416** |

### DAIN-CEEMDAN-WT

| Model & Optimizer | RMSE | MAE | R² |
|-------------------|------|-----|-----|
| LSTM Adam | 3.2052 | 2.4646 | 0.9649 |
| LSTM NAG | 4.3383 | 3.2920 | 0.9356 |
| GRU Adam | **2.9928** | **2.3180** | **0.9694** |
| GRU NAG | 3.3188 | 2.5304 | 0.9623 |

### CEEMDAN-WT-DAIN (proposed)

| Model & Optimizer | RMSE | MAE | R² |
|-------------------|------|-----|-----|
| LSTM Adam | 3.3119 | 2.5699 | 0.9480 |
| LSTM NAG | 4.5479 | 3.5167 | 0.9292 |
| GRU Adam | **2.5237** | **1.9415** | **0.9782** |
| GRU NAG | 3.2815 | 2.5138 | 0.9533 |

### Best configuration per variant

| Variant | Best model | RMSE ↓ | R² ↑ |
|---------|------------|--------|------|
| Control | GRU NAG | 13.9353 | 0.3419 |
| CEEMDAN-WT-MinMax | GRU NAG | 9.9029 | 0.5749 |
| MinMax-CEEMDAN-WT | GRU NAG | 15.9109 | −0.1416 |
| DAIN-CEEMDAN-WT | GRU Adam | 2.9928 | 0.9694 |
| **CEEMDAN-WT-DAIN** | **GRU Adam** | **2.5237** | **0.9782** |

**Takeaways**

- **CEEMDAN → WT → DAIN** is the strongest pipeline overall (GRU + Adam).  
- Applying **MinMax before** CEEMDAN–WT degrades performance relative to CEEMDAN–WT then MinMax.  
- **DAIN** (adaptive normalization) yields a large gain over fixed MinMax, whether placed before or after CEEMDAN–WT; placing DAIN **after** denoising works best in these runs.  
- GRU consistently outperforms LSTM under the same preprocessing and optimizer settings.

---

## Repository Layout

```
.
├── Control.ipynb              # Comparative study / control (MinMax only)
├── CEEMDAN-WT-MinMax.ipynb    # Ablation: CEEMDAN → WT → MinMax
├── MinMax-CEEMDAN-WT.ipynb    # Ablation: MinMax → CEEMDAN → WT
├── DAIN-CEEMDAN-WT.ipynb      # Ablation: DAIN → CEEMDAN → WT
├── CEEMDAN-WT-DAIN.ipynb      # Proposed: CEEMDAN → WT → DAIN
├── DAIN.ipynb                 # Related: DAIN only
└── README.md
```

Each notebook follows the same workflow: data prep → preprocessing → sliding-window tuning → train LSTM/GRU (Adam & NAG) → evaluate → visualize (loss curves, metric tables, heatmaps).

---

## Requirements

Typical stack used in the notebooks:

- Python 3.x  
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `tabulate`  
- `yfinance`  
- `scikit-learn`  
- `tensorflow` / `keras`  
- `torch` (DAIN layer)  
- `PyEMD` (CEEMDAN)  
- `PyWavelets` (`pywt`)

Install example:

```bash
pip install numpy pandas matplotlib seaborn tabulate yfinance scikit-learn tensorflow torch EMD-signal PyWavelets
```

Then open and run the notebooks in order of the ablation table (or start with `CEEMDAN-WT-DAIN.ipynb` for the proposed method).

---

## References

Makinde, A. (2024). Optimizing Time Series Forecasting: A Comparative Study of Adam and Nesterov Accelerated Gradient on LSTM and GRU networks Using Stock Market data. http://arxiv.org/abs/2410.01843

This codebase implements the preprocessing and forecasting experiments for predicting daily closing prices with CEEMDAN wave decomposition, wavelet thresholding, and DAIN adaptive normalization, evaluated against an improved form of the Makinde (2024) comparative LSTM/GRU baseline (used as control) and the ablation variants listed above.
