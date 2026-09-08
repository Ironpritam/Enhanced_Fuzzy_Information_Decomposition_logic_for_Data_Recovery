# Enhanced Fuzzy Information Decomposition (FID) for Data Recovery

> A hardware-accelerated, non-linear Fuzzy Data Recovery engine leveraging GPU/TPU vectorization, Pearson & Distance Correlation weighting, and Shallow Neural Network integration to reconstruct corrupted multi-attribute tabular & time-series data.

---

## Overview

In modern IoT sensor networks, intelligent transportation systems, and cybersecurity platforms, data streams frequently suffer from missing values, sensor failures, or malicious noise injection. Standard statistical imputation techniques (mean, median, or KNN imputation) often fail to preserve non-linear cross-attribute dependencies and dynamic feature covariance.

Developed as a **Master's Thesis Project (MTP)** at **Indian Institute of Technology (IIT) Ropar**, this repository introduces an **Enhanced Fuzzy Information Decomposition (FID)** framework. By decomposing multi-attribute datasets into fuzzy membership intervals and learning inter-attribute contribution weights via GPU-accelerated **Pearson Correlation**, **Distance Correlation**, and **Shallow Neural Network (MLP)** regressors, the system recovers corrupted features while preserving statistical distribution characteristics and downstream machine learning performance.

---

## Key Features

### ⚡ GPU & TPU Accelerated Matrix Engine
- **Vectorized TensorFlow Operations**: Native `@tf.function` compilation for ultra-fast parallel matrix operations across large-scale tabular datasets.
- **Hardware-Aware Auto-Detection**: Dynamic allocation and fallback strategy across TPU Cluster Resolvers, CUDA GPUs, and multi-core CPUs.
- **VRAM & Memory Leak Management**: Integrated memory cleanup routines (`clear_gpu_memory`) and deterministic model hashing to prevent VRAM memory buildup during extensive grid sweeps.

### 🧠 Multi-Variant Fuzzy Information Decomposition
- **Base FID Recovery**: Decomposes feature spaces into fuzzy upper/lower bounds and computes membership interval reconstructions.
- **FID + Pearson Correlation**: Dynamic weight allocation scaling linearly with cross-feature covariance vectors.
- **FID + Distance Correlation**: Vectorized batch distance matrix computation (`_gpu_distance_correlation_weights`) capturing non-linear feature relationships.
- **FID + Shallow Neural Network (FID + NN)**: Learns complex non-linear inter-attribute dependencies using lightweight TensorFlow/Keras MLP regressors.

### 📊 Comprehensive Evaluation Engine
- **Direct Reconstruction Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Pearson Correlation Coefficient ($r$).
- **Statistical Distribution Preservation**: Jensen-Shannon Divergence ($JS$) and Kolmogorov-Smirnov ($KS$) two-sample statistical testing.
- **Downstream Task Retainability**: Measures accuracy preservation across downstream Machine Learning classifiers (**Random Forest**, **K-Nearest Neighbors**, **Deep AutoEncoder**).

---

## End-to-End System Pipeline

```text
       Multi-Attribute Benchmark Dataset (ForestFires | PeMS-Bay | UNSW-NB15)
                                       │
                                       ▼
                   Synthetic Noise & Outlier Injection (σ, OP)
                                       │
                                       ▼
                     Enhanced Data Preprocessor & Scaling
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │               Hardware-Accelerated FID Recovery Engine                    │
 │                                                                           │
 │   ┌────────────────┐   ┌───────────────────────┐   ┌──────────────────┐   │
 │   │ Base Fuzzy ID  │   │  FID + Pearson/Dist   │   │     FID + NN     │   │
 │   │ Decomposition  │   │ Correlation Weighting │   │  MLP Regression  │   │
 │   └───────┬────────┘   └───────────┬───────────┘   └────────┬─────────┘   │
 └───────────┼────────────────────────┼────────────────────────┼─────────────┘
             │                        │                        │
             ▼                        ▼                        ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                       Recovery Evaluator Module                           │
 │  • MAE / RMSE / Pearson r    • Jensen-Shannon / KS-Test    • Model Acc    │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

The project is structured into a decoupled Python library package (`src/`) that isolates mathematical engines, preprocessing, evaluation metrics, and application interfaces:

```text
┌────────────────────────────────────────────────────────┐
│                   User Interfaces                      │
│        Streamlit Web App   │     CLI Benchmark Engine  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Core Service Layer                     │
│      FID Engine (TF/GPU)   │   Recovery Evaluator      │
└─────────────┬──────────────────────────┬───────────────┘
              │                          │
              ▼                          ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│     TensorFlow / Keras  │    │     Scikit-Learn /      │
│  @tf.function & TPU/GPU │    │      SciPy / Pandas     │
└─────────────────────────┘    └─────────────────────────┘
```

---

## Technology Stack

| Domain | Technology / Library |
|---|---|
| **Deep Learning & Computing** | Python, TensorFlow 2.x, Keras, Scikit-Learn |
| **Statistical Computing** | SciPy (Stats, Spatial Distance), NumPy, Pandas |
| **Machine Learning Benchmarks** | Random Forest, K-Nearest Neighbors, Deep AutoEncoder |
| **GPU / Acceleration** | CUDA, TensorFlow TPU Cluster Resolver, `@tf.function` |
| **Web Dashboard** | Streamlit, Matplotlib, Seaborn |
| **Caching & Memory** | Pickle, Hashlib, Garbage Collector (`gc`) |

---

## Project Structure

```text
.
├── src/                          # Core Python package
│   ├── core/
│   │   ├── fid_engine.py         # TF GPU/TPU vectorized FID recovery engine
│   │   └── memory.py             # VRAM allocation & model hash caching
│   ├── data/
│   │   └── preprocessor.py       # Data preprocessor & noise generator (sigma, OP)
│   ├── models/
│   │   └── trainers.py           # Downstream evaluators (RF, KNN, AutoEncoder)
│   └── evaluation/
│       └── metrics.py            # Reconstruction & statistical distribution metrics
├── Modular_Improved_FID_GPU_efficient_0.ipynb # Baseline research notebook (v0)
├── Modular_Improved_FID_GPU_efficient_1_0.ipynb # Refined research notebook (v1.0)
├── app.py                        # Streamlit Interactive Web Application
├── main.py                       # CLI Benchmark Executor
├── requirements.txt              # Unified Python dependency specification
└── README.md                     # Documentation
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.9 – 3.11
- Recommended: Virtual environment (`venv` or `conda`)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/PritamMahajan/Enhanced_Fuzzy_Information_Decomposition_logic_for_Data_Recovery.main.git
cd Enhanced_Fuzzy_Information_Decomposition_logic_for_Data_Recovery-main

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run CLI Benchmark

```bash
python main.py --csv forestfires --feature temp --sigma 0.3 --op 15 --method all
```

### 3. Launch Interactive Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the interactive web dashboard.

---

## Benchmark Datasets

| Dataset | Domain | Attributes | Primary Target Features |
|---|---|---|---|
| **ForestFires** | Environmental IoT | Meteorological Sensors | `temp`, `RH` |
| **PeMS-Bay** | Intelligent Transportation | Traffic Speed Sensors | Sensor `400001`, `400017` |
| **UNSW-NB15** | Cybersecurity | Network Intrusion Logs | `dur`, `rate` |

---

## Academic Context & Credits

This project was developed during Master's Thesis research at **Indian Institute of Technology (IIT) Ropar**.

- **Author**: Pritam Sunil Mahajan (Master's in Artificial Intelligence — IIT Ropar)

---

## License

This project is licensed under the [MIT License](LICENSE).
