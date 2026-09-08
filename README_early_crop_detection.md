# Early Crop Detection Using Satellite Time-Series Data

> A GeoAI and Satellite Remote Sensing system leveraging multi-spectral time-series observations and multi-scale 1D Inception Deep Learning to classify early-stage agricultural crops.

---

## Overview

Accurate early detection of crop types is vital for precision agriculture, regional food security, yield forecasting, and disaster management. This project processes multi-band time-series satellite imagery from **Sentinel-2** to track crop growth trajectories and automatically classify crop types—specifically **Paddy**, **Sugarcane**, and **Other** crops.

Developed during academic coursework at **IIT Ropar**, the repository combines geospatial raster transformations, vegetation index calculation (NDVI), parallelized data extraction, a custom **1D Inception Convolutional Neural Network (CNN)**, and both modern web and desktop interfaces.

---

## Key Features

### 📡 Satellite Multi-Spectral Data Pipeline
- **Sentinel-2 Band Extraction**: Extracts Red (Band 6) and Near-Infrared (Band 8) reflections across temporal acquisition dates.
- **NDVI Trajectory Calculation**: Computes Normalized Difference Vegetation Index `(NIR - Red) / (NIR + Red)` time-series signatures representing seasonal crop growth cycles.
- **Geospatial Coordinate Alignment**: Maps global Latitude/Longitude coordinates (`EPSG:4326`) to raster pixel matrix indices using `rasterio` and `pyproj`.

### 🧠 Multi-Scale 1D Inception Deep Learning
- **Custom Keras Inception Layer**: Uses parallel 1D Conv kernels (1x1, 3x3, 5x5) and MaxPool branches to capture multi-scale temporal-spectral features.
- **Robust Feature Representation**: Handles short-term phenological variations alongside long-term seasonal trends.
- **Model Checkpointing & Evaluation**: Includes model evaluation metrics (Accuracy, Weighted Precision, Recall, and F1 Score) with saved pre-trained checkpoints.

### 💻 Dual Application Interfaces
- **Streamlit Web Application (`streamlit_app.py`)**: An interactive dashboard featuring pipeline explanations, live model evaluation, NDVI temporal trajectory visualizations, and GIS coordinate tools.
- **Desktop Application (`Crop_app.py`)**: A desktop GUI built with Tkinter for dataset selection, preprocessing, offline training, and evaluation.

---

## End-to-End System Pipeline

```text
               Sentinel-2 Satellite Rasters (.tif)
                                │
                                ▼
         Geospatial Coordinate Transformation (EPSG:4326)
                                │
                                ▼
         Multi-Band Pixel Extraction (Red: B6, NIR: B8)
                                │
                                ▼
            NDVI Phenological Trajectory Calculation
                                │
                                ▼
       1D Inception Neural Network (1x1, 3x3, 5x5 Kernels)
                                │
                                ▼
          Crop Classification: Paddy | Sugarcane | Other
```

---

## Architecture

The project is structured into a clean Python library package (`src/`) decouples model architecture, dataset preprocessing, and application logic:

```text
┌────────────────────────────────────────────────────────┐
│                   User Interfaces                      │
│        Streamlit Web App   │    Tkinter Desktop GUI    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Core Service Layer                    │
│      Model Inference      │   Geospatial Processing    │
└─────────────┬──────────────────────────┬───────────────┘
              │                          │
              ▼                          ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│     TensorFlow/Keras    │    │   Rasterio / PyProj     │
│   1D Inception Network  │    │ Multi-spectral Rasters  │
└─────────────────────────┘    └─────────────────────────┘
```

---

## Technology Stack

| Area | Technology |
|---|---|
| **Deep Learning** | Python, TensorFlow 2.x, Keras, Scikit-Learn |
| **Geospatial / GIS** | Rasterio, PyProj, GeoPandas |
| **Feature Engineering** | NDVI Time-Series Extraction |
| **Web Interface** | Streamlit, Matplotlib, Seaborn |
| **Desktop Interface** | Tkinter, Pillow |
| **Data Processing** | Pandas, NumPy, ProcessPoolExecutor |

---

## Project Structure

```text
.
├── models/                       # Saved pre-trained H5 model checkpoints
│   ├── best_model_epoch_66.h5
│   └── best_model_epoch_70.h5
├── src/                          # Core Python package
│   ├── __init__.py               # Package initialization
│   ├── model.py                  # Inception architecture, data loader & evaluation
│   ├── dataset_preprocessing.py # Raster extraction & NDVI calculations
│   └── utils.py                  # Parallel processing & dataset utilities
├── test/                         # Sample test dataset partitioned by crop class
│   ├── Paddy/
│   ├── Sugarcane/
│   └── other/
├── Images/                       # Application asset resources
├── streamlit_app.py              # Interactive Streamlit Web Interface
├── Crop_app.py                   # Tkinter Desktop GUI Application
├── dataset_build.py              # CSV Dataset generator utility
├── setup.py                      # Executable builder configuration
├── requirements.txt              # Unified Python dependency specification
└── README.md                     # Documentation
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.9 - 3.11
- Recommended: Virtual environment (`venv` or `conda`)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/PritamMahajan/Early-Crop-Detection-Using-Satellite-Timeseries-Data.git
cd Early-Crop-Detection-Using-Satellite-Timeseries-Data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application

```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501` to access the interactive web dashboard.

### 3. Launch Tkinter Desktop Application

```bash
python Crop_app.py
```

---

## Model Performance

Evaluated on test set observations using pre-trained checkpoint `best_model_epoch_70.h5`:

- **Classification Target**: Paddy vs. Sugarcane vs. Other Crops
- **Primary Metric**: Weighted F1-Score & Accuracy
- **Feature Representations**: 10-step temporal NDVI signatures

---

## Academic Context & Credits

This project was developed during master's coursework at **Indian Institute of Technology (IIT) Ropar**.

- **Author**: Pritam Sunil Mahajan (Master's in Artificial Intelligence — IIT Ropar)

---

## License

This project is open-source under the [MIT License](LICENSE).
