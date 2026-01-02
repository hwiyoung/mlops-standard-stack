# 🚀 MLOps Standard Stack: satellite-change-detection & NVS

This repository provides a standardized MLOps stack designed for **Satellite Imagery Change Detection (CD)** and **3D Novel View Synthesis (NVS)** using 3D Gaussian Splatting. It integrates data management, model training, experiment tracking, and inference visualization into a unified web dashboard.

## 🏗️ System Architecture

The project consists of several containerized services managed by Docker Compose:

*   **📊 Streamlit Dashboard**: Central hub for managing the entire MLOps lifecycle.
*   **📡 MLflow**: Experiment tracking and model registry.
*   **📦 MinIO**: High-performance S3-compatible object storage for data and artifacts.
*   **🐘 PostgreSQL**: Database backend for MLflow metadata.

## 🌟 Key Features

### 1. Unified Dashboard
- **📂 Data Manager**: Seamlessly upload local datasets from `/workspace/data/` using a folder dropdown. Generate secure, 7-day temporary download links (Presigned URLs).
- **📍 Map Browser**: Interactive GIS interface powered by **PostGIS**.
    - **Photo Tracking**: Visualize drone/aerial photos as markers with GPS-based location.
    - **Ortho-Visualization**: View high-resolution GeoTIFF (Orthoimage) extents as polygons.
    - **Smart popups**: Preview thumbnails and click to view or download full-resolution images.
- **🔬 Training Lab**: Execute Change Detection or 3DGS training with real-time log monitoring and dynamic parameter/path overrides.
- **📦 Model Registry**: Review experiment metrics (IoU, PSNR) and directly access granular MLflow run details.
- **🔮 Inference**: Visualize CD overlays and render 360-degree NVS videos directly in the browser.

### 2. Remote Access Optimization
- Built-in intelligent IP detection and manual override for seamless access from any computer on the network.
- Dynamic URL generation for all sub-services (MLflow, MinIO) based on the detected Public IP.

## 🚀 Quick Start

### 📋 Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA Support (for training/inference)

### 🛠️ Installation & Execution
1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd mlops-standard-stack
    ```

2.  **Environment Settings**:
    Copy `.env.example` to `.env` and fill in your credentials.
    ```bash
    cp .env.example .env
    ```

3.  **Launch Services**:
    ```bash
    docker-compose up -d
    ```

4.  **Access the Dashboard**:
    Open [http://localhost:8501](http://localhost:8501) in your browser.

## 🌐 External Access Configuration

To access the dashboard from another machine, set the `PUBLIC_IP` in your `.env` file or via the dashboard sidebar:

```bash
# Example .env
PUBLIC_IP=192.168.10.203
```

## 📂 Project Structure

```text
├── dashboard/          # Streamlit dashboard application
├── src/
│   ├── data_loaders/   # TorchGeo & Custom datasets
│   ├── models/         # U-Net (CD) & Gaussian Splatting (NVS)
│   ├── training/       # Training scripts with MLflow integration
│   └── inference/      # Prediction and visualization logic
├── scripts/            # Utility scripts (upload, environment setup)
├── configs/            # YAML configuration files
└── docker-compose.yml  # Infrastructure definition
```

## 🗺️ Roadmap & WIP
Upcoming features and development milestones:

### 📍 GIS & Visualization (Core)
- [ ] **Real-time Indexing**: Implement `src/indexer/watch_bucket.py` for automated indexing via MinIO bucket event notifications.
- [ ] **Advanced GIS Tools**: Add layer transparency sliders and multi-temporal swipe comparison for orthoimages.
- [ ] **3D Visualization**: Integrate a 3D point cloud viewer to visualize 3DGS results directly on the map.

### 🧠 ML Engineering (Algorithm)
- [ ] **PyTorch Lightning Refactory**: Transition training scripts (`train_cd.py`, `train_nvs.py`) to Lightning for cleaner code and better scalability.
- [ ] **DVC Data Versioning**: Connect MinIO with DVC to manage dataset versions alongside code commits.

### 📊 Data Quality & Ops
- [ ] **Geo-Data Validation**: Use **Great Expectations** to ensure GeoTIFF/JPEG metadata (GPS, Resolution) meets quality standards.
- [ ] **Metadata Profiling**: Integrate **ydata-profiling** for automated statistical reports on the `image_metadata` database.

### ⚙️ System & Infrastructure
- [ ] **Scalability**: Implement background task queues (e.g., Celery/Redis) for batch processing massive datasets.
- [ ] **Security**: Add user authentication (RBAC) and multi-tenanted project isolation.

## 🛡️ License
This project is licensed under the Apache 2.0 License.