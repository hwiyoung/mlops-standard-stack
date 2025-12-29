# MLOps Standard Stack

**Standard Open Source Stack 기반의 MLOps 환경**

위성 이미지 변화탐지(Change Detection)와 Novel View Synthesis(3D Gaussian Splatting) 연구를 위한 MLOps 인프라입니다.

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      MLOps Stack                            │
├─────────────────────────────────────────────────────────────┤
│  🎛️ Dashboard (Streamlit)       → localhost:8501           │
│  📊 MLflow (Tracking Server)     → localhost:5000           │
│  📦 MinIO (S3 Compatible)        → localhost:9000/9001      │
│  🗄️ PostgreSQL (Metadata DB)    → localhost:5432           │
│  🎬 NVS Training (GPU)           → docker-compose run       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 프로젝트 구조

```
mlops-standard-stack/
├── docker-compose.yml     # 인프라 서비스 정의
├── .env                   # 환경변수 (Git 제외)
│
├── dashboard/             # 🆕 Streamlit 웹 대시보드
│   ├── app.py             # 메인 앱 (4탭 UI)
│   ├── Dockerfile
│   └── requirements.txt
│
├── src/
│   ├── models/
│   │   ├── unet.py           # Change Detection U-Net
│   │   └── gaussian_model.py # 🆕 3D Gaussian Splatting
│   │
│   ├── training/
│   │   ├── train_cd.py       # Change Detection 학습
│   │   └── train_nvs.py      # 🆕 NVS (3DGS) 학습
│   │
│   ├── inference/
│   │   ├── predict_cd.py     # CD 추론
│   │   └── render_nvs.py     # 🆕 NVS 렌더링 & 비디오
│   │
│   ├── data_loaders/
│   │   └── cd_dataset.py     # TorchGeo 데이터셋
│   │
│   └── utils/
│       ├── config.py         # YAML 설정 관리
│       └── visualization.py  # 시각화 유틸
│
├── configs/
│   ├── train_cd.yaml         # CD 학습 설정
│   └── train_nvs.yaml        # 🆕 NVS 학습 설정
│
├── scripts/
│   ├── upload_data.py        # 데이터 MinIO 업로드
│   ├── upload_nvs_data.py    # 🆕 COLMAP 데이터 업로드
│   └── init_minio.sh         # MinIO 버킷 초기화
│
├── docker/
│   ├── mlflow/Dockerfile     # MLflow 커스텀 이미지
│   └── nvs/                  # 🆕 NVS GPU 환경
│       ├── Dockerfile        # CUDA 12.1 + gsplat
│       └── requirements.txt
│
└── tests/
    └── test_tracking.py      # MLflow 연동 테스트
```

## 🚀 Quick Start

### 1. 인프라 시작
```bash
docker-compose up -d postgres minio mlflow
```

### 2. Python 환경 설정
```bash
mamba create -n mlops python=3.11 -y
mamba activate mlops
pip install -r requirements.txt
```

### 3. 대시보드 실행
```bash
# 로컬 실행
streamlit run dashboard/app.py --server.port 8501

# 또는 Docker
docker-compose up dashboard
```

### 4. 학습 실행

**Change Detection:**
```bash
python src/training/train_cd.py --config configs/train_cd.yaml
```

**Novel View Synthesis (3DGS):**
```bash
# 로컬 (gsplat 필요)
python src/training/train_nvs.py --config configs/train_nvs.yaml

# Docker GPU
docker-compose run nvs-train
```

### 5. 추론 실행

**CD 추론:**
```bash
python src/inference/predict_cd.py --run-id <mlflow_run_id> --pre pre.tif --post post.tif
```

**NVS 렌더링:**
```bash
python src/inference/render_nvs.py --run-id <mlflow_run_id> --auto-orbit --num-frames 120
```

## 🎛️ 웹 대시보드

**http://localhost:8501**

| 탭 | 기능 |
|----|------|
| 📂 Data Manager | MinIO 데이터 업로드/조회 |
| 🔬 Training Lab | CD/NVS 학습 실행, 실시간 로그 |
| 📦 Model Registry | MLflow 실험 조회, 성능 요약 |
| 🔮 Inference | 모델 추론, 결과 시각화 |

## 🌐 서비스 접속 정보

| 서비스 | URL | 설명 |
|--------|-----|------|
| **Dashboard** | http://localhost:8501 | MLOps 통합 대시보드 |
| **MLflow UI** | http://localhost:5000 | 실험 추적, 모델 관리 |
| **MinIO Console** | http://localhost:9001 | 오브젝트 스토리지 관리 |

## 📦 MinIO 버킷

| 버킷 | 용도 |
|------|------|
| `raw-data` | CD 원본 데이터 |
| `raw-data-nvs` | NVS COLMAP 데이터 |
| `processed-data` | 전처리된 데이터 |
| `mlflow-artifacts` | MLflow 아티팩트 |
| `models` | 프로덕션 모델 |

## 🔧 주요 기능

### Change Detection (위성 변화탐지)
- **모델**: U-Net (SMP 기반)
- **데이터**: TorchGeo NonGeoDataset
- **출력**: GeoTIFF 변화맵, 시각화 이미지

### Novel View Synthesis (3D Gaussian Splatting)
- **모델**: gsplat 기반 3DGS
- **입력**: COLMAP 구조 (images/, sparse/)
- **출력**: PLY Point Cloud, MP4 비디오
- **특징**: SH 색상, Densification, VRAM 모니터링

## 📝 설정 예시

**configs/train_nvs.yaml:**
```yaml
model:
  sh_degree: 3

training:
  iterations: 30000
  learning_rate:
    position_lr_init: 0.00016
  densification:
    interval: 100

logging:
  mlflow:
    tracking_uri: http://localhost:5000
```

## 📝 License

MIT License