# MLOps Standard Stack

**Standard Open Source Stack 기반의 MLOps 환경**

위성 이미지 변화탐지(Change Detection)와 Novel View Synthesis(NVS) 연구를 위한 MLOps 인프라입니다.

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    MLOps Stack                          │
├─────────────────────────────────────────────────────────┤
│  📊 MLflow (Tracking Server)     → localhost:5000       │
│  📦 MinIO (S3 Compatible)        → localhost:9000/9001  │
│  🗄️ PostgreSQL (Metadata DB)    → localhost:5432       │
└─────────────────────────────────────────────────────────┘
```

## 📁 프로젝트 구조

```
mlops-standard-stack/
├── docker-compose.yml     # 인프라 서비스 정의
├── .env                   # 환경변수 (Git 제외)
│
├── src/                   # 프로덕션 소스 코드
│   ├── models/            # 모델 정의
│   ├── data_loaders/      # 데이터 로드 클래스
│   ├── training/          # 학습 스크립트
│   └── utils/             # 유틸리티 함수
│
├── examples/              # 데모/예제 스크립트
│   ├── demo_cd_torchgeo.py    # 변화탐지 TorchGeo 데모
│   └── demo_nvs_dummy.py      # NVS Gaussian Splatting 데모
│
├── tests/                 # 테스트 코드
│   └── test_tracking.py   # MLflow 연동 테스트
│
├── scripts/               # 실행 스크립트
│   ├── set_env.sh         # 환경변수 설정
│   └── init_minio.sh      # MinIO 버킷 초기화
│
├── configs/               # 설정 파일
└── data/                  # 데이터 폴더 (Git 제외)
```

## 🚀 Quick Start

### 1. 인프라 시작
```bash
docker-compose up -d
```

### 2. Python 환경 설정
```bash
mamba create -n mlops python=3.11 -y
mamba activate mlops
pip install -r requirements.txt
```

### 3. 환경변수 설정
```bash
source scripts/set_env.sh
```

### 4. 예제 실행
```bash
# MLflow 연동 테스트
python tests/test_tracking.py

# 변화탐지 데모
python examples/demo_cd_torchgeo.py

# NVS 데모
python examples/demo_nvs_dummy.py
```

## 🌐 접속 정보

| 서비스 | URL | 설명 |
|--------|-----|------|
| MLflow UI | http://localhost:5000 | 실험 추적, 모델 관리 |
| MinIO Console | http://localhost:9001 | 오브젝트 스토리지 관리 |

## 📦 생성되는 MinIO 버킷

| 버킷 | 용도 |
|------|------|
| `raw-data` | 원본 데이터 |
| `processed-data` | 전처리된 데이터 |
| `mlflow-artifacts` | MLflow 아티팩트 |
| `models` | 프로덕션 모델 |

## 🔧 실제 모델 적용

### Change Detection
`src/models/`에 실제 모델 구현 후 `examples/demo_cd_torchgeo.py`의 `MockChangeDetectionModel`을 교체

### NVS (Gaussian Splatting)
`gsplat`, `nerfstudio` 등 실제 라이브러리 사용 권장

## 📝 License

MIT License