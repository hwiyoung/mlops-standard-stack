# NVS Training Docker Image
# CUDA + PyTorch + gsplat for 3D Gaussian Splatting

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    colmap \
    imagemagick \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 12.1)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# gsplat 설치
RUN pip install gsplat

# 추가 의존성
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 작업 디렉토리
WORKDIR /workspace

# 프로젝트 복사
COPY . /workspace

# 환경변수 설정
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minio_secure_password_2024
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# 기본 명령어
CMD ["python", "src/training/train_nvs.py", "--config", "configs/train_nvs.yaml"]
