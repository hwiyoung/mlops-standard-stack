FROM python:3.11-slim

WORKDIR /workspace

# 시스템 패키지 및 mc CLI 설치
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc \
    && chmod +x /usr/local/bin/mc

# Python 의존성
COPY dashboard/requirements.txt /tmp/dashboard-requirements.txt
RUN pip install --no-cache-dir -r /tmp/dashboard-requirements.txt

# 프로젝트 복사
COPY . /workspace

# Streamlit 설정
RUN mkdir -p ~/.streamlit
RUN echo '\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
' > ~/.streamlit/config.toml

# mc alias 자동 설정 및 Streamlit 실행 스크립트
RUN echo '#!/bin/bash\n\
mc alias set myminio http://minio:9000 ${AWS_ACCESS_KEY_ID:-minioadmin} ${AWS_SECRET_ACCESS_KEY:-minioadmin123} --api S3v4 2>/dev/null || true\n\
exec streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8501

CMD ["/entrypoint.sh"]
