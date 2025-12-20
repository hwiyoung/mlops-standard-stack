# MLflow 서버용 커스텀 Dockerfile
# PostgreSQL + S3(MinIO) 지원을 위한 추가 패키지 포함

FROM ghcr.io/mlflow/mlflow:v2.17.2

# PostgreSQL 및 S3 드라이버 설치
RUN pip install --no-cache-dir \
    psycopg2-binary \
    boto3

# 헬스체크 스크립트
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1
