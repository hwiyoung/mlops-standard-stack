FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# stac-fastapi-pgstac 설치
RUN pip install --no-cache-dir \
    stac-fastapi.pgstac==2.5.0 \
    uvicorn[standard] \
    psycopg[pool,binary]

# 환경 변수 (docker-compose에서 오버라이드)
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8080
ENV POSTGRES_USER=mlflow
ENV POSTGRES_PASSWORD=mlflow123
ENV POSTGRES_DB=mlflow
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432

EXPOSE 8080

# stac-fastapi 실행
CMD ["uvicorn", "stac_fastapi.pgstac.app:app", "--host", "0.0.0.0", "--port", "8080"]
