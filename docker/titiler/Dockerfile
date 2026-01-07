FROM ghcr.io/developmentseed/titiler:0.18.0

# 환경 변수 설정
ENV PORT=8082
ENV HOST=0.0.0.0
ENV WORKERS=1

# MinIO 연결용 S3 환경 변수 (docker-compose에서 오버라이드)
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin123
ENV AWS_S3_ENDPOINT=http://minio:9000
ENV AWS_HTTPS=NO
ENV AWS_VIRTUAL_HOSTING=FALSE

EXPOSE 8082

CMD ["uvicorn", "titiler.application.main:app", "--host", "0.0.0.0", "--port", "8082"]
