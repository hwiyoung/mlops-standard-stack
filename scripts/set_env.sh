#!/bin/bash
# ============================================
# MLflow 테스트 환경변수 설정 스크립트
# 사용법: source scripts/set_env.sh
# ============================================

# MLflow Tracking Server
export MLFLOW_TRACKING_URI="http://localhost:5000"

# MinIO (S3 호환) 설정
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minio_secure_password_2024"

# Python HTTPS 경고 무시 (로컬 개발용)
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

echo "✅ MLflow 환경변수가 설정되었습니다:"
echo "   MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "   MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL"
echo "   AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
echo ""
echo "📌 테스트 실행: python tests/test_tracking.py"
