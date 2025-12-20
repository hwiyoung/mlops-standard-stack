#!/bin/bash
# ============================================
# MinIO 초기화 스크립트
# 버킷 자동 생성 및 정책 설정
# ============================================

set -e

# 환경변수 로드 (.env 파일에서)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# 기본값 설정
MINIO_ENDPOINT="${MINIO_S3_ENDPOINT_URL:-http://localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ROOT_USER:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_ROOT_PASSWORD:-minio_secure_password_2024}"

# 생성할 버킷 목록
BUCKETS=("raw-data" "mlflow-artifacts" "processed-data" "models")

echo "============================================"
echo "MinIO 초기화 스크립트"
echo "============================================"
echo "Endpoint: $MINIO_ENDPOINT"
echo "Buckets to create: ${BUCKETS[*]}"
echo ""

# MinIO 연결 대기 함수
wait_for_minio() {
    echo "⏳ MinIO 서버 연결 대기 중..."
    MAX_RETRIES=30
    RETRY_INTERVAL=2
    
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s "$MINIO_ENDPOINT/minio/health/live" > /dev/null 2>&1; then
            echo "✅ MinIO 서버가 준비되었습니다!"
            return 0
        fi
        echo "   시도 $i/$MAX_RETRIES - 재시도 중..."
        sleep $RETRY_INTERVAL
    done
    
    echo "❌ MinIO 서버에 연결할 수 없습니다. docker-compose up -d 로 먼저 서비스를 시작하세요."
    exit 1
}

# mc (MinIO Client) 설치 확인
check_mc_installed() {
    if command -v mc &> /dev/null; then
        echo "✅ MinIO Client (mc) 가 설치되어 있습니다."
        return 0
    else
        echo "⚠️  MinIO Client (mc) 가 설치되어 있지 않습니다."
        echo "   설치 방법:"
        echo "   curl -O https://dl.min.io/client/mc/release/linux-amd64/mc"
        echo "   chmod +x mc && sudo mv mc /usr/local/bin/"
        echo ""
        echo "📌 Python 방식으로 대체 실행합니다..."
        return 1
    fi
}

# mc를 사용한 버킷 생성
create_buckets_with_mc() {
    echo ""
    echo "🔧 mc alias 설정 중..."
    mc alias set myminio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api S3v4
    
    echo ""
    echo "📦 버킷 생성 중..."
    for bucket in "${BUCKETS[@]}"; do
        if mc ls myminio/"$bucket" > /dev/null 2>&1; then
            echo "   ⏭️  '$bucket' 버킷이 이미 존재합니다."
        else
            mc mb myminio/"$bucket"
            echo "   ✅ '$bucket' 버킷이 생성되었습니다."
        fi
    done
    
    echo ""
    echo "📋 현재 버킷 목록:"
    mc ls myminio/
}

# Python을 사용한 버킷 생성 (mc가 없을 경우)
create_buckets_with_python() {
    python3 << PYTHON_SCRIPT
import os
import sys

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("❌ minio 패키지가 설치되어 있지 않습니다.")
    print("   pip install minio 로 설치하세요.")
    sys.exit(1)

endpoint = "${MINIO_ENDPOINT}".replace("http://", "").replace("https://", "")
access_key = "${MINIO_ACCESS_KEY}"
secret_key = "${MINIO_SECRET_KEY}"
buckets = "${BUCKETS[*]}".split()

print(f"🔧 MinIO 클라이언트 연결 중... ({endpoint})")

client = Minio(
    endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False
)

print("")
print("📦 버킷 생성 중...")
for bucket in buckets:
    try:
        if client.bucket_exists(bucket):
            print(f"   ⏭️  '{bucket}' 버킷이 이미 존재합니다.")
        else:
            client.make_bucket(bucket)
            print(f"   ✅ '{bucket}' 버킷이 생성되었습니다.")
    except S3Error as e:
        print(f"   ❌ '{bucket}' 버킷 생성 실패: {e}")

print("")
print("📋 현재 버킷 목록:")
for bucket in client.list_buckets():
    print(f"   - {bucket.name}")

PYTHON_SCRIPT
}

# 메인 실행
main() {
    wait_for_minio
    
    if check_mc_installed; then
        create_buckets_with_mc
    else
        create_buckets_with_python
    fi
    
    echo ""
    echo "============================================"
    echo "✅ MinIO 초기화 완료!"
    echo "============================================"
    echo ""
    echo "📌 접속 정보:"
    echo "   - MinIO Console: http://localhost:9001"
    echo "   - S3 API:        $MINIO_ENDPOINT"
    echo "   - Access Key:    $MINIO_ACCESS_KEY"
    echo ""
}

main "$@"
