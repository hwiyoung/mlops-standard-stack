#!/bin/bash
# ============================================
# MinIO CLI (mc) 설치 및 설정 스크립트
# ============================================
# 용도: 대용량 데이터 업로드를 위한 mc CLI 자동 설치 및 alias 등록
# 사용법: ./scripts/setup_minio_cli.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "🔧 MinIO CLI (mc) 설치 및 설정"
echo "============================================"

# .env 파일 로드
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "📄 .env 파일 로드 중..."
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
else
    echo "⚠️  .env 파일이 없습니다. 기본값을 사용합니다."
fi

# 환경변수 기본값 설정
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minio_secure_password_2024}"
PUBLIC_IP="${PUBLIC_IP:-localhost}"

# mc 설치 확인
if command -v mc &> /dev/null; then
    echo "✅ mc가 이미 설치되어 있습니다: $(which mc)"
else
    echo "📥 mc 다운로드 중..."
    
    # OS 감지
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    case "$ARCH" in
        x86_64) ARCH="amd64" ;;
        aarch64) ARCH="arm64" ;;
        *) echo "❌ 지원하지 않는 아키텍처: $ARCH"; exit 1 ;;
    esac
    
    MC_URL="https://dl.min.io/client/mc/release/${OS}-${ARCH}/mc"
    
    # 다운로드 및 설치
    curl -fsSL "$MC_URL" -o /tmp/mc
    chmod +x /tmp/mc
    
    # 설치 위치 결정
    if [ -w /usr/local/bin ]; then
        sudo mv /tmp/mc /usr/local/bin/mc
        echo "✅ mc 설치 완료: /usr/local/bin/mc"
    else
        mkdir -p "$HOME/.local/bin"
        mv /tmp/mc "$HOME/.local/bin/mc"
        echo "✅ mc 설치 완료: $HOME/.local/bin/mc"
        echo "⚠️  PATH에 $HOME/.local/bin 추가가 필요할 수 있습니다."
    fi
fi

# alias 설정
echo ""
echo "🔗 MinIO alias 설정 중..."

# 외부 접속용 endpoint 설정
if [ "$PUBLIC_IP" != "localhost" ] && [ "$PUBLIC_IP" != "127.0.0.1" ]; then
    EXTERNAL_ENDPOINT="http://${PUBLIC_IP}:9000"
else
    EXTERNAL_ENDPOINT="$MINIO_ENDPOINT"
fi

# alias 등록
mc alias set myminio "$EXTERNAL_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" --api S3v4

echo ""
echo "============================================"
echo "✅ 설정 완료!"
echo "============================================"
echo ""
echo "📌 등록된 alias:"
mc alias list myminio
echo ""
echo "📖 사용 예시:"
echo ""
echo "   # 버킷 목록 확인"
echo "   mc ls myminio"
echo ""
echo "   # 폴더 업로드 (대용량)"
echo "   mc mirror ./local_folder/ myminio/raw-data/project/"
echo ""
echo "   # 끊긴 업로드 이어서 진행"
echo "   mc mirror --continue ./local_folder/ myminio/raw-data/project/"
echo ""
echo "   # 파일 다운로드"
echo "   mc cp myminio/raw-data/file.tif ./local/"
echo ""
echo "   # 버킷 동기화 (양방향)"
echo "   mc mirror --watch ./local/ myminio/raw-data/"
echo ""
