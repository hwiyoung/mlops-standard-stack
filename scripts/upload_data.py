#!/usr/bin/env python3
"""
MinIO 데이터 업로드 스크립트
로컬 폴더의 파일들을 MinIO 버킷으로 재귀적으로 업로드

사용법:
    python scripts/upload_data.py --source ./data/raw --bucket raw-data
    python scripts/upload_data.py -s ./data/images -b raw-data --skip-existing
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm


def load_minio_config() -> dict:
    """
    .env 파일에서 MinIO 접속 정보 로드
    
    Returns:
        MinIO 설정 딕셔너리
    """
    # .env 파일 로드
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    config = {
        "endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        "access_key": os.getenv("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ROOT_USER", "minioadmin")),
        "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minio_secure_password_2024")),
    }
    
    return config


def create_s3_client(config: dict):
    """
    boto3 S3 클라이언트 생성 (MinIO 호환)
    
    Args:
        config: MinIO 설정
    
    Returns:
        boto3 S3 클라이언트
    """
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
    )


def get_local_files(source_dir: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    로컬 폴더에서 파일 목록 재귀적으로 수집
    
    Args:
        source_dir: 소스 디렉토리
        extensions: 필터링할 확장자 (None이면 모든 파일)
    
    Returns:
        파일 경로 리스트
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"소스 디렉토리가 존재하지 않습니다: {source_dir}")
    
    files = []
    for path in source_dir.rglob("*"):
        if path.is_file():
            if extensions is None:
                files.append(path)
            elif path.suffix.lower() in extensions:
                files.append(path)
    
    return sorted(files)


def check_object_exists(s3_client, bucket: str, key: str) -> bool:
    """
    MinIO에 객체가 이미 존재하는지 확인
    
    Args:
        s3_client: S3 클라이언트
        bucket: 버킷명
        key: 객체 키
    
    Returns:
        존재 여부
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_file_with_progress(
    s3_client,
    local_path: Path,
    bucket: str,
    key: str,
    pbar: tqdm
) -> bool:
    """
    파일을 MinIO에 업로드 (진행률 콜백 포함)
    
    Args:
        s3_client: S3 클라이언트
        local_path: 로컬 파일 경로
        bucket: 대상 버킷
        key: 객체 키
        pbar: tqdm 프로그레스 바
    
    Returns:
        업로드 성공 여부
    """
    file_size = local_path.stat().st_size
    
    def progress_callback(bytes_transferred):
        pbar.update(bytes_transferred)
    
    try:
        s3_client.upload_file(
            str(local_path),
            bucket,
            key,
            Callback=progress_callback
        )
        return True
    except Exception as e:
        print(f"\n❌ 업로드 실패: {local_path} - {e}")
        return False


def upload_directory(
    source_dir: str,
    bucket: str,
    prefix: str = "",
    skip_existing: bool = False,
    extensions: Optional[List[str]] = None,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    디렉토리를 MinIO 버킷으로 업로드
    
    Args:
        source_dir: 소스 디렉토리
        bucket: 대상 버킷
        prefix: 버킷 내 경로 prefix
        skip_existing: 이미 존재하는 파일 스킵
        extensions: 필터링할 확장자
        dry_run: 실제 업로드 없이 테스트만
    
    Returns:
        (업로드된 파일 수, 스킵된 파일 수, 실패한 파일 수)
    """
    source_path = Path(source_dir).resolve()
    
    print("=" * 60)
    print("📤 MinIO 데이터 업로드")
    print("=" * 60)
    
    # MinIO 설정 로드
    config = load_minio_config()
    print(f"📡 MinIO Endpoint: {config['endpoint_url']}")
    print(f"🪣 Target Bucket: {bucket}")
    print(f"📁 Source Directory: {source_path}")
    if prefix:
        print(f"📂 Prefix: {prefix}")
    print()
    
    # S3 클라이언트 생성
    s3_client = create_s3_client(config)
    
    # 버킷 존재 확인
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            print(f"❌ 버킷이 존재하지 않습니다: {bucket}")
            print(f"   먼저 버킷을 생성하세요: docker-compose up -d")
            return 0, 0, 0
        raise
    
    # 파일 목록 수집
    print("🔍 파일 스캔 중...")
    files = get_local_files(source_path, extensions)
    
    if not files:
        print("⚠️  업로드할 파일이 없습니다.")
        return 0, 0, 0
    
    # 총 크기 계산
    total_size = sum(f.stat().st_size for f in files)
    print(f"   📊 총 {len(files)}개 파일, {total_size / (1024*1024):.2f} MB")
    print()
    
    if dry_run:
        print("🧪 Dry Run 모드 - 실제 업로드 없음")
        for f in files[:10]:  # 처음 10개만 표시
            rel_path = f.relative_to(source_path)
            key = f"{prefix}/{rel_path}".lstrip("/") if prefix else str(rel_path)
            print(f"   📄 {key}")
        if len(files) > 10:
            print(f"   ... 외 {len(files) - 10}개 파일")
        return len(files), 0, 0
    
    # 업로드 실행
    uploaded = 0
    skipped = 0
    failed = 0
    
    print("📤 업로드 시작...")
    
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="전체 진행률") as pbar:
        for local_file in files:
            rel_path = local_file.relative_to(source_path)
            key = f"{prefix}/{rel_path}".lstrip("/") if prefix else str(rel_path)
            
            # 이미 존재하는 파일 스킵
            if skip_existing and check_object_exists(s3_client, bucket, key):
                skipped += 1
                pbar.update(local_file.stat().st_size)
                continue
            
            # 업로드
            if upload_file_with_progress(s3_client, local_file, bucket, key, pbar):
                uploaded += 1
            else:
                failed += 1
    
    # 결과 출력
    print()
    print("=" * 60)
    print("✅ 업로드 완료!")
    print("=" * 60)
    print(f"   📤 업로드됨: {uploaded}개")
    print(f"   ⏭️  스킵됨: {skipped}개")
    print(f"   ❌ 실패: {failed}개")
    print()
    print(f"📌 MinIO Console에서 확인: http://localhost:9001/browser/{bucket}")
    
    return uploaded, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="로컬 파일을 MinIO 버킷으로 업로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 업로드
  python scripts/upload_data.py --source ./data/raw --bucket raw-data

  # prefix 지정
  python scripts/upload_data.py -s ./images -b raw-data --prefix project1/images

  # 이미 있는 파일 스킵
  python scripts/upload_data.py -s ./data -b raw-data --skip-existing

  # 특정 확장자만 업로드
  python scripts/upload_data.py -s ./data -b raw-data --extensions .tif .tiff .jpg

  # Dry run (테스트)
  python scripts/upload_data.py -s ./data -b raw-data --dry-run
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="업로드할 로컬 디렉토리 경로"
    )
    
    parser.add_argument(
        "--bucket", "-b",
        type=str,
        default="raw-data",
        help="대상 MinIO 버킷 (기본값: raw-data)"
    )
    
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        default="",
        help="버킷 내 경로 prefix (예: project1/images)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="이미 존재하는 파일은 스킵"
    )
    
    parser.add_argument(
        "--extensions", "-e",
        type=str,
        nargs="*",
        default=None,
        help="업로드할 파일 확장자 (예: .tif .jpg .png)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 업로드 없이 테스트만 수행"
    )
    
    args = parser.parse_args()
    
    # 확장자 처리
    extensions = None
    if args.extensions:
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in args.extensions]
    
    # 업로드 실행
    upload_directory(
        source_dir=args.source,
        bucket=args.bucket,
        prefix=args.prefix,
        skip_existing=args.skip_existing,
        extensions=extensions,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
