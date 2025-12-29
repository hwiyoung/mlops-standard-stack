#!/usr/bin/env python3
"""
NVS 데이터 업로드 스크립트
COLMAP 처리된 폴더 구조를 유지하면서 MinIO에 업로드

폴더 구조 예시:
    project_folder/
    ├── images/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    └── (optional) dense/
        └── ...

사용법:
    python scripts/upload_nvs_data.py --source ./my_scene --project my_project
    python scripts/upload_nvs_data.py -s ./scene -p project_name --date 2024-01-15
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm


def load_minio_config() -> dict:
    """
    .env 파일에서 MinIO 접속 정보 로드
    """
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
    """boto3 S3 클라이언트 생성"""
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
    )


def ensure_bucket_exists(s3_client, bucket: str) -> bool:
    """버킷이 없으면 생성"""
    try:
        s3_client.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            print(f"📦 버킷 생성 중: {bucket}")
            s3_client.create_bucket(Bucket=bucket)
            return True
        raise


def validate_colmap_structure(source_dir: Path) -> dict:
    """
    COLMAP 폴더 구조 검증
    
    Returns:
        검증 결과 딕셔너리
    """
    result = {
        "valid": True,
        "images_dir": None,
        "sparse_dir": None,
        "dense_dir": None,
        "warnings": [],
        "errors": []
    }
    
    # images 폴더 확인
    images_dir = source_dir / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.[jJ][pP][gG]")) + \
                      list(images_dir.glob("*.[pP][nN][gG]")) + \
                      list(images_dir.glob("*.[jJ][pP][eE][gG]"))
        if image_files:
            result["images_dir"] = images_dir
            result["image_count"] = len(image_files)
        else:
            result["warnings"].append("images/ 폴더에 이미지 파일이 없습니다.")
    else:
        result["errors"].append("images/ 폴더가 없습니다.")
        result["valid"] = False
    
    # sparse 폴더 확인
    sparse_dir = source_dir / "sparse"
    if sparse_dir.exists():
        # sparse/0 또는 sparse/ 직접 확인
        sparse_0 = sparse_dir / "0"
        if sparse_0.exists():
            result["sparse_dir"] = sparse_0
        else:
            result["sparse_dir"] = sparse_dir
        
        # COLMAP 파일 확인
        check_dir = result["sparse_dir"]
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        for f in required_files:
            if not (check_dir / f).exists():
                # .txt 형식도 허용
                if not (check_dir / f.replace(".bin", ".txt")).exists():
                    result["warnings"].append(f"sparse 폴더에 {f} 파일이 없습니다.")
    else:
        result["warnings"].append("sparse/ 폴더가 없습니다 (COLMAP 결과 필요).")
    
    # dense 폴더 확인 (선택사항)
    dense_dir = source_dir / "dense"
    if dense_dir.exists():
        result["dense_dir"] = dense_dir
    
    return result


def get_all_files(source_dir: Path) -> List[Path]:
    """재귀적으로 모든 파일 수집"""
    files = []
    for path in source_dir.rglob("*"):
        if path.is_file():
            files.append(path)
    return sorted(files)


def check_object_exists(s3_client, bucket: str, key: str) -> bool:
    """MinIO에 객체가 존재하는지 확인"""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_nvs_data(
    source_dir: str,
    bucket: str = "raw-data-nvs",
    project_name: Optional[str] = None,
    date_prefix: Optional[str] = None,
    skip_existing: bool = True,
    validate: bool = True,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    NVS 데이터를 MinIO에 업로드
    
    Args:
        source_dir: COLMAP 처리된 소스 디렉토리
        bucket: 대상 MinIO 버킷
        project_name: 프로젝트명 (prefix)
        date_prefix: 날짜 prefix (없으면 오늘 날짜)
        skip_existing: 이미 있는 파일 스킵
        validate: COLMAP 구조 검증
        dry_run: 실제 업로드 없이 테스트
    
    Returns:
        (업로드 수, 스킵 수, 실패 수)
    """
    source_path = Path(source_dir).resolve()
    
    print("=" * 60)
    print("📤 NVS 데이터 업로드 (COLMAP 구조)")
    print("=" * 60)
    
    # 소스 검증
    if not source_path.exists():
        raise FileNotFoundError(f"소스 디렉토리가 없습니다: {source_path}")
    
    # COLMAP 구조 검증
    if validate:
        print("\n🔍 COLMAP 폴더 구조 검증...")
        validation = validate_colmap_structure(source_path)
        
        if validation["images_dir"]:
            print(f"   ✅ images/: {validation.get('image_count', 0)}개 이미지")
        if validation["sparse_dir"]:
            print(f"   ✅ sparse/: {validation['sparse_dir'].relative_to(source_path)}")
        if validation["dense_dir"]:
            print(f"   ✅ dense/: 있음")
        
        for warn in validation["warnings"]:
            print(f"   ⚠️  {warn}")
        for err in validation["errors"]:
            print(f"   ❌ {err}")
        
        if not validation["valid"]:
            print("\n❌ COLMAP 구조 검증 실패. --no-validate 옵션으로 우회 가능.")
            return 0, 0, 0
    
    # Prefix 생성
    if date_prefix is None:
        date_prefix = datetime.now().strftime("%Y%m%d")
    
    if project_name:
        prefix = f"{project_name}/{date_prefix}"
    else:
        # 소스 폴더 이름을 프로젝트명으로 사용
        project_name = source_path.name
        prefix = f"{project_name}/{date_prefix}"
    
    # MinIO 설정
    config = load_minio_config()
    print(f"\n📡 MinIO Endpoint: {config['endpoint_url']}")
    print(f"🪣 Target Bucket: {bucket}")
    print(f"📁 Source: {source_path}")
    print(f"📂 Prefix: {prefix}/")
    
    # S3 클라이언트
    s3_client = create_s3_client(config)
    
    # 버킷 확인/생성
    ensure_bucket_exists(s3_client, bucket)
    
    # 파일 목록 수집
    print("\n🔍 파일 스캔 중...")
    files = get_all_files(source_path)
    
    if not files:
        print("⚠️  업로드할 파일이 없습니다.")
        return 0, 0, 0
    
    # 통계
    total_size = sum(f.stat().st_size for f in files)
    print(f"   📊 총 {len(files)}개 파일, {total_size / (1024*1024):.2f} MB")
    
    # 파일 타입별 통계
    file_types = {}
    for f in files:
        ext = f.suffix.lower() or "(no ext)"
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"   📄 파일 타입: {', '.join(f'{k}({v})' for k, v in sorted(file_types.items()))}")
    
    if dry_run:
        print("\n🧪 Dry Run 모드 - 실제 업로드 없음")
        print("\n업로드 예정 파일 (처음 10개):")
        for f in files[:10]:
            rel_path = f.relative_to(source_path)
            key = f"{prefix}/{rel_path}"
            print(f"   📄 s3://{bucket}/{key}")
        if len(files) > 10:
            print(f"   ... 외 {len(files) - 10}개 파일")
        return len(files), 0, 0
    
    # 업로드 실행
    print("\n📤 업로드 시작...")
    
    uploaded = 0
    skipped = 0
    failed = 0
    
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="전체 진행률") as pbar:
        for local_file in files:
            rel_path = local_file.relative_to(source_path)
            key = f"{prefix}/{rel_path}"
            file_size = local_file.stat().st_size
            
            # 이미 존재하는 파일 스킵
            if skip_existing and check_object_exists(s3_client, bucket, key):
                skipped += 1
                pbar.update(file_size)
                continue
            
            # 업로드
            try:
                def progress_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                s3_client.upload_file(
                    str(local_file),
                    bucket,
                    key,
                    Callback=progress_callback
                )
                uploaded += 1
            except Exception as e:
                print(f"\n❌ 업로드 실패: {rel_path} - {e}")
                failed += 1
                pbar.update(file_size)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("✅ 업로드 완료!")
    print("=" * 60)
    print(f"   📤 업로드됨: {uploaded}개")
    print(f"   ⏭️  스킵됨: {skipped}개")
    print(f"   ❌ 실패: {failed}개")
    print(f"\n📂 업로드 경로: s3://{bucket}/{prefix}/")
    print(f"📌 MinIO Console: http://localhost:9001/browser/{bucket}/{prefix}/")
    
    # 메타데이터 파일 생성 및 업로드
    meta_content = f"""# NVS Dataset Metadata
project: {project_name}
upload_date: {datetime.now().isoformat()}
source_path: {source_path}
total_files: {len(files)}
total_size_mb: {total_size / (1024*1024):.2f}

## Structure
- images: {validation.get('image_count', 'N/A')} files
- sparse: {'yes' if validation.get('sparse_dir') else 'no'}
- dense: {'yes' if validation.get('dense_dir') else 'no'}
"""
    
    meta_key = f"{prefix}/_metadata.md"
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=meta_key,
            Body=meta_content.encode('utf-8'),
            ContentType='text/markdown'
        )
        print(f"   📝 메타데이터: s3://{bucket}/{meta_key}")
    except Exception as e:
        print(f"   ⚠️ 메타데이터 업로드 실패: {e}")
    
    return uploaded, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="NVS 데이터(COLMAP 구조) MinIO 업로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 업로드 (프로젝트명: 폴더이름, 날짜: 오늘)
  python scripts/upload_nvs_data.py --source ./my_scene

  # 프로젝트명과 날짜 지정
  python scripts/upload_nvs_data.py -s ./scene -p campus_building --date 2024-01-15

  # 검증 없이 업로드
  python scripts/upload_nvs_data.py -s ./data --no-validate

  # Dry run (테스트)
  python scripts/upload_nvs_data.py -s ./data -p test --dry-run
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="COLMAP 처리된 소스 디렉토리"
    )
    
    parser.add_argument(
        "--bucket", "-b",
        type=str,
        default="raw-data-nvs",
        help="대상 MinIO 버킷 (기본값: raw-data-nvs)"
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="프로젝트명 (prefix). 없으면 소스 폴더 이름 사용"
    )
    
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="날짜 prefix (YYYYMMDD 형식). 없으면 오늘 날짜"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="이미 존재하는 파일 스킵 (기본값: True)"
    )
    
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="이미 존재하는 파일도 덮어쓰기"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="COLMAP 구조 검증 스킵"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 업로드 없이 테스트"
    )
    
    args = parser.parse_args()
    
    # 업로드 실행
    upload_nvs_data(
        source_dir=args.source,
        bucket=args.bucket,
        project_name=args.project,
        date_prefix=args.date,
        skip_existing=not args.no_skip,
        validate=not args.no_validate,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
