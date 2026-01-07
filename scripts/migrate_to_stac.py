#!/usr/bin/env python3
"""
image_metadata 테이블에서 STAC Items으로 데이터 마이그레이션
기존 데이터를 STAC API에 등록합니다.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.indexer.stac_client import STACClient


def get_db_connection():
    """PostgreSQL 연결"""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "mlflow"),
        password=os.getenv("POSTGRES_PASSWORD", "mlflow123"),
        dbname=os.getenv("POSTGRES_DB", "mlflow"),
    )


def migrate_photos(stac: STACClient, limit: int = None):
    """드론 사진 마이그레이션"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    query = """
        SELECT id, bucket, object_key, filename, 
               ST_X(location) as lon, ST_Y(location) as lat,
               thumbnail_key, file_size, captured_at
        FROM image_metadata 
        WHERE location IS NOT NULL AND data_type = 'photo'
    """
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query)
    rows = cur.fetchall()
    
    print(f"📷 드론 사진 {len(rows)}개 마이그레이션 시작...")
    
    success = 0
    failed = 0
    minio_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    
    for row in rows:
        id_, bucket, key, filename, lon, lat, thumb_key, file_size, captured_at = row
        
        if lon is None or lat is None:
            failed += 1
            continue
        
        item_id = f"{bucket}-{key.replace('/', '-').replace('.', '-')}"
        
        # 에셋 URL
        assets = {
            "image": {
                "href": f"{minio_url}/{bucket}/{key}",
                "type": "image/jpeg",
                "roles": ["data"]
            }
        }
        if thumb_key:
            assets["thumbnail"] = {
                "href": f"{minio_url}/{bucket}/{thumb_key}",
                "type": "image/jpeg",
                "roles": ["thumbnail"]
            }
        
        datetime_str = captured_at.isoformat() if captured_at else datetime.now(timezone.utc).isoformat()
        
        item = stac.create_drone_photo_item(
            item_id=item_id,
            longitude=lon,
            latitude=lat,
            datetime_str=datetime_str,
            assets=assets,
            properties={
                "filename": filename,
                "bucket": bucket,
                "object_key": key,
                "file_size": file_size or 0,
            }
        )
        
        if stac.add_item("drone-photos", item):
            success += 1
        else:
            failed += 1
    
    cur.close()
    conn.close()
    
    print(f"   ✅ 성공: {success}개, ❌ 실패: {failed}개")
    return success, failed


def migrate_orthos(stac: STACClient, limit: int = None):
    """정사영상 마이그레이션"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    query = """
        SELECT id, bucket, object_key, filename,
               ST_AsGeoJSON(extent) as extent_geojson,
               ST_XMin(extent) as minx, ST_YMin(extent) as miny,
               ST_XMax(extent) as maxx, ST_YMax(extent) as maxy,
               thumbnail_key, file_size, crs, resolution, captured_at
        FROM image_metadata 
        WHERE extent IS NOT NULL AND data_type = 'ortho'
    """
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query)
    rows = cur.fetchall()
    
    print(f"🗺️ 정사영상 {len(rows)}개 마이그레이션 시작...")
    
    success = 0
    failed = 0
    minio_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    
    for row in rows:
        import json
        import re
        
        id_, bucket, key, filename, extent_json, minx, miny, maxx, maxy, thumb_key, file_size, crs, resolution, captured_at = row
        
        if extent_json is None:
            failed += 1
            continue
        
        item_id = f"{bucket}-{key.replace('/', '-').replace('.', '-')}"
        
        # geometry
        geometry = json.loads(extent_json)
        bbox = [minx, miny, maxx, maxy]
        
        # 에셋 URL
        assets = {
            "image": {
                "href": f"{minio_url}/{bucket}/{key}",
                "type": "image/tiff",
                "roles": ["data"]
            }
        }
        if thumb_key:
            assets["thumbnail"] = {
                "href": f"{minio_url}/{bucket}/{thumb_key}",
                "type": "image/jpeg",
                "roles": ["thumbnail"]
            }
        
        # EPSG 추출
        epsg = None
        if crs:
            epsg_match = re.search(r"EPSG:(\d+)", crs)
            if epsg_match:
                epsg = int(epsg_match.group(1))
        
        datetime_str = captured_at.isoformat() if captured_at else datetime.now(timezone.utc).isoformat()
        
        item = stac.create_orthoimage_item(
            item_id=item_id,
            bbox=bbox,
            geometry=geometry,
            datetime_str=datetime_str,
            assets=assets,
            epsg=epsg,
            resolution=resolution,
            properties={
                "filename": filename,
                "bucket": bucket,
                "object_key": key,
                "file_size": file_size or 0,
            }
        )
        
        if stac.add_item("orthoimages", item):
            success += 1
        else:
            failed += 1
    
    cur.close()
    conn.close()
    
    print(f"   ✅ 성공: {success}개, ❌ 실패: {failed}개")
    return success, failed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="image_metadata를 STAC으로 마이그레이션")
    parser.add_argument("--limit", "-l", type=int, help="마이그레이션할 최대 개수")
    parser.add_argument("--photos-only", action="store_true", help="사진만 마이그레이션")
    parser.add_argument("--orthos-only", action="store_true", help="정사영상만 마이그레이션")
    
    args = parser.parse_args()
    
    stac = STACClient()
    
    # 연결 확인
    if not stac.health_check():
        print("❌ STAC API에 연결할 수 없습니다.")
        print(f"   URL: {stac.base_url}")
        sys.exit(1)
    
    print(f"🚀 STAC API 연결됨: {stac.base_url}")
    
    # 컬렉션 확인
    collections = stac.get_collections()
    print(f"📁 컬렉션: {[c['id'] for c in collections]}")
    
    if not collections:
        print("⚠️ 컬렉션이 없습니다. init_stac_collections.sh를 먼저 실행하세요.")
        sys.exit(1)
    
    total_success = 0
    total_failed = 0
    
    if not args.orthos_only:
        s, f = migrate_photos(stac, args.limit)
        total_success += s
        total_failed += f
    
    if not args.photos_only:
        s, f = migrate_orthos(stac, args.limit)
        total_success += s
        total_failed += f
    
    print(f"\n🎉 마이그레이션 완료!")
    print(f"   총 성공: {total_success}개")
    print(f"   총 실패: {total_failed}개")


if __name__ == "__main__":
    main()
