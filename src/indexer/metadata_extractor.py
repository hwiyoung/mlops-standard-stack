#!/usr/bin/env python3
"""
이미지 메타데이터 추출 모듈
- JPEG: EXIF GPS 태그에서 위치 추출
- GeoTIFF: rasterio를 통해 bounds(범위) 추출
"""

import os
import io
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

import boto3
from PIL import Image
from dotenv import load_dotenv

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_minio_client():
    """MinIO S3 클라이언트 생성"""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ROOT_USER", "minioadmin")),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")),
    )


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


def extract_exif_gps(image_data: bytes) -> Optional[Tuple[float, float]]:
    """
    JPEG 이미지에서 EXIF GPS 좌표 추출
    
    Returns:
        (longitude, latitude) 또는 None
    """
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        
        img = Image.open(io.BytesIO(image_data))
        exif_data = img._getexif()
        
        if not exif_data:
            return None
        
        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
        
        if not gps_info:
            return None
        
        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + float(m) / 60 + float(s) / 3600
        
        lat = convert_to_degrees(gps_info.get("GPSLatitude", (0, 0, 0)))
        lon = convert_to_degrees(gps_info.get("GPSLongitude", (0, 0, 0)))
        
        if gps_info.get("GPSLatitudeRef") == "S":
            lat = -lat
        if gps_info.get("GPSLongitudeRef") == "W":
            lon = -lon
        
        if lat == 0 and lon == 0:
            return None
            
        return (lon, lat)
        
    except Exception as e:
        print(f"EXIF 추출 실패: {e}")
        return None


def extract_geotiff_extent(file_path: str) -> Optional[Dict[str, Any]]:
    """
    GeoTIFF에서 범위(bounds) 및 메타데이터 추출
    
    Returns:
        {
            "extent_wkt": "POLYGON(...)",
            "crs": "EPSG:4326",
            "resolution": 0.5,
            "width": 1000,
            "height": 1000
        }
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from shapely.geometry import box
        
        with rasterio.open(file_path) as src:
            bounds = src.bounds
            crs = src.crs
            
            # WGS84로 변환
            if crs and crs.to_epsg() != 4326:
                bounds = transform_bounds(crs, "EPSG:4326", *bounds)
            
            extent_polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
            
            return {
                "extent_wkt": extent_polygon.wkt,
                "crs": crs.to_string() if crs else None,
                "resolution": src.res[0] if src.res else None,
                "width": src.width,
                "height": src.height,
            }
            
    except Exception as e:
        print(f"GeoTIFF 추출 실패: {e}")
        return None


def create_thumbnail(image_data: bytes, max_size: int = 200) -> Optional[bytes]:
    """썸네일 생성 (이미지 데이터 기준)"""
    try:
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail((max_size, max_size))
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()
        
    except Exception as e:
        print(f"썸네일 생성 실패: {e}")
        return None


def create_geotiff_thumbnail(file_path: str, max_size: int = 200) -> Optional[bytes]:
    """GeoTIFF에서 썸네일 생성"""
    try:
        import rasterio
        from rasterio.enums import Resampling
        import numpy as np
        
        with rasterio.open(file_path) as src:
            # 해상도 계산 (가장 긴 축이 max_size가 되도록)
            h, w = src.height, src.width
            if h == 0 or w == 0:
                return None
            
            ratio = max_size / max(h, w)
            new_h, new_w = max(1, int(h * ratio)), max(1, int(w * ratio))

            # 밴드 읽기 (RGB 시도)
            bands = []
            indexes = src.indexes[:3] if len(src.indexes) >= 3 else [1]
            for i in indexes:
                data = src.read(i, out_shape=(new_h, new_w), resampling=Resampling.bilinear)
                bands.append(data)
            
            if not bands:
                return None
            
            # 데이터 구성
            if len(bands) >= 3:
                rgb = np.stack(bands[:3], axis=2)
            else:
                rgb = bands[0]
            
            # 정규화 (최소/최대 스케일링)
            rgb = rgb.astype(float)
            for i in range(rgb.shape[2] if len(rgb.shape) == 3 else 1):
                layer = rgb[:,:,i] if len(rgb.shape) == 3 else rgb
                l_min, l_max = layer.min(), layer.max()
                if l_max > l_min:
                    layer = (layer - l_min) / (l_max - l_min) * 255
                else:
                    layer = np.zeros_like(layer)
                
                if len(rgb.shape) == 3:
                    rgb[:,:,i] = layer
                else:
                    rgb = layer
            
            img = Image.fromarray(rgb.astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            return buffer.getvalue()
            
    except Exception as e:
        print(f"GeoTIFF 썸네일 생성 실패: {e}")
        return None


def index_object(bucket: str, key: str, use_stac: bool = True) -> bool:
    """
    MinIO 객체의 메타데이터를 추출하여 STAC API 또는 DB에 저장
    
    Args:
        bucket: MinIO 버킷명
        key: 객체 키
        use_stac: True면 STAC API 사용, False면 기존 DB 직접 저장
    
    Returns:
        성공 여부
    """
    from datetime import timezone
    
    s3 = get_minio_client()
    filename = Path(key).name
    suffix = Path(key).suffix.lower()
    
    # 썸네일 폴더는 인덱싱 제외
    if key.startswith("thumbnails/") or "/thumbnails/" in key:
        return False
    
    # 지원 형식 확인
    if suffix not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        print(f"지원하지 않는 형식: {suffix}")
        return False
    
    try:
        # 파일 정보 조회
        head = s3.head_object(Bucket=bucket, Key=key)
        file_size = head.get("ContentLength", 0)
        last_modified = head.get("LastModified")
        
        # MinIO 기본 URL
        minio_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        
        data_type = "photo"
        longitude = None
        latitude = None
        bbox = None
        geometry = None
        crs = None
        resolution = None
        width = None
        height = None
        thumbnail_key = None
        captured_at = last_modified.isoformat() if last_modified else datetime.now(timezone.utc).isoformat()
        
        if suffix in [".tif", ".tiff"]:
            # GeoTIFF 처리
            data_type = "ortho"
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                s3.download_file(bucket, key, tmp.name)
                # 메타데이터 추출
                result = extract_geotiff_extent(tmp.name)
                # 썸네일 생성
                thumbnail = create_geotiff_thumbnail(tmp.name)
                os.unlink(tmp.name)
            
            if result:
                # WKT에서 좌표 추출
                import re
                extent_wkt = result["extent_wkt"]
                coords = re.findall(r"[-\d.]+", extent_wkt)
                if len(coords) >= 8:
                    x_coords = [float(coords[i]) for i in range(0, len(coords), 2)]
                    y_coords = [float(coords[i]) for i in range(1, len(coords), 2)]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [[
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]],
                            [bbox[0], bbox[1]]
                        ]]
                    }
                
                crs = result.get("crs")
                resolution = result.get("resolution")
                width = result.get("width")
                height = result.get("height")
            
            if thumbnail:
                thumb_key = f"thumbnails/{Path(key).stem}_thumb.jpg"
                s3.put_object(Bucket=bucket, Key=thumb_key, Body=thumbnail, ContentType="image/jpeg")
                thumbnail_key = thumb_key
        else:
            # JPEG/PNG 처리
            response = s3.get_object(Bucket=bucket, Key=key)
            image_data = response["Body"].read()
            
            # GPS 추출
            coords = extract_exif_gps(image_data)
            if coords:
                longitude, latitude = coords
            
            # EXIF 촬영 시간 추출
            try:
                from PIL.ExifTags import TAGS
                img = Image.open(io.BytesIO(image_data))
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == "DateTimeOriginal":
                            # EXIF 형식: "2024:01:15 10:30:00"
                            captured_at = datetime.strptime(value, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat()
                            break
                width, height = img.size
            except Exception:
                pass
            
            # 썸네일 생성 및 업로드
            thumbnail = create_thumbnail(image_data)
            if thumbnail:
                thumb_key = f"thumbnails/{Path(key).stem}_thumb.jpg"
                s3.put_object(Bucket=bucket, Key=thumb_key, Body=thumbnail, ContentType="image/jpeg")
                thumbnail_key = thumb_key
        
        # STAC 모드
        if use_stac:
            from .stac_client import STACClient
            
            stac = STACClient()
            item_id = f"{bucket}-{key.replace('/', '-').replace('.', '-')}"
            
            # Asset URL 생성
            image_url = f"{minio_url}/{bucket}/{key}"
            assets = {
                "image": {
                    "href": image_url,
                    "type": "image/tiff" if data_type == "ortho" else f"image/{suffix[1:]}",
                    "roles": ["data"]
                }
            }
            if thumbnail_key:
                assets["thumbnail"] = {
                    "href": f"{minio_url}/{bucket}/{thumbnail_key}",
                    "type": "image/jpeg",
                    "roles": ["thumbnail"]
                }
            
            # 추가 속성
            properties = {
                "filename": filename,
                "bucket": bucket,
                "object_key": key,
                "file_size": file_size,
            }
            if width:
                properties["width"] = width
            if height:
                properties["height"] = height
            
            if data_type == "ortho" and geometry and bbox:
                # 정사영상: orthoimages 컬렉션
                epsg = None
                if crs:
                    import re
                    epsg_match = re.search(r"EPSG:(\d+)", crs)
                    if epsg_match:
                        epsg = int(epsg_match.group(1))
                
                item = stac.create_orthoimage_item(
                    item_id=item_id,
                    bbox=bbox,
                    geometry=geometry,
                    datetime_str=captured_at,
                    assets=assets,
                    epsg=epsg,
                    resolution=resolution,
                    properties=properties,
                )
                success = stac.add_item("orthoimages", item)
            elif longitude and latitude:
                # 드론 사진: drone-photos 컬렉션
                item = stac.create_drone_photo_item(
                    item_id=item_id,
                    longitude=longitude,
                    latitude=latitude,
                    datetime_str=captured_at,
                    assets=assets,
                    properties=properties,
                )
                success = stac.add_item("drone-photos", item)
            else:
                print(f"⚠️ 위치 정보 없음: {key}")
                return False
            
            return success
        
        # 기존 DB 모드 (레거시)
        else:
            location_wkt = f"POINT({longitude} {latitude})" if longitude and latitude else None
            extent_wkt = None
            if geometry:
                from shapely.geometry import shape
                extent_wkt = shape(geometry).wkt
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO image_metadata 
                (bucket, object_key, filename, data_type, location, extent, 
                 file_size, width, height, crs, resolution, thumbnail_key)
                VALUES (%s, %s, %s, %s, 
                        ST_GeomFromText(%s, 4326), 
                        ST_GeomFromText(%s, 4326),
                        %s, %s, %s, %s, %s, %s)
                ON CONFLICT (object_key) DO UPDATE SET
                    data_type = EXCLUDED.data_type,
                    location = EXCLUDED.location,
                    extent = EXCLUDED.extent,
                    file_size = EXCLUDED.file_size,
                    width = EXCLUDED.width,
                    height = EXCLUDED.height,
                    crs = EXCLUDED.crs,
                    resolution = EXCLUDED.resolution,
                    thumbnail_key = EXCLUDED.thumbnail_key,
                    indexed_at = NOW()
            """, (bucket, key, filename, data_type, location_wkt, extent_wkt,
                  file_size, width, height, crs, resolution, thumbnail_key))
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"✅ 인덱싱 완료: {key} ({data_type})")
            return True
        
    except Exception as e:
        print(f"❌ 인덱싱 실패 ({key}): {e}")
        import traceback
        traceback.print_exc()
        return False


def index_bucket(bucket: str, prefix: str = "", use_stac: bool = True) -> Tuple[int, int]:
    """
    버킷 전체 인덱싱
    
    Args:
        bucket: MinIO 버킷명
        prefix: 경로 prefix
        use_stac: True면 STAC API 사용, False면 기존 DB 직접 저장
    
    Returns:
        (성공 수, 실패 수)
    """
    s3 = get_minio_client()
    paginator = s3.get_paginator("list_objects_v2")
    
    success = 0
    failed = 0
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue
        
        for obj in page["Contents"]:
            key = obj["Key"]
            if index_object(bucket, key, use_stac=use_stac):
                success += 1
            else:
                failed += 1
    
    print(f"\n📊 인덱싱 완료: 성공 {success}개, 실패 {failed}개")
    return success, failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MinIO 이미지 메타데이터 인덱싱")
    parser.add_argument("--bucket", "-b", default="raw-data", help="대상 버킷")
    parser.add_argument("--prefix", "-p", default="", help="경로 prefix")
    parser.add_argument("--key", "-k", help="단일 객체 키")
    parser.add_argument("--legacy", action="store_true", help="기존 DB 모드 사용 (STAC 대신)")
    
    args = parser.parse_args()
    use_stac = not args.legacy
    
    print(f"🔄 인덱싱 모드: {'STAC API' if use_stac else 'Legacy DB'}")
    
    if args.key:
        index_object(args.bucket, args.key, use_stac=use_stac)
    else:
        index_bucket(args.bucket, args.prefix, use_stac=use_stac)

