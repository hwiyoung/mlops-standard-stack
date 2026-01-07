#!/usr/bin/env python3
"""
STAC API 클라이언트 모듈
- STAC Item 생성 및 API 호출
- projection, timestamps, view, eo 확장 지원
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List
from pathlib import Path

import requests
from dotenv import load_dotenv

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# STAC 확장 URL
STAC_EXTENSIONS = {
    "projection": "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
    "timestamps": "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json",
    "view": "https://stac-extensions.github.io/view/v1.0.0/schema.json",
    "eo": "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
}


class STACClient:
    """STAC API 클라이언트"""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("STAC_API_URL", "http://localhost:8080")
    
    def create_drone_photo_item(
        self,
        item_id: str,
        longitude: float,
        latitude: float,
        datetime_str: str,
        assets: Dict[str, Dict],
        properties: Optional[Dict] = None,
        view_azimuth: Optional[float] = None,
        view_off_nadir: Optional[float] = None,
        sun_elevation: Optional[float] = None,
    ) -> Dict:
        """
        드론 사진용 STAC Item 생성
        
        Args:
            item_id: 고유 ID
            longitude: 경도
            latitude: 위도
            datetime_str: ISO 8601 형식 날짜
            assets: 자산 정보 (image, thumbnail 등)
            properties: 추가 속성
            view_azimuth: 카메라 방위각
            view_off_nadir: 천저각
            sun_elevation: 태양 고도
        """
        now = datetime.now(timezone.utc).isoformat()
        
        item = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": item_id,
            "geometry": {
                "type": "Point",
                "coordinates": [longitude, latitude]
            },
            "bbox": [longitude, latitude, longitude, latitude],
            "properties": {
                "datetime": datetime_str,
                "created": now,
                "updated": now,
                **(properties or {})
            },
            "assets": assets,
            "links": [],
            "stac_extensions": [
                STAC_EXTENSIONS["timestamps"],
            ]
        }
        
        # view 확장 (촬영 조건)
        if any([view_azimuth, view_off_nadir, sun_elevation]):
            item["stac_extensions"].append(STAC_EXTENSIONS["view"])
            if view_azimuth is not None:
                item["properties"]["view:azimuth"] = view_azimuth
            if view_off_nadir is not None:
                item["properties"]["view:off_nadir"] = view_off_nadir
            if sun_elevation is not None:
                item["properties"]["view:sun_elevation"] = sun_elevation
        
        return item
    
    def create_orthoimage_item(
        self,
        item_id: str,
        bbox: List[float],
        geometry: Dict,
        datetime_str: str,
        assets: Dict[str, Dict],
        epsg: Optional[int] = None,
        resolution: Optional[float] = None,
        properties: Optional[Dict] = None,
        bands: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        정사영상용 STAC Item 생성
        
        Args:
            item_id: 고유 ID
            bbox: [minx, miny, maxx, maxy]
            geometry: GeoJSON geometry
            datetime_str: ISO 8601 형식 날짜
            assets: 자산 정보
            epsg: EPSG 코드
            resolution: 해상도 (m)
            properties: 추가 속성
            bands: EO 밴드 정보
        """
        now = datetime.now(timezone.utc).isoformat()
        
        item = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": item_id,
            "geometry": geometry,
            "bbox": bbox,
            "properties": {
                "datetime": datetime_str,
                "created": now,
                "updated": now,
                **(properties or {})
            },
            "assets": assets,
            "links": [],
            "stac_extensions": [
                STAC_EXTENSIONS["timestamps"],
            ]
        }
        
        # projection 확장
        if epsg or resolution:
            item["stac_extensions"].append(STAC_EXTENSIONS["projection"])
            if epsg:
                item["properties"]["proj:epsg"] = epsg
            if resolution:
                item["properties"]["proj:resolution"] = [resolution, resolution]
        
        # eo 확장
        if bands:
            item["stac_extensions"].append(STAC_EXTENSIONS["eo"])
            item["properties"]["eo:bands"] = bands
        
        return item
    
    def add_item(self, collection_id: str, item: Dict) -> bool:
        """
        STAC API에 Item 추가
        
        Args:
            collection_id: 컬렉션 ID (drone-photos 또는 orthoimages)
            item: STAC Item JSON
        
        Returns:
            성공 여부
        """
        url = f"{self.base_url}/collections/{collection_id}/items"
        
        try:
            response = requests.post(
                url,
                json=item,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ STAC Item 추가됨: {item['id']}")
                return True
            elif response.status_code == 409:
                # 이미 존재하는 경우 업데이트 시도
                return self.update_item(collection_id, item)
            else:
                print(f"❌ STAC Item 추가 실패: {response.status_code} - {response.text[:200]}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ STAC API 연결 실패: {e}")
            return False
    
    def update_item(self, collection_id: str, item: Dict) -> bool:
        """기존 Item 업데이트"""
        url = f"{self.base_url}/collections/{collection_id}/items/{item['id']}"
        
        try:
            response = requests.put(
                url,
                json=item,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code in [200, 204]:
                print(f"✅ STAC Item 업데이트됨: {item['id']}")
                return True
            else:
                print(f"❌ STAC Item 업데이트 실패: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ STAC API 연결 실패: {e}")
            return False
    
    def search(
        self,
        collections: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        datetime_range: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        STAC 검색
        
        Args:
            collections: 검색할 컬렉션 목록
            bbox: 공간 범위 [minx, miny, maxx, maxy]
            datetime_range: 시간 범위 (예: "2024-01-01/2024-12-31")
            limit: 최대 결과 수
        """
        url = f"{self.base_url}/search"
        
        params = {"limit": limit}
        if collections:
            params["collections"] = collections
        if bbox:
            params["bbox"] = bbox
        if datetime_range:
            params["datetime"] = datetime_range
        
        try:
            response = requests.post(url, json=params, timeout=30)
            if response.status_code == 200:
                return response.json().get("features", [])
            else:
                print(f"검색 실패: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"검색 실패: {e}")
            return []
    
    def get_collections(self) -> List[Dict]:
        """컬렉션 목록 조회"""
        url = f"{self.base_url}/collections"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get("collections", [])
            return []
        except requests.exceptions.RequestException:
            return []
    
    def health_check(self) -> bool:
        """API 상태 확인"""
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


# 테스트
if __name__ == "__main__":
    client = STACClient()
    
    print(f"🔍 STAC API URL: {client.base_url}")
    
    if client.health_check():
        print("✅ STAC API 연결됨")
        collections = client.get_collections()
        print(f"📁 컬렉션: {[c['id'] for c in collections]}")
    else:
        print("❌ STAC API에 연결할 수 없음")
