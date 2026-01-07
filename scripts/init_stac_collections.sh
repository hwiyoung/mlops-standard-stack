#!/bin/bash
# ============================================
# STAC 초기 컬렉션 생성 스크립트
# ============================================

STAC_API_URL="${STAC_API_URL:-http://localhost:8080}"

echo "🚀 STAC 컬렉션 초기화 중..."

# 드론 사진 컬렉션 생성
curl -X POST "${STAC_API_URL}/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "drone-photos",
    "type": "Collection",
    "title": "드론 사진",
    "description": "드론으로 촬영한 항공 사진 컬렉션",
    "license": "proprietary",
    "stac_version": "1.0.0",
    "stac_extensions": [
      "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json",
      "https://stac-extensions.github.io/view/v1.0.0/schema.json"
    ],
    "extent": {
      "spatial": {"bbox": [[124.0, 33.0, 132.0, 43.0]]},
      "temporal": {"interval": [["2020-01-01T00:00:00Z", null]]}
    },
    "links": []
  }' 2>/dev/null && echo " ✅ drone-photos 컬렉션 생성됨" || echo " ⚠️ drone-photos 이미 존재하거나 오류"

# 정사영상 컬렉션 생성
curl -X POST "${STAC_API_URL}/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "orthoimages",
    "type": "Collection",
    "title": "정사영상",
    "description": "정사보정된 항공/위성 영상 컬렉션",
    "license": "proprietary",
    "stac_version": "1.0.0",
    "stac_extensions": [
      "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
      "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
    ],
    "extent": {
      "spatial": {"bbox": [[124.0, 33.0, 132.0, 43.0]]},
      "temporal": {"interval": [["2020-01-01T00:00:00Z", null]]}
    },
    "links": []
  }' 2>/dev/null && echo " ✅ orthoimages 컬렉션 생성됨" || echo " ⚠️ orthoimages 이미 존재하거나 오류"

echo "🎉 STAC 컬렉션 초기화 완료!"
