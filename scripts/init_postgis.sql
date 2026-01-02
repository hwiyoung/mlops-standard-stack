-- ============================================
-- PostGIS 초기화 및 이미지 메타데이터 테이블 생성
-- ============================================

-- PostGIS 확장 활성화
CREATE EXTENSION IF NOT EXISTS postgis;

-- 이미지 메타데이터 테이블
CREATE TABLE IF NOT EXISTS image_metadata (
    id SERIAL PRIMARY KEY,
    bucket VARCHAR(255) NOT NULL,
    object_key VARCHAR(1024) NOT NULL UNIQUE,
    filename VARCHAR(512),
    
    -- 데이터 유형: 'photo' (드론/항공 사진) 또는 'ortho' (정사영상)
    data_type VARCHAR(50) DEFAULT 'photo',
    
    -- 사진: 중심점 (Point), 정사영상: 범위 (Polygon)
    location GEOMETRY(Point, 4326),
    extent GEOMETRY(Polygon, 4326),
    
    captured_at TIMESTAMP,
    file_size BIGINT,
    width INT,
    height INT,
    crs VARCHAR(50),
    resolution FLOAT,
    thumbnail_key VARCHAR(1024),
    indexed_at TIMESTAMP DEFAULT NOW()
);

-- 공간 인덱스
CREATE INDEX IF NOT EXISTS idx_location ON image_metadata USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_extent ON image_metadata USING GIST(extent);

-- 일반 인덱스
CREATE INDEX IF NOT EXISTS idx_captured_at ON image_metadata(captured_at);
CREATE INDEX IF NOT EXISTS idx_data_type ON image_metadata(data_type);
CREATE INDEX IF NOT EXISTS idx_bucket ON image_metadata(bucket);
