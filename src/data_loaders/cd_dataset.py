"""
변화탐지 데이터셋 모듈
TorchGeo 기반 + MinIO 캐싱 지원
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader

try:
    from torchgeo.datasets import RasterDataset
    from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
except ImportError:
    raise ImportError("torchgeo가 필요합니다: pip install torchgeo")


class ChangeDetectionPreDataset(RasterDataset):
    """변화 전(Pre) 이미지 데이터셋"""
    filename_glob = "*.tif"
    is_image = True
    separate_files = False


class ChangeDetectionPostDataset(RasterDataset):
    """변화 후(Post) 이미지 데이터셋"""
    filename_glob = "*.tif"
    is_image = True
    separate_files = False


class ChangeDetectionMaskDataset(RasterDataset):
    """변화 마스크 데이터셋"""
    filename_glob = "*.tif"
    is_image = False
    separate_files = False


class ChangeDetectionDataModule:
    """
    변화탐지 데이터 모듈
    TorchGeo 데이터셋과 샘플러를 관리
    """
    
    def __init__(
        self,
        data_dir: str,
        pre_dir: str = "pre",
        post_dir: str = "post",
        mask_dir: str = "mask",
        patch_size: int = 256,
        batch_size: int = 8,
        samples_per_epoch: int = 1000,
        num_workers: int = 4,
        crs: Optional[str] = None,
        res: Optional[float] = None,
    ):
        """
        Args:
            data_dir: 데이터 루트 디렉토리
            pre_dir: Pre 이미지 서브디렉토리
            post_dir: Post 이미지 서브디렉토리
            mask_dir: 마스크 서브디렉토리
            patch_size: 패치 크기
            batch_size: 배치 크기
            samples_per_epoch: 에폭당 샘플 수
            num_workers: DataLoader worker 수
            crs: 좌표계 (None이면 자동)
            res: 해상도 (None이면 자동)
        """
        self.data_dir = Path(data_dir)
        self.pre_path = self.data_dir / pre_dir
        self.post_path = self.data_dir / post_dir
        self.mask_path = self.data_dir / mask_dir
        
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.crs = crs
        self.res = res
        
        self._verify_paths()
        self._setup_datasets()
    
    def _verify_paths(self):
        """경로 검증"""
        for path, name in [
            (self.pre_path, "Pre"),
            (self.post_path, "Post"),
            (self.mask_path, "Mask")
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{name} 디렉토리가 존재하지 않습니다: {path}")
            
            tif_files = list(path.glob("*.tif"))
            if not tif_files:
                raise FileNotFoundError(f"{name} 디렉토리에 TIF 파일이 없습니다: {path}")
    
    def _setup_datasets(self):
        """데이터셋 초기화"""
        print(f"📂 데이터셋 로딩: {self.data_dir}")
        
        # 개별 데이터셋 생성
        self.pre_dataset = ChangeDetectionPreDataset(
            paths=str(self.pre_path),
            crs=self.crs,
            res=self.res
        )
        
        self.post_dataset = ChangeDetectionPostDataset(
            paths=str(self.post_path),
            crs=self.crs,
            res=self.res
        )
        
        self.mask_dataset = ChangeDetectionMaskDataset(
            paths=str(self.mask_path),
            crs=self.crs,
            res=self.res
        )
        
        # 교차 데이터셋 (모든 데이터가 같은 영역 커버하는 부분만)
        self.combined_dataset = self.pre_dataset & self.post_dataset & self.mask_dataset
        
        print(f"   ✅ Pre Dataset: {len(list(self.pre_path.glob('*.tif')))} files")
        print(f"   ✅ Post Dataset: {len(list(self.post_path.glob('*.tif')))} files")  
        print(f"   ✅ Mask Dataset: {len(list(self.mask_path.glob('*.tif')))} files")
        print(f"   📊 Combined bounds: {self.combined_dataset.bounds}")
    
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        커스텀 collate 함수
        Pre/Post/Mask를 분리하여 배치 구성
        """
        pre_imgs = []
        post_imgs = []
        masks = []
        
        for sample in samples:
            img = sample['image']
            
            # TorchGeo는 이미지를 합쳐서 반환하므로 분리
            if img.shape[0] >= 6:
                pre_imgs.append(img[:3])
                post_imgs.append(img[3:6])
            else:
                # Fallback: 같은 이미지 사용
                pre_imgs.append(img[:3] if img.shape[0] >= 3 else img)
                post_imgs.append(img[:3] if img.shape[0] >= 3 else img)
            
            if 'mask' in sample:
                masks.append(sample['mask'])
        
        batch = {
            'pre': torch.stack(pre_imgs),
            'post': torch.stack(post_imgs),
        }
        
        if masks:
            batch['mask'] = torch.stack(masks)
        
        return batch
    
    def get_train_dataloader(self) -> DataLoader:
        """학습용 DataLoader 반환"""
        sampler = RandomGeoSampler(
            dataset=self.combined_dataset,
            size=self.patch_size,
            length=self.samples_per_epoch
        )
        
        return DataLoader(
            self.combined_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_dataloader(self, samples: int = 100) -> DataLoader:
        """검증용 DataLoader 반환"""
        sampler = RandomGeoSampler(
            dataset=self.combined_dataset,
            size=self.patch_size,
            length=samples
        )
        
        return DataLoader(
            self.combined_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True
        )


# MinIO 캐싱 유틸리티
class MinIODataCache:
    """
    MinIO에서 데이터를 로컬로 캐싱하는 유틸리티
    """
    
    def __init__(
        self,
        endpoint_url: str = "http://localhost:9000",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            endpoint_url: MinIO 엔드포인트
            access_key: 접근 키
            secret_key: 비밀 키
            cache_dir: 로컬 캐시 디렉토리
        """
        import boto3
        from dotenv import load_dotenv
        
        load_dotenv()
        
        self.endpoint_url = endpoint_url
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", "minio_secure_password_2024")
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "minio_cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
    
    def download_dataset(
        self,
        bucket: str,
        prefix: str,
        target_dir: Optional[str] = None,
        skip_existing: bool = True
    ) -> Path:
        """
        MinIO에서 데이터셋 다운로드
        
        Args:
            bucket: 버킷명
            prefix: 경로 prefix
            target_dir: 대상 디렉토리 (None이면 캐시 디렉토리)
            skip_existing: 이미 있는 파일 스킵
        
        Returns:
            다운로드된 디렉토리 경로
        """
        from tqdm import tqdm
        
        target = Path(target_dir) if target_dir else self.cache_dir / bucket / prefix
        target.mkdir(parents=True, exist_ok=True)
        
        # 객체 목록 조회
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        objects = []
        for page in pages:
            if 'Contents' in page:
                objects.extend(page['Contents'])
        
        if not objects:
            print(f"⚠️ 버킷에서 파일을 찾을 수 없습니다: s3://{bucket}/{prefix}")
            return target
        
        print(f"📥 MinIO에서 다운로드 중: s3://{bucket}/{prefix}")
        print(f"   📊 총 {len(objects)}개 파일")
        
        for obj in tqdm(objects, desc="다운로드"):
            key = obj['Key']
            rel_path = key[len(prefix):].lstrip('/')
            
            if not rel_path:
                continue
            
            local_path = target / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if skip_existing and local_path.exists():
                continue
            
            self.s3_client.download_file(bucket, key, str(local_path))
        
        print(f"   ✅ 다운로드 완료: {target}")
        return target


def build_data_module(config) -> ChangeDetectionDataModule:
    """
    Config에서 DataModule 빌드
    """
    if hasattr(config, 'data'):
        data_cfg = config.data.local
        data_dir = data_cfg.root_dir
        pre_dir = data_cfg.pre_dir
        post_dir = data_cfg.post_dir
        mask_dir = data_cfg.mask_dir
    else:
        data_cfg = config.get("data", {}).get("local", {})
        data_dir = data_cfg.get("root_dir", "./data/change_detection")
        pre_dir = data_cfg.get("pre_dir", "pre")
        post_dir = data_cfg.get("post_dir", "post")
        mask_dir = data_cfg.get("mask_dir", "mask")
    
    if hasattr(config, 'torchgeo'):
        tg_cfg = config.torchgeo
        patch_size = tg_cfg.patch_size
        samples_per_epoch = tg_cfg.samples_per_epoch
    else:
        tg_cfg = config.get("torchgeo", {})
        patch_size = tg_cfg.get("patch_size", 256)
        samples_per_epoch = tg_cfg.get("samples_per_epoch", 1000)
    
    if hasattr(config, 'training'):
        batch_size = config.training.batch_size
    else:
        batch_size = config.get("training", {}).get("batch_size", 8)
    
    if hasattr(config, 'hardware'):
        num_workers = config.hardware.num_workers
    else:
        num_workers = config.get("hardware", {}).get("num_workers", 4)
    
    return ChangeDetectionDataModule(
        data_dir=data_dir,
        pre_dir=pre_dir,
        post_dir=post_dir,
        mask_dir=mask_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        num_workers=num_workers
    )


# 가짜 데이터 생성 유틸리티 (테스트용)
def create_dummy_data(data_dir: str, size: int = 1024) -> Path:
    """테스트용 더미 데이터 생성"""
    data_path = Path(data_dir)
    pre_dir = data_path / "pre"
    post_dir = data_path / "post"
    mask_dir = data_path / "mask"
    
    for d in [pre_dir, post_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 좌표계 설정
    west, south, east, north = 126.9, 37.5, 127.0, 37.6
    transform = from_bounds(west, south, east, north, size, size)
    crs = CRS.from_epsg(4326)
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': size,
        'height': size,
        'count': 3,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    # Pre 이미지
    np.random.seed(42)
    pre_data = np.random.randint(0, 255, (3, size, size), dtype=np.uint8)
    with rasterio.open(pre_dir / "pre_image.tif", 'w', **profile) as dst:
        dst.write(pre_data)
    
    # Post 이미지
    np.random.seed(123)
    post_data = np.random.randint(0, 255, (3, size, size), dtype=np.uint8)
    with rasterio.open(post_dir / "post_image.tif", 'w', **profile) as dst:
        dst.write(post_data)
    
    # Mask
    profile['count'] = 1
    mask_data = np.zeros((1, size, size), dtype=np.uint8)
    for _ in range(10):
        x, y = np.random.randint(0, size-100), np.random.randint(0, size-100)
        w, h = np.random.randint(20, 100), np.random.randint(20, 100)
        mask_data[0, y:y+h, x:x+w] = 1
    
    with rasterio.open(mask_dir / "mask.tif", 'w', **profile) as dst:
        dst.write(mask_data)
    
    print(f"✅ 더미 데이터 생성: {data_path}")
    return data_path


if __name__ == "__main__":
    # 테스트
    print("=== DataModule 테스트 ===")
    
    # 더미 데이터 생성
    data_dir = "./data/cd_test"
    create_dummy_data(data_dir)
    
    # DataModule 생성
    dm = ChangeDetectionDataModule(
        data_dir=data_dir,
        patch_size=256,
        batch_size=4,
        samples_per_epoch=10,
        num_workers=0
    )
    
    # DataLoader 테스트
    train_loader = dm.get_train_dataloader()
    
    for batch in train_loader:
        print(f"Pre shape: {batch['pre'].shape}")
        print(f"Post shape: {batch['post'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        break
