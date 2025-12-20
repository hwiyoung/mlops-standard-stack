#!/usr/bin/env python3
"""
위성 이미지 변화탐지(Change Detection) 학습 스크립트
TorchGeo 기반 대용량 GeoTIFF 패치 샘플링 + MLflow 로깅
"""

import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.samplers import RandomGeoSampler


# ============================================
# 1. 가짜 대용량 GeoTIFF 데이터 생성
# ============================================
def create_fake_geotiff(filepath: str, size: int = 1024, num_bands: int = 3, seed: int = None):
    """
    TorchGeo가 인식 가능한 가짜 GeoTIFF 생성
    
    Args:
        filepath: 저장 경로
        size: 이미지 크기 (size x size)
        num_bands: 밴드 수
        seed: 랜덤 시드
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 가짜 위성 이미지 데이터 생성 (uint8, 0-255)
    data = np.random.randint(0, 255, (num_bands, size, size), dtype=np.uint8)
    
    # 좌표계 및 Transform 설정 (EPSG:4326 - WGS84)
    # 서울 근처 좌표로 설정
    west, south, east, north = 126.9, 37.5, 127.0, 37.6
    transform = from_bounds(west, south, east, north, size, size)
    crs = CRS.from_epsg(4326)
    
    # GeoTIFF 저장
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': size,
        'height': size,
        'count': num_bands,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data)
    
    print(f"✅ GeoTIFF 생성: {filepath} ({size}x{size}, {num_bands} bands)")
    return filepath


def create_fake_mask(filepath: str, size: int = 1024, seed: int = None):
    """변화탐지 마스크 생성 (0: 변화없음, 1: 변화있음)"""
    if seed is not None:
        np.random.seed(seed)
    
    # 랜덤 변화 영역 생성 (약 10-20% 영역에 변화)
    mask = np.zeros((1, size, size), dtype=np.uint8)
    
    # 몇 개의 랜덤 사각형 영역에 변화 표시
    num_changes = random.randint(5, 15)
    for _ in range(num_changes):
        x1 = random.randint(0, size - 100)
        y1 = random.randint(0, size - 100)
        w = random.randint(20, 100)
        h = random.randint(20, 100)
        mask[0, y1:y1+h, x1:x1+w] = 1
    
    # 좌표계 설정
    west, south, east, north = 126.9, 37.5, 127.0, 37.6
    transform = from_bounds(west, south, east, north, size, size)
    crs = CRS.from_epsg(4326)
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': size,
        'height': size,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(mask)
    
    print(f"✅ 마스크 생성: {filepath} ({size}x{size})")
    return filepath


# ============================================
# 2. TorchGeo 커스텀 데이터셋
# ============================================
class ChangeDetectionDataset(RasterDataset):
    """변화탐지용 커스텀 RasterDataset"""
    
    filename_glob = "*.tif"
    is_image = True
    separate_files = False
    
    def __init__(self, paths, crs=None, res=None, transforms=None):
        super().__init__(paths=paths, crs=crs, res=res, transforms=transforms)


class ChangeDetectionMaskDataset(RasterDataset):
    """변화탐지 마스크 데이터셋"""
    
    filename_glob = "*.tif"
    is_image = False
    separate_files = False
    
    def __init__(self, paths, crs=None, res=None, transforms=None):
        super().__init__(paths=paths, crs=crs, res=res, transforms=transforms)


# ============================================
# 3. Mock 모델 (실제로는 U-Net 등 사용)
# ============================================
class MockChangeDetectionModel(nn.Module):
    """변화탐지 Mock 모델"""
    
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()
        # 간단한 Conv 레이어 (실제로는 U-Net, Siamese 등 사용)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, pre, post):
        # Pre/Post 이미지를 채널 방향으로 결합
        x = torch.cat([pre, post], dim=1)
        return self.conv(x)


# ============================================
# 4. 메트릭 계산
# ============================================
def calculate_iou(pred, target, num_classes=2):
    """IoU (Intersection over Union) 계산"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


# ============================================
# 5. 시각화 함수
# ============================================
def visualize_batch(pre_img, post_img, pred_mask, true_mask, save_path):
    """배치 시각화 및 저장"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Pre Image (RGB)
    pre_np = pre_img[0].permute(1, 2, 0).cpu().numpy()
    pre_np = (pre_np - pre_np.min()) / (pre_np.max() - pre_np.min() + 1e-8)
    axes[0, 0].imshow(pre_np[:, :, :3])
    axes[0, 0].set_title("Pre-change Image", fontsize=14)
    axes[0, 0].axis('off')
    
    # Post Image (RGB)
    post_np = post_img[0].permute(1, 2, 0).cpu().numpy()
    post_np = (post_np - post_np.min()) / (post_np.max() - post_np.min() + 1e-8)
    axes[0, 1].imshow(post_np[:, :, :3])
    axes[0, 1].set_title("Post-change Image", fontsize=14)
    axes[0, 1].axis('off')
    
    # True Mask
    axes[1, 0].imshow(true_mask[0, 0].cpu().numpy(), cmap='RdYlGn_r')
    axes[1, 0].set_title("Ground Truth Mask", fontsize=14)
    axes[1, 0].axis('off')
    
    # Predicted Mask
    pred_np = pred_mask[0].argmax(dim=0).cpu().numpy()
    axes[1, 1].imshow(pred_np, cmap='RdYlGn_r')
    axes[1, 1].set_title("Predicted Change Mask", fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 시각화 저장: {save_path}")
    return save_path


# ============================================
# 6. 메인 학습 함수
# ============================================
def train_change_detection(
    data_dir: str = "./data/change_detection",
    patch_size: int = 256,
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 0.001
):
    """TorchGeo 기반 변화탐지 학습 파이프라인"""
    
    print("=" * 60)
    print("🛰️  위성 이미지 변화탐지 학습 파이프라인")
    print("=" * 60)
    
    # 데이터 디렉토리 생성
    data_path = Path(data_dir)
    pre_dir = data_path / "pre"
    post_dir = data_path / "post"
    mask_dir = data_path / "mask"
    
    for d in [pre_dir, post_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 가짜 GeoTIFF 데이터 생성
    print("\n📦 가짜 대용량 GeoTIFF 데이터 생성 중...")
    create_fake_geotiff(str(pre_dir / "pre_image.tif"), size=1024, seed=42)
    create_fake_geotiff(str(post_dir / "post_image.tif"), size=1024, seed=123)
    create_fake_mask(str(mask_dir / "change_mask.tif"), size=1024, seed=42)
    
    # TorchGeo 데이터셋 로드
    print("\n📂 TorchGeo 데이터셋 로딩...")
    pre_dataset = ChangeDetectionDataset(paths=str(pre_dir))
    post_dataset = ChangeDetectionDataset(paths=str(post_dir))
    mask_dataset = ChangeDetectionMaskDataset(paths=str(mask_dir))
    
    # 데이터셋 교차 (같은 영역만 사용)
    combined_dataset = pre_dataset & post_dataset & mask_dataset
    
    print(f"   - Pre Dataset bounds: {pre_dataset.bounds}")
    print(f"   - Combined Dataset length: {len(combined_dataset)}")
    
    # RandomGeoSampler로 패치 추출
    sampler = RandomGeoSampler(
        dataset=combined_dataset,
        size=patch_size,
        length=100  # 에폭당 100개 패치
    )
    
    # Custom collate function
    def collate_fn(samples):
        pre_imgs = []
        post_imgs = []
        masks = []
        
        for sample in samples:
            # sample['image']에서 pre/post 분리 (6채널 -> 3+3)
            img = sample['image']
            if img.shape[0] == 6:
                pre_imgs.append(img[:3])
                post_imgs.append(img[3:])
            else:
                pre_imgs.append(img)
                post_imgs.append(img)
            
            if 'mask' in sample:
                masks.append(sample['mask'])
        
        return {
            'pre': torch.stack(pre_imgs),
            'post': torch.stack(post_imgs),
            'mask': torch.stack(masks) if masks else None
        }
    
    # DataLoader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"   - Patch size: {patch_size}x{patch_size}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Samples per epoch: {len(sampler)}")
    
    # 모델, 손실함수, 옵티마이저
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    model = MockChangeDetectionModel(in_channels=6, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # MLflow 실험 시작
    experiment_name = "satellite-change-detection"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"cd-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # 파라미터 로깅
        mlflow.log_params({
            "patch_size": patch_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "model": "MockChangeDetectionModel",
            "device": str(device)
        })
        
        print("\n🚀 학습 시작...")
        first_batch_saved = False
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_iou = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                pre_img = batch['pre'].float().to(device)
                post_img = batch['post'].float().to(device)
                
                # Forward pass
                outputs = model(pre_img, post_img)
                
                # 임의의 타겟 생성 (실제로는 mask 사용)
                target = torch.randint(0, 2, (outputs.shape[0], outputs.shape[2], outputs.shape[3])).to(device)
                
                # Loss 계산
                loss = criterion(outputs, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 메트릭 계산
                pred = outputs.argmax(dim=1)
                iou = calculate_iou(pred, target)
                
                epoch_loss += loss.item()
                epoch_iou += iou
                num_batches += 1
                
                # 첫 번째 배치 시각화 저장
                if not first_batch_saved and epoch == 1 and batch_idx == 0:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        viz_path = os.path.join(tmpdir, "change_detection_viz.png")
                        visualize_batch(
                            pre_img, post_img, outputs.detach(),
                            target.unsqueeze(1), viz_path
                        )
                        mlflow.log_artifact(viz_path)
                        first_batch_saved = True
            
            # 에폭 메트릭 계산 및 로깅
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_iou = epoch_iou / max(num_batches, 1)
            
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_iou": avg_iou
            }, step=epoch)
            
            print(f"   Epoch [{epoch}/{num_epochs}] Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")
        
        print("\n✅ 학습 완료!")
        print(f"   - MLflow UI: http://localhost:5000")
        print(f"   - Experiment: {experiment_name}")


if __name__ == "__main__":
    train_change_detection(
        data_dir="./data/change_detection",
        patch_size=256,
        batch_size=4,
        num_epochs=10,
        learning_rate=0.001
    )
