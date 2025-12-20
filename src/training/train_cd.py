#!/usr/bin/env python3
"""
변화탐지(Change Detection) 학습 스크립트
YAML 설정 파일 기반 학습 + MLflow 로깅 + 모델 레지스트리

사용법:
    python src/training/train_cd.py --config configs/train_cd.yaml
    python src/training/train_cd.py -c configs/train_cd.yaml -o training.epochs=100
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가 (임포트 전에)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env 파일에서 환경변수 로드 (AWS_ACCESS_KEY_ID 등)
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ 환경변수 로드: {env_file}")

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.config import parse_args_with_config, load_config
from src.models.unet import build_model, build_loss
from src.data_loaders.cd_dataset import build_data_module, create_dummy_data


# ============================================
# 메트릭 계산
# ============================================
def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """
    세그멘테이션 메트릭 계산
    
    Args:
        pred: 예측 logits [B, C, H, W]
        target: 타겟 마스크 [B, H, W] 또는 [B, 1, H, W]
        num_classes: 클래스 수
    
    Returns:
        메트릭 딕셔너리
    """
    pred_mask = pred.argmax(dim=1).flatten()
    target_mask = target.flatten() if target.dim() == 3 else target[:, 0].flatten()
    
    # IoU 계산
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        target_cls = (target_mask == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    # Accuracy
    correct = (pred_mask == target_mask).sum().float()
    total = target_mask.numel()
    accuracy = (correct / total).item()
    
    # F1 Score (binary)
    tp = ((pred_mask == 1) & (target_mask == 1)).sum().float()
    fp = ((pred_mask == 1) & (target_mask == 0)).sum().float()
    fn = ((pred_mask == 0) & (target_mask == 1)).sum().float()
    
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "iou": mean_iou,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================
# 시각화
# ============================================
def visualize_predictions(
    pre: torch.Tensor,
    post: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    save_path: str,
    num_samples: int = 4
):
    """예측 결과 시각화"""
    import matplotlib.pyplot as plt
    
    num_samples = min(num_samples, pre.shape[0])
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Pre
        pre_img = pre[i].permute(1, 2, 0).cpu().numpy()
        pre_img = (pre_img - pre_img.min()) / (pre_img.max() - pre_img.min() + 1e-8)
        axes[i, 0].imshow(pre_img[:, :, :3])
        axes[i, 0].set_title("Pre Image")
        axes[i, 0].axis('off')
        
        # Post
        post_img = post[i].permute(1, 2, 0).cpu().numpy()
        post_img = (post_img - post_img.min()) / (post_img.max() - post_img.min() + 1e-8)
        axes[i, 1].imshow(post_img[:, :, :3])
        axes[i, 1].set_title("Post Image")
        axes[i, 1].axis('off')
        
        # Target
        target_mask = target[i].squeeze().cpu().numpy()
        axes[i, 2].imshow(target_mask, cmap='RdYlGn_r')
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis('off')
        
        # Prediction
        pred_mask = pred[i].argmax(dim=0).cpu().numpy()
        axes[i, 3].imshow(pred_mask, cmap='RdYlGn_r')
        axes[i, 3].set_title("Prediction")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ============================================
# 학습 함수
# ============================================
def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False
) -> dict:
    """한 에폭 학습"""
    model.train()
    
    total_loss = 0.0
    all_metrics = {"iou": [], "accuracy": [], "f1": []}
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        pre = batch['pre'].float().to(device)
        post = batch['post'].float().to(device)
        mask = batch['mask'].long().to(device)
        
        # 마스크 차원 처리
        if mask.dim() == 4:
            mask = mask[:, 0]
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(pre, post)
                loss = criterion(outputs, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pre, post)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # 메트릭 계산
        with torch.no_grad():
            metrics = calculate_metrics(outputs, mask)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{metrics['iou']:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics["loss"] = avg_loss
    
    return avg_metrics


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """검증"""
    model.eval()
    
    total_loss = 0.0
    all_metrics = {"iou": [], "accuracy": [], "f1": []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            pre = batch['pre'].float().to(device)
            post = batch['post'].float().to(device)
            mask = batch['mask'].long().to(device)
            
            if mask.dim() == 4:
                mask = mask[:, 0]
            
            outputs = model(pre, post)
            loss = criterion(outputs, mask)
            
            total_loss += loss.item()
            
            metrics = calculate_metrics(outputs, mask)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics["loss"] = avg_loss
    
    return avg_metrics


# ============================================
# 메인 학습 함수
# ============================================
def train(config):
    """
    메인 학습 함수
    
    Args:
        config: Config 객체
    """
    print("=" * 60)
    print("🚀 변화탐지(Change Detection) 학습 시작")
    print("=" * 60)
    
    # 재현성 설정
    seed = config.reproducibility.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 디바이스 설정
    if config.hardware.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.hardware.device)
    print(f"📱 Device: {device}")
    
    # 데이터 준비
    print("\n📂 데이터 로딩...")
    data_dir = Path(config.data.local.root_dir)
    
    if not data_dir.exists():
        print("   ⚠️ 데이터가 없습니다. 더미 데이터 생성 중...")
        create_dummy_data(str(data_dir))
    
    data_module = build_data_module(config)
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    # 모델 생성
    print("\n🧠 모델 생성...")
    model = build_model(config).to(device)
    
    # 손실 함수
    criterion = build_loss(config)
    
    # 옵티마이저
    opt_cfg = config.training.optimizer
    if opt_cfg.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas)
        )
    elif opt_cfg.name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas)
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            momentum=0.9
        )
    
    # 스케줄러
    sched_cfg = config.training.scheduler
    if sched_cfg.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs - sched_cfg.warmup_epochs,
            eta_min=sched_cfg.min_lr
        )
    else:
        scheduler = None
    
    # AMP
    use_amp = config.hardware.mixed_precision and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("   ⚡ Mixed Precision 활성화")
    
    # MLflow 설정
    mlflow.set_tracking_uri(config.logging.mlflow.tracking_uri)
    mlflow.set_experiment(config.experiment.name)
    
    # 체크포인트 디렉토리
    ckpt_dir = Path(config.checkpoint.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 학습 시작
    run_name = config.experiment.run_name or f"cd-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # 파라미터 로깅
        mlflow.log_params({
            "model_architecture": config.model.architecture,
            "encoder": config.model.encoder.name,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.optimizer.lr,
            "optimizer": config.training.optimizer.name,
            "loss": config.training.loss.name,
            "patch_size": config.torchgeo.patch_size,
            "device": str(device)
        })
        
        # 태그 로깅
        if hasattr(config.experiment, 'tags'):
            tags = config.experiment.tags
            mlflow.set_tags({
                "project": tags.project,
                "task": tags.task,
                "environment": tags.environment
            })
        
        best_iou = 0.0
        best_model_path = None
        epochs = config.training.epochs
        
        print(f"\n🏃 학습 시작 ({epochs} epochs)")
        
        for epoch in range(1, epochs + 1):
            print(f"\n📌 Epoch {epoch}/{epochs}")
            
            # 학습
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, use_amp
            )
            
            # 검증
            val_metrics = validate(model, val_loader, criterion, device)
            
            # 스케줄러 스텝
            if scheduler is not None:
                scheduler.step()
            
            # 메트릭 로깅
            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_f1": val_metrics["f1"],
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}")
            
            # 시각화 (첫 에폭, 중간, 마지막)
            if epoch == 1 or epoch == epochs // 2 or epoch == epochs:
                with tempfile.TemporaryDirectory() as tmpdir:
                    for batch in val_loader:
                        pre = batch['pre'].float().to(device)
                        post = batch['post'].float().to(device)
                        mask = batch['mask']
                        
                        with torch.no_grad():
                            pred = model(pre, post)
                        
                        viz_path = os.path.join(tmpdir, f"predictions_epoch_{epoch:03d}.png")
                        visualize_predictions(pre, post, pred, mask, viz_path)
                        mlflow.log_artifact(viz_path, artifact_path="visualizations")
                        break
            
            # Best 모델 저장
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                best_model_path = ckpt_dir / f"best_model_epoch_{epoch:03d}_iou_{best_iou:.4f}.pth"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': best_iou,
                    'config': config.to_dict()
                }, best_model_path)
                
                print(f"   ✅ Best 모델 저장: {best_model_path.name}")
        
        # 최종 모델 저장 및 아티팩트 업로드
        print("\n📦 모델 저장 중...")
        
        # 최종 모델 체크포인트 저장
        final_model_path = ckpt_dir / "final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'best_val_iou': best_iou
        }, final_model_path)
        print(f"   ✅ 최종 모델 저장: {final_model_path}")
        
        # 최종 메트릭 로깅
        mlflow.log_metrics({
            "best_val_iou": best_iou
        })
        
        # 모델 체크포인트 아티팩트로 저장
        if best_model_path and best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(final_model_path), artifact_path="checkpoints")
        print("   ✅ 모델 아티팩트 업로드 완료 (MinIO)")
        
        print("\n" + "=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)
        print(f"   📊 Best Val IoU: {best_iou:.4f}")
        print(f"   📦 Checkpoints: {ckpt_dir}")
        print(f"   🌐 MLflow UI: {config.logging.mlflow.tracking_uri}")


def main():
    """메인 함수"""
    args, config = parse_args_with_config()
    train(config)


if __name__ == "__main__":
    main()
