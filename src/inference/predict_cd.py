#!/usr/bin/env python3
"""
변화탐지 추론(Inference) 스크립트
학습된 모델을 불러와 새로운 이미지에 대해 변화탐지 수행

사용법:
    # Run ID로 모델 로드
    python src/inference/predict_cd.py --run-id abc123 --pre pre.tif --post post.tif -o output/

    # 체크포인트 파일 직접 지정
    python src/inference/predict_cd.py --checkpoint checkpoints/best_model.pth --pre pre.tif --post post.tif
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env 로드
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)

from src.models.unet import ChangeDetectionModel
from src.utils.visualization import (
    visualize_change_detection,
    save_prediction_geotiff,
    create_change_overlay
)


class ChangeDetectionPredictor:
    """
    변화탐지 추론 클래스
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        run_id: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            run_id: MLflow Run ID
            device: 디바이스 (auto, cuda, cpu)
        """
        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"📱 Device: {self.device}")
        
        # 모델 로드
        self.model = None
        self.config = None
        
        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)
        elif run_id:
            self._load_from_mlflow(run_id)
        else:
            raise ValueError("checkpoint_path 또는 run_id 중 하나를 지정해야 합니다.")
    
    def _load_from_checkpoint(self, checkpoint_path: str):
        """체크포인트 파일에서 모델 로드"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {checkpoint_path}")
        
        print(f"📦 체크포인트 로드: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint.get('config', {})
        
        # 모델 생성
        model_cfg = self.config.get('model', {})
        self.model = ChangeDetectionModel(
            architecture=model_cfg.get('architecture', 'unet'),
            encoder_name=model_cfg.get('encoder', {}).get('name', 'resnet50'),
            encoder_weights=None,  # 가중치는 체크포인트에서 로드
            in_channels=model_cfg.get('in_channels', 6),
            num_classes=model_cfg.get('num_classes', 2)
        )
        
        # 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"   ✅ 모델 로드 완료")
        if 'val_iou' in checkpoint:
            print(f"   📊 Checkpoint IoU: {checkpoint['val_iou']:.4f}")
    
    def _load_from_mlflow(self, run_id: str):
        """MLflow Run에서 모델 로드"""
        import mlflow
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"📡 MLflow: {tracking_uri}")
        print(f"🔍 Run ID: {run_id}")
        
        client = mlflow.tracking.MlflowClient()
        
        # Run 정보 조회
        run = client.get_run(run_id)
        print(f"   📌 Run Name: {run.info.run_name}")
        
        # 아티팩트 다운로드
        artifact_path = client.download_artifacts(run_id, "checkpoints")
        print(f"   📥 아티팩트 다운로드: {artifact_path}")
        
        # 체크포인트 파일 찾기
        artifact_dir = Path(artifact_path)
        checkpoint_files = list(artifact_dir.glob("*.pth"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"Run {run_id}에서 체크포인트를 찾을 수 없습니다.")
        
        # best_model 또는 final_model 우선
        checkpoint_path = None
        for pattern in ["best_model*.pth", "final_model.pth"]:
            matches = list(artifact_dir.glob(pattern))
            if matches:
                checkpoint_path = matches[0]
                break
        
        if checkpoint_path is None:
            checkpoint_path = checkpoint_files[0]
        
        self._load_from_checkpoint(str(checkpoint_path))
    
    def load_geotiff(self, path: str) -> Tuple[np.ndarray, dict]:
        """
        GeoTIFF 이미지 로드
        
        Returns:
            (이미지 배열 [C, H, W], 메타데이터)
        """
        with rasterio.open(path) as src:
            image = src.read()
            meta = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds
            }
        return image, meta
    
    def predict_single(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        patch_size: int = 256,
        overlap: int = 32
    ) -> np.ndarray:
        """
        단일 이미지 쌍에 대한 예측
        
        Args:
            pre_image: Pre 이미지 [C, H, W]
            post_image: Post 이미지 [C, H, W]
            patch_size: 패치 크기
            overlap: 오버랩 크기
        
        Returns:
            예측 마스크 [H, W]
        """
        _, h, w = pre_image.shape
        
        # 작은 이미지는 한 번에 처리
        if h <= patch_size and w <= patch_size:
            return self._predict_patch(pre_image, post_image)
        
        # 큰 이미지는 슬라이딩 윈도우로 처리
        return self._predict_sliding_window(
            pre_image, post_image, patch_size, overlap
        )
    
    def _predict_patch(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray
    ) -> np.ndarray:
        """패치 단위 예측"""
        # 텐서로 변환
        pre_tensor = torch.from_numpy(pre_image).float().unsqueeze(0).to(self.device)
        post_tensor = torch.from_numpy(post_image).float().unsqueeze(0).to(self.device)
        
        # 정규화 (0-255 -> 0-1)
        if pre_tensor.max() > 1.0:
            pre_tensor = pre_tensor / 255.0
        if post_tensor.max() > 1.0:
            post_tensor = post_tensor / 255.0
        
        # 추론
        with torch.no_grad():
            outputs = self.model(pre_tensor, post_tensor)
            pred = outputs.argmax(dim=1).squeeze().cpu().numpy()
        
        return pred
    
    def _predict_sliding_window(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        patch_size: int,
        overlap: int
    ) -> np.ndarray:
        """슬라이딩 윈도우 예측 (대용량 이미지용)"""
        _, h, w = pre_image.shape
        stride = patch_size - overlap
        
        # 결과 및 카운트 배열
        prediction = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        # 슬라이딩 윈도우
        y_positions = list(range(0, h - patch_size + 1, stride))
        x_positions = list(range(0, w - patch_size + 1, stride))
        
        # 마지막 패치가 누락되지 않도록
        if y_positions[-1] + patch_size < h:
            y_positions.append(h - patch_size)
        if x_positions[-1] + patch_size < w:
            x_positions.append(w - patch_size)
        
        from tqdm import tqdm
        total = len(y_positions) * len(x_positions)
        
        with tqdm(total=total, desc="Predicting") as pbar:
            for y in y_positions:
                for x in x_positions:
                    # 패치 추출
                    pre_patch = pre_image[:, y:y+patch_size, x:x+patch_size]
                    post_patch = post_image[:, y:y+patch_size, x:x+patch_size]
                    
                    # 예측
                    pred_patch = self._predict_patch(pre_patch, post_patch)
                    
                    # 누적
                    prediction[y:y+patch_size, x:x+patch_size] += pred_patch
                    count[y:y+patch_size, x:x+patch_size] += 1
                    
                    pbar.update(1)
        
        # 평균
        prediction = prediction / np.maximum(count, 1)
        
        # 임계값 적용
        prediction = (prediction > 0.5).astype(np.uint8)
        
        return prediction
    
    def predict_files(
        self,
        pre_path: str,
        post_path: str,
        output_dir: str,
        save_geotiff: bool = True,
        save_visualization: bool = True,
        patch_size: int = 256
    ) -> dict:
        """
        파일 기반 예측
        
        Args:
            pre_path: Pre 이미지 경로
            post_path: Post 이미지 경로
            output_dir: 출력 디렉토리
            save_geotiff: GeoTIFF 저장 여부
            save_visualization: 시각화 저장 여부
            patch_size: 패치 크기
        
        Returns:
            결과 딕셔너리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n📂 입력 파일 로드...")
        print(f"   Pre: {pre_path}")
        print(f"   Post: {post_path}")
        
        # 이미지 로드
        pre_image, pre_meta = self.load_geotiff(pre_path)
        post_image, post_meta = self.load_geotiff(post_path)
        
        print(f"   이미지 크기: {pre_image.shape}")
        
        # 예측
        print("\n🔮 변화탐지 수행 중...")
        prediction = self.predict_single(pre_image, post_image, patch_size)
        
        results = {
            "prediction": prediction,
            "pre_meta": pre_meta,
            "post_meta": post_meta
        }
        
        # GeoTIFF 저장
        if save_geotiff:
            geotiff_path = output_dir / "prediction.tif"
            save_prediction_geotiff(
                prediction,
                str(geotiff_path),
                reference_path=pre_path
            )
            results["geotiff_path"] = str(geotiff_path)
        
        # 시각화 저장
        if save_visualization:
            viz_path = output_dir / "visualization.png"
            visualize_change_detection(
                pre_image,
                post_image,
                prediction,
                save_path=str(viz_path)
            )
            results["visualization_path"] = str(viz_path)
        
        # 통계
        change_ratio = prediction.sum() / prediction.size * 100
        print(f"\n📊 변화 영역: {change_ratio:.2f}%")
        results["change_ratio"] = change_ratio
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="변화탐지 추론",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 모델 소스
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--run-id", "-r",
        type=str,
        help="MLflow Run ID"
    )
    model_group.add_argument(
        "--checkpoint", "-c",
        type=str,
        help="체크포인트 파일 경로"
    )
    
    # 입력
    parser.add_argument(
        "--pre", "-p",
        type=str,
        required=True,
        help="Pre 이미지 경로 (GeoTIFF)"
    )
    parser.add_argument(
        "--post", "-t",
        type=str,
        required=True,
        help="Post 이미지 경로 (GeoTIFF)"
    )
    
    # 출력
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="출력 디렉토리"
    )
    
    # 옵션
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="패치 크기 (기본값: 256)"
    )
    parser.add_argument(
        "--no-geotiff",
        action="store_true",
        help="GeoTIFF 저장 안 함"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="시각화 저장 안 함"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="디바이스 (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔮 변화탐지 추론 (Inference)")
    print("=" * 60)
    
    # Predictor 생성
    predictor = ChangeDetectionPredictor(
        checkpoint_path=args.checkpoint,
        run_id=args.run_id,
        device=args.device
    )
    
    # 예측
    results = predictor.predict_files(
        pre_path=args.pre,
        post_path=args.post,
        output_dir=args.output,
        save_geotiff=not args.no_geotiff,
        save_visualization=not args.no_viz,
        patch_size=args.patch_size
    )
    
    print("\n" + "=" * 60)
    print("✅ 추론 완료!")
    print("=" * 60)
    print(f"   📂 출력 디렉토리: {args.output}")
    
    if "geotiff_path" in results:
        print(f"   📄 GeoTIFF: {results['geotiff_path']}")
    if "visualization_path" in results:
        print(f"   🖼️  시각화: {results['visualization_path']}")


if __name__ == "__main__":
    main()
