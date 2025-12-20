"""
시각화 유틸리티 모듈
변화탐지 결과 시각화, 오버레이 생성 등
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None


def normalize_image(img: np.ndarray) -> np.ndarray:
    """이미지를 0-1 범위로 정규화"""
    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img


def create_change_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    change_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    이미지에 변화 마스크 오버레이 생성
    
    Args:
        image: 원본 이미지 [H, W, C] 또는 [C, H, W]
        mask: 변화 마스크 [H, W] (1=변화, 0=비변화)
        alpha: 오버레이 투명도
        change_color: 변화 영역 색상 (R, G, B)
    
    Returns:
        오버레이 이미지 [H, W, 3]
    """
    # 차원 정리
    if image.ndim == 3 and image.shape[0] <= 4:
        image = np.transpose(image, (1, 2, 0))
    
    # 정규화
    image = normalize_image(image)
    
    # RGB로 변환
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[:, :, :3]
    
    # 8비트로 변환
    image = (image * 255).astype(np.uint8)
    
    # 마스크 처리
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = mask.astype(bool)
    
    # 오버레이 생성
    overlay = image.copy()
    change_color_arr = np.array(change_color, dtype=np.uint8)
    
    overlay[mask] = (
        (1 - alpha) * overlay[mask] + alpha * change_color_arr
    ).astype(np.uint8)
    
    return overlay


def visualize_change_detection(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    title: Optional[str] = None
) -> plt.Figure:
    """
    변화탐지 결과 시각화
    
    Args:
        pre_image: 변화 전 이미지 [H, W, C] 또는 [C, H, W]
        post_image: 변화 후 이미지
        prediction: 예측 마스크 [H, W]
        ground_truth: 정답 마스크 (옵션)
        save_path: 저장 경로
        figsize: 그림 크기
        title: 제목
    
    Returns:
        matplotlib Figure
    """
    # 차원 정리
    if pre_image.ndim == 3 and pre_image.shape[0] <= 4:
        pre_image = np.transpose(pre_image, (1, 2, 0))
    if post_image.ndim == 3 and post_image.shape[0] <= 4:
        post_image = np.transpose(post_image, (1, 2, 0))
    
    # 정규화
    pre_image = normalize_image(pre_image)
    post_image = normalize_image(post_image)
    
    # 서브플롯 수 결정
    num_cols = 5 if ground_truth is not None else 4
    
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    
    # Pre Image
    axes[0].imshow(pre_image[:, :, :3] if pre_image.shape[-1] >= 3 else pre_image)
    axes[0].set_title("Pre Image")
    axes[0].axis('off')
    
    # Post Image
    axes[1].imshow(post_image[:, :, :3] if post_image.shape[-1] >= 3 else post_image)
    axes[1].set_title("Post Image")
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    # Overlay
    overlay = create_change_overlay(post_image, prediction)
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay")
    axes[3].axis('off')
    
    # Ground Truth (옵션)
    if ground_truth is not None:
        axes[4].imshow(ground_truth, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[4].set_title("Ground Truth")
        axes[4].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 시각화 저장: {save_path}")
    
    return fig


def save_prediction_geotiff(
    prediction: np.ndarray,
    output_path: str,
    reference_path: Optional[str] = None,
    crs: str = "EPSG:4326",
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> str:
    """
    예측 결과를 GeoTIFF로 저장
    
    Args:
        prediction: 예측 마스크 [H, W]
        output_path: 출력 경로
        reference_path: 참조 GeoTIFF (CRS, Transform 복사)
        crs: 좌표계
        bounds: 경계 (west, south, east, north)
    
    Returns:
        저장된 파일 경로
    """
    if rasterio is None:
        raise ImportError("rasterio가 필요합니다: pip install rasterio")
    
    # 차원 정리
    if prediction.ndim == 2:
        prediction = prediction[np.newaxis, ...]
    
    height, width = prediction.shape[-2:]
    
    # 참조 파일에서 메타데이터 복사
    if reference_path and os.path.exists(reference_path):
        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
            profile.update(
                count=1,
                dtype=prediction.dtype
            )
    else:
        # 기본 프로파일
        if bounds is None:
            bounds = (0, 0, width, height)
        
        transform = from_bounds(*bounds, width, height)
        profile = {
            'driver': 'GTiff',
            'dtype': prediction.dtype,
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
    
    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(str(output_path), 'w', **profile) as dst:
        dst.write(prediction)
    
    print(f"✅ GeoTIFF 저장: {output_path}")
    return str(output_path)


def compare_predictions(
    image: np.ndarray,
    predictions: dict,  # {"model1": pred1, "model2": pred2, ...}
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = None
) -> plt.Figure:
    """
    여러 모델의 예측 결과 비교
    
    Args:
        image: 입력 이미지
        predictions: 모델별 예측 딕셔너리
        ground_truth: 정답 마스크
        save_path: 저장 경로
        figsize: 그림 크기
    """
    num_preds = len(predictions)
    num_cols = num_preds + 2 if ground_truth is not None else num_preds + 1
    
    if figsize is None:
        figsize = (4 * num_cols, 4)
    
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    
    # 차원 정리
    if image.ndim == 3 and image.shape[0] <= 4:
        image = np.transpose(image, (1, 2, 0))
    image = normalize_image(image)
    
    # 원본 이미지
    axes[0].imshow(image[:, :, :3] if image.shape[-1] >= 3 else image)
    axes[0].set_title("Input")
    axes[0].axis('off')
    
    # 각 모델 예측
    for i, (name, pred) in enumerate(predictions.items()):
        axes[i + 1].imshow(pred, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')
    
    # Ground Truth
    if ground_truth is not None:
        axes[-1].imshow(ground_truth, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[-1].set_title("Ground Truth")
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # 테스트
    print("=== Visualization 테스트 ===")
    
    # 더미 데이터
    pre = np.random.rand(256, 256, 3).astype(np.float32)
    post = np.random.rand(256, 256, 3).astype(np.float32)
    pred = (np.random.rand(256, 256) > 0.7).astype(np.uint8)
    
    # 시각화
    fig = visualize_change_detection(pre, post, pred, save_path="/tmp/test_viz.png")
    plt.close(fig)
    
    print("✅ 테스트 완료")
