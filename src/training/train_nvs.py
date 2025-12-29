#!/usr/bin/env python3
"""
Novel View Synthesis (3D Gaussian Splatting) 학습 스크립트
gsplat 라이브러리 기반 + MLflow 로깅 + MinIO 데이터/아티팩트 관리

사용법:
    python src/training/train_nvs.py --config configs/train_nvs.yaml
    python src/training/train_nvs.py -c configs/train_nvs.yaml -o training.iterations=7000
    
Docker:
    docker-compose run nvs-train --config configs/train_nvs.yaml
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env 로드
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ 환경변수 로드: {env_file}")

from src.utils.config import parse_args_with_config, load_config
from src.models.gaussian_model import GaussianModel, GaussianModelConfig, build_gaussian_model

# gsplat 가용성 체크
try:
    import gsplat
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("⚠️ gsplat 미설치. Mock 모드로 실행됩니다.")


# ============================================
# VRAM 모니터링
# ============================================
def get_gpu_memory_usage() -> dict:
    """GPU VRAM 사용량 조회"""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0, "utilization_percent": 0}
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    utilization = (allocated / total) * 100 if total > 0 else 0
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
        "utilization_percent": utilization,
        "total_mb": total
    }


def log_gpu_metrics(step: int):
    """GPU 메트릭을 MLflow에 로깅"""
    mem = get_gpu_memory_usage()
    mlflow.log_metrics({
        "gpu/vram_allocated_mb": mem["allocated_mb"],
        "gpu/vram_reserved_mb": mem["reserved_mb"],
        "gpu/vram_utilization_percent": mem["utilization_percent"]
    }, step=step)


# ============================================
# 카메라 & 이미지 로딩
# ============================================
class Camera:
    """단일 카메라 뷰"""
    def __init__(
        self,
        R: np.ndarray,          # [3, 3] 회전 행렬
        T: np.ndarray,          # [3] 변환 벡터
        FoVx: float,            # 수평 FOV (radians)
        FoVy: float,            # 수직 FOV (radians)
        image: np.ndarray,      # [H, W, 3] 이미지
        image_name: str,
        width: int,
        height: int,
        device: str = "cuda"
    ):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.width = width
        self.height = height
        self.device = device
        
        # 이미지 텐서
        self.original_image = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
        
        # World-to-Camera 변환
        self.view_matrix = self._get_view_matrix()
        self.projection_matrix = self._get_projection_matrix()
        self.full_proj_transform = self.projection_matrix @ self.view_matrix
        self.camera_center = self._get_camera_center()
    
    def _get_view_matrix(self) -> torch.Tensor:
        """World-to-Camera 변환 행렬"""
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R.T
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        return torch.tensor(Rt, dtype=torch.float32, device=self.device)
    
    def _get_projection_matrix(self, znear: float = 0.01, zfar: float = 100.0) -> torch.Tensor:
        """Projection 행렬"""
        tan_half_fov_y = np.tan(self.FoVy / 2)
        tan_half_fov_x = np.tan(self.FoVx / 2)
        
        top = tan_half_fov_y * znear
        bottom = -top
        right = tan_half_fov_x * znear
        left = -right
        
        P = torch.zeros(4, 4, device=self.device)
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
        P[3, 2] = -1.0
        
        return P
    
    def _get_camera_center(self) -> torch.Tensor:
        """카메라 월드 좌표"""
        return torch.tensor(-self.R.T @ self.T, dtype=torch.float32, device=self.device)


class SceneDataset:
    """COLMAP 데이터셋 로더"""
    
    def __init__(self, source_path: Path, resolution: int = -1, device: str = "cuda"):
        self.source_path = Path(source_path)
        self.resolution = resolution
        self.device = device
        
        self.cameras: List[Camera] = []
        self.point_cloud: Optional[np.ndarray] = None
        self.point_colors: Optional[np.ndarray] = None
        
        self._load_colmap()
    
    def _load_colmap(self):
        """COLMAP 데이터 로드"""
        images_dir = self.source_path / "images"
        sparse_dir = self.source_path / "sparse" / "0"
        
        if not sparse_dir.exists():
            sparse_dir = self.source_path / "sparse"
        
        # 이미지 로드
        image_files = sorted(list(images_dir.glob("*.jpg")) + 
                           list(images_dir.glob("*.png")) +
                           list(images_dir.glob("*.JPG")) +
                           list(images_dir.glob("*.PNG")))
        
        if not image_files:
            print(f"⚠️ 이미지 없음: {images_dir}")
            self._create_dummy_data()
            return
        
        # COLMAP 파일 체크
        cameras_bin = sparse_dir / "cameras.bin"
        images_bin = sparse_dir / "images.bin"
        points3d_bin = sparse_dir / "points3D.bin"
        
        if cameras_bin.exists() and images_bin.exists():
            self._load_colmap_binary(cameras_bin, images_bin, points3d_bin, images_dir)
        else:
            print(f"⚠️ COLMAP 파일 없음. 더미 카메라 사용")
            self._create_dummy_cameras(image_files)
    
    def _load_colmap_binary(self, cameras_bin, images_bin, points3d_bin, images_dir):
        """COLMAP Binary 파일 로드"""
        try:
            import pycolmap
            
            reconstruction = pycolmap.Reconstruction()
            reconstruction.read_binary(str(cameras_bin.parent))
            
            # 카메라 로드
            for img_id, img in reconstruction.images.items():
                cam = reconstruction.cameras[img.camera_id]
                
                # 이미지 로드
                img_path = images_dir / img.name
                if not img_path.exists():
                    continue
                
                image = np.array(Image.open(img_path).convert("RGB"))
                height, width = image.shape[:2]
                
                # FOV 계산
                if cam.model_name in ["SIMPLE_PINHOLE", "PINHOLE"]:
                    fx = cam.params[0]
                    fy = cam.params[1] if len(cam.params) > 1 else fx
                else:
                    fx = fy = cam.params[0]
                
                FoVx = 2 * np.arctan(width / (2 * fx))
                FoVy = 2 * np.arctan(height / (2 * fy))
                
                # 카메라 변환
                R = img.rotation_matrix()
                T = img.tvec
                
                self.cameras.append(Camera(
                    R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                    image=image, image_name=img.name,
                    width=width, height=height, device=self.device
                ))
            
            # Point Cloud 로드
            if points3d_bin.exists():
                points = []
                colors = []
                for pt3d_id, pt3d in reconstruction.points3D.items():
                    points.append(pt3d.xyz)
                    colors.append(pt3d.color / 255.0)
                
                if points:
                    self.point_cloud = np.array(points)
                    self.point_colors = np.array(colors)
            
            print(f"   ✅ COLMAP 로드: {len(self.cameras)} 카메라, {len(self.point_cloud) if self.point_cloud is not None else 0} 포인트")
            
        except ImportError:
            print("⚠️ pycolmap 미설치. 더미 카메라 사용")
            image_files = sorted(images_dir.glob("*.[jJpP][pPnN][gG]"))
            self._create_dummy_cameras(image_files)
    
    def _create_dummy_cameras(self, image_files: List[Path]):
        """더미 카메라 생성 (테스트용)"""
        num_cameras = len(image_files)
        
        for i, img_path in enumerate(image_files[:50]):  # 최대 50개
            image = np.array(Image.open(img_path).convert("RGB"))
            height, width = image.shape[:2]
            
            # 구면 배치
            angle = 2 * np.pi * i / num_cameras
            radius = 3.0
            
            # 카메라 위치
            cam_pos = np.array([
                radius * np.cos(angle),
                0.5,
                radius * np.sin(angle)
            ])
            
            # Look-at 변환
            target = np.array([0, 0, 0])
            up = np.array([0, 1, 0])
            
            forward = target - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            R = np.stack([right, up, -forward], axis=1)
            T = -R.T @ cam_pos
            
            FoVx = np.pi / 3  # 60 degrees
            FoVy = FoVx * height / width
            
            self.cameras.append(Camera(
                R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                image=image, image_name=img_path.name,
                width=width, height=height, device=self.device
            ))
        
        print(f"   ✅ 더미 카메라 생성: {len(self.cameras)} 뷰")
    
    def _create_dummy_data(self):
        """완전한 더미 데이터 생성"""
        height, width = 480, 640
        
        for i in range(8):
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            angle = 2 * np.pi * i / 8
            R = np.eye(3)
            T = np.array([np.cos(angle) * 3, 0.5, np.sin(angle) * 3])
            
            self.cameras.append(Camera(
                R=R, T=T, FoVx=np.pi/3, FoVy=np.pi/4,
                image=image, image_name=f"dummy_{i:03d}.jpg",
                width=width, height=height, device=self.device
            ))
        
        # 랜덤 포인트 클라우드
        self.point_cloud = np.random.randn(10000, 3) * 0.5
        self.point_colors = np.random.rand(10000, 3)
        
        print(f"   ⚠️ 더미 데이터 생성: {len(self.cameras)} 뷰")
    
    def __len__(self):
        return len(self.cameras)
    
    def __getitem__(self, idx):
        return self.cameras[idx]


# ============================================
# Gaussian Splatting 렌더러
# ============================================
class GaussianRenderer:
    """gsplat 기반 렌더러"""
    
    def __init__(self, sh_degree: int = 3, device: str = "cuda"):
        self.sh_degree = sh_degree
        self.device = device
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    def render(
        self,
        camera: Camera,
        gaussians: GaussianModel,
        scaling_modifier: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Gaussian Splatting 렌더링
        
        Returns:
            {"render": [3, H, W], "viewspace_points": ..., "visibility_filter": ..., "radii": ...}
        """
        if not GSPLAT_AVAILABLE:
            return self._mock_render(camera, gaussians)
        
        # Gaussian 속성 가져오기
        means3D = gaussians.xyz
        opacity = gaussians.opacity
        scales = gaussians.scaling * scaling_modifier
        rotations = gaussians.rotation
        shs = gaussians.features
        
        # gsplat rasterization
        try:
            # Viewmat [4, 4]
            viewmat = camera.view_matrix.unsqueeze(0)  # [1, 4, 4]
            
            # K matrix
            fx = camera.width / (2 * np.tan(camera.FoVx / 2))
            fy = camera.height / (2 * np.tan(camera.FoVy / 2))
            cx = camera.width / 2
            cy = camera.height / 2
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=self.device)
            
            # Rasterization
            renders, alphas, meta = rasterization(
                means=means3D,
                quats=rotations,
                scales=scales,
                opacities=opacity.squeeze(-1),
                colors=shs[:, 0, :],  # DC component만 사용 (간소화)
                viewmats=viewmat,
                Ks=K.unsqueeze(0),
                width=camera.width,
                height=camera.height,
                sh_degree=0,  # 간소화: DC만
                backgrounds=self.background.unsqueeze(0),
            )
            
            rendered_image = renders[0].permute(2, 0, 1)  # [3, H, W]
            
            return {
                "render": rendered_image,
                "viewspace_points": means3D,
                "visibility_filter": alphas[0] > 0,
                "radii": meta.get("radii", torch.zeros(len(means3D), device=self.device))
            }
            
        except Exception as e:
            print(f"⚠️ gsplat 렌더링 실패: {e}")
            return self._mock_render(camera, gaussians)
    
    def _mock_render(self, camera: Camera, gaussians: GaussianModel) -> Dict[str, torch.Tensor]:
        """Mock 렌더링 (gsplat 없을 때) - 학습용 differentiable 버전"""
        H, W = camera.height, camera.width
        
        # Gaussian 속성 (gradient 연결 유지)
        means3D = gaussians.xyz
        colors = gaussians.features[:, 0, :]  # DC [N, 3]
        opacities = gaussians.opacity.squeeze(-1)  # [N]
        
        # 전체 Gaussian의 가중 평균 색상 계산 (항상 differentiable)
        # SH to RGB
        rgb_colors = (colors * 0.28209479177387814 + 0.5).clamp(0, 1)  # [N, 3]
        
        # Opacity 가중 평균 (모든 점 사용)
        weights = opacities.unsqueeze(-1)  # [N, 1]
        weighted_color = (rgb_colors * weights).sum(dim=0) / (weights.sum() + 1e-8)  # [3]
        
        # 렌더링 이미지 생성 (전체를 평균색으로 - 단순화된 differentiable 렌더링)
        rendered = weighted_color.view(3, 1, 1).expand(3, H, W).contiguous()
        
        return {
            "render": rendered,
            "viewspace_points": means3D.detach(),
            "visibility_filter": torch.ones(len(means3D), dtype=torch.bool, device=self.device),
            "radii": torch.ones(len(means3D), device=self.device)
        }


# ============================================
# 손실 함수
# ============================================
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural Similarity Loss"""
    try:
        from pytorch_msssim import ssim
        return 1 - ssim(pred.unsqueeze(0), target.unsqueeze(0), data_range=1.0, size_average=True)
    except ImportError:
        # 간단한 대체
        return l1_loss(pred, target)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


# ============================================
# MinIO 데이터 관리
# ============================================
class MinIODataManager:
    """MinIO 데이터 다운로드/업로드 관리"""
    
    def __init__(self, config):
        import boto3
        
        minio_cfg = config.data.minio
        self.endpoint_url = minio_cfg.endpoint
        self.bucket_raw = getattr(minio_cfg, 'bucket_raw', 'raw-data-nvs')
        self.bucket_artifacts = minio_cfg.bucket_artifacts
        
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio_secure_password_2024"),
        )
    
    def download_dataset(self, s3_prefix: str, local_dir: Path) -> Path:
        """MinIO에서 데이터셋 다운로드"""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 데이터 다운로드: s3://{self.bucket_raw}/{s3_prefix}")
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_raw, Prefix=s3_prefix)
        
        objects = []
        for page in pages:
            if 'Contents' in page:
                objects.extend(page['Contents'])
        
        if not objects:
            raise FileNotFoundError(f"데이터를 찾을 수 없습니다: s3://{self.bucket_raw}/{s3_prefix}")
        
        print(f"   📊 총 {len(objects)}개 파일")
        
        for obj in tqdm(objects, desc="다운로드"):
            key = obj['Key']
            rel_path = key[len(s3_prefix):].lstrip('/')
            if not rel_path:
                continue
            
            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_raw, key, str(local_path))
        
        return local_dir
    
    def upload_artifact(self, local_path: Path, s3_key: str, bucket: Optional[str] = None):
        """아티팩트 업로드"""
        bucket = bucket or self.bucket_artifacts
        self.s3_client.upload_file(str(local_path), bucket, s3_key)
        print(f"   📤 업로드: s3://{bucket}/{s3_key}")


# ============================================
# 메인 학습 함수
# ============================================
def train(config):
    """NVS (3D Gaussian Splatting) 학습"""
    print("=" * 60)
    print("🎬 Novel View Synthesis (3D Gaussian Splatting) 학습")
    print("=" * 60)
    
    # gsplat 상태
    if GSPLAT_AVAILABLE:
        print(f"✅ gsplat 버전: {gsplat.__version__}")
    else:
        print("⚠️ gsplat 미설치 - Mock 렌더러 사용")
    
    # 재현성
    seed = config.reproducibility.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 디바이스
    if config.hardware.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.hardware.device
    print(f"📱 Device: {device}")
    
    if torch.cuda.is_available():
        mem = get_gpu_memory_usage()
        print(f"   💾 VRAM: {mem['total_mb']:.0f} MB 총")
        torch.cuda.reset_peak_memory_stats()
    
    # 데이터 준비
    print("\n📂 데이터 로딩...")
    source_path = Path(config.data.source_path)
    
    if not source_path.exists():
        try:
            data_manager = MinIODataManager(config)
            s3_prefix = str(source_path).replace("./data/", "").replace("data/", "")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                source_path = data_manager.download_dataset(s3_prefix, Path(tmpdir) / "data")
        except Exception as e:
            print(f"   ⚠️ MinIO 다운로드 실패: {e}")
    
    # Scene 로드
    scene = SceneDataset(source_path, device=device)
    print(f"   📸 카메라: {len(scene)} 뷰")
    
    # 출력 디렉토리
    output_dir = Path(config.pipeline.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = output_dir / "renders"
    render_dir.mkdir(exist_ok=True)
    
    # Gaussian 모델 생성
    print("\n🧠 Gaussian 모델 초기화...")
    gaussians = build_gaussian_model(config)
    
    if scene.point_cloud is not None:
        gaussians.init_from_pcd(scene.point_cloud, scene.point_colors, device)
    else:
        gaussians.init_random(config.model.init.num_points or 10000, device=device)
    
    # 렌더러
    renderer = GaussianRenderer(sh_degree=config.model.sh_degree, device=device)
    
    # 옵티마이저
    optimizer = torch.optim.Adam(gaussians.get_param_groups(), lr=0.0, eps=1e-15)
    
    # 학습률 스케줄러
    def get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps):
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                return 0.0
            if lr_delay_steps > 0:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp
        return helper
    
    lr_func = get_expon_lr_func(
        config.training.learning_rate.position_lr_init,
        config.training.learning_rate.position_lr_final,
        0, 1.0,
        config.training.learning_rate.position_lr_max_steps
    )
    
    # MLflow 설정
    mlflow.set_tracking_uri(config.logging.mlflow.tracking_uri)
    mlflow.set_experiment(config.experiment.name)
    
    run_name = config.experiment.run_name or f"gs-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # 파라미터 로깅
        mlflow.log_params({
            "sh_degree": config.model.sh_degree,
            "iterations": config.training.iterations,
            "lambda_dssim": config.training.loss.lambda_dssim,
            "densify_interval": config.training.densification.interval,
            "device": device,
            "gsplat_available": GSPLAT_AVAILABLE,
        })
        
        # 태그
        tags = config.experiment.tags
        mlflow.set_tags({
            "project": tags.project,
            "task": tags.task,
            "method": tags.method,
        })
        
        total_iterations = config.training.iterations
        lambda_dssim = config.training.loss.lambda_dssim
        
        densify_start = config.training.densification.start_iteration
        densify_end = config.training.densification.end_iteration
        densify_interval = config.training.densification.interval
        
        print(f"\n🚀 학습 시작 ({total_iterations} iterations)")
        
        pbar = tqdm(range(1, total_iterations + 1), desc="Training")
        
        for iteration in pbar:
            # 학습률 업데이트
            for param_group in optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group["lr"] = lr_func(iteration)
            
            # 랜덤 카메라 선택
            cam_idx = np.random.randint(0, len(scene))
            camera = scene[cam_idx]
            
            # 렌더링
            render_output = renderer.render(camera, gaussians)
            rendered_image = render_output["render"]
            
            # 손실 계산
            gt_image = camera.original_image
            
            Ll1 = l1_loss(rendered_image, gt_image)
            Lssim = ssim_loss(rendered_image, gt_image)
            
            loss = (1 - lambda_dssim) * Ll1 + lambda_dssim * Lssim
            
            # 역전파
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Densification
            if densify_start <= iteration <= densify_end and iteration % densify_interval == 0:
                gaussians.densify_and_prune(
                    grad_threshold=config.training.densification.grad_threshold,
                    opacity_threshold=config.training.densification.opacity_threshold,
                    scale_threshold=config.training.densification.scale_threshold,
                    xyz_grad=gaussians._xyz.grad if gaussians._xyz.grad is not None else None
                )
                # 옵티마이저 재설정
                optimizer = torch.optim.Adam(gaussians.get_param_groups(), lr=0.0, eps=1e-15)
            
            # Opacity Reset
            if iteration % config.training.opacity_reset_interval == 0:
                gaussians.reset_opacity()
            
            # 로깅
            if iteration % config.logging.log_interval == 0:
                current_psnr = psnr(rendered_image, gt_image)
                
                mlflow.log_metrics({
                    "loss": loss.item(),
                    "l1_loss": Ll1.item(),
                    "psnr": current_psnr,
                    "num_gaussians": gaussians.num_gaussians,
                }, step=iteration)
                
                log_gpu_metrics(iteration)
                
                pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{current_psnr:.2f}", gaussians=gaussians.num_gaussians)
            
            # 렌더링 저장 (1000 iteration마다)
            if iteration % 1000 == 0 or iteration in config.training.test_iterations:
                with torch.no_grad():
                    # 첫 번째 카메라로 렌더링
                    test_camera = scene[0]
                    test_render = renderer.render(test_camera, gaussians)["render"]
                    
                    # 이미지 저장
                    render_np = (test_render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    render_path = render_dir / f"render_iter_{iteration:06d}.png"
                    cv2.imwrite(str(render_path), cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR))
                    
                    mlflow.log_artifact(str(render_path), artifact_path="renders")
                    
                    tqdm.write(f"   📸 Iter {iteration}: PSNR={psnr(test_render, test_camera.original_image):.2f}, Gaussians={gaussians.num_gaussians}")
        
        # 최종 저장
        print("\n💾 최종 결과 저장...")
        
        # Point Cloud PLY 저장
        ply_path = output_dir / "point_cloud.ply"
        gaussians.save_ply(ply_path)
        mlflow.log_artifact(str(ply_path), artifact_path="model")
        
        # Config 저장
        config_path = output_dir / "config.yaml"
        config.save(config_path)
        mlflow.log_artifact(str(config_path), artifact_path="config")
        
        # 최종 메트릭
        with torch.no_grad():
            test_camera = scene[0]
            final_render = renderer.render(test_camera, gaussians)["render"]
            final_psnr = psnr(final_render, test_camera.original_image)
        
        mlflow.log_metrics({
            "final_psnr": final_psnr,
            "final_num_gaussians": gaussians.num_gaussians,
        })
        
        # GPU 최대 사용량
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            mlflow.log_metric("gpu/max_vram_mb", max_mem)
            print(f"   💾 최대 VRAM 사용량: {max_mem:.0f} MB")
        
        # MinIO 업로드
        try:
            data_manager = MinIODataManager(config)
            s3_key = f"nvs/{run_name}/point_cloud.ply"
            data_manager.upload_artifact(ply_path, s3_key)
        except Exception as e:
            print(f"   ⚠️ MinIO 업로드 실패: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)
        print(f"   📊 Final PSNR: {final_psnr:.2f} dB")
        print(f"   🎯 Final Gaussians: {gaussians.num_gaussians}")
        print(f"   💾 Point Cloud: {ply_path}")
        print(f"   🌐 MLflow UI: {config.logging.mlflow.tracking_uri}")


def main():
    args, config = parse_args_with_config()
    train(config)


if __name__ == "__main__":
    main()
