"""
3D Gaussian Splatting 모델 정의
gsplat 라이브러리 기반 Gaussian 모델

참고: https://github.com/nerfstudio-project/gsplat
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import gsplat
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("⚠️ gsplat이 설치되지 않았습니다. pip install gsplat")


@dataclass
class GaussianModelConfig:
    """Gaussian 모델 설정"""
    sh_degree: int = 3
    num_points: int = 100000
    spatial_lr_scale: float = 1.0
    
    # 초기 값
    opacity_init: float = 0.1
    scale_init: float = 1.0
    
    # 학습률
    position_lr: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001


class GaussianModel(nn.Module):
    """
    3D Gaussian Splatting 모델
    
    각 Gaussian은 다음 속성을 가짐:
    - position (3D): xyz 좌표
    - features (SH): Spherical Harmonics 계수 (뷰 의존 색상)
    - opacity: 투명도
    - scale (3D): 각 축 스케일
    - rotation (quaternion): 회전
    """
    
    def __init__(self, config: GaussianModelConfig):
        super().__init__()
        self.config = config
        self.sh_degree = config.sh_degree
        self.max_sh_degree = config.sh_degree
        
        # Gaussian 속성 (학습 가능 파라미터)
        self._xyz = None           # [N, 3] 위치
        self._features_dc = None   # [N, 1, 3] DC 성분
        self._features_rest = None # [N, (sh_deg+1)^2-1, 3] 나머지 SH 성분
        self._opacity = None       # [N, 1] opacity (sigmoid 전)
        self._scaling = None       # [N, 3] scale (log space)
        self._rotation = None      # [N, 4] quaternion
        
        self.num_gaussians = 0
        self.spatial_lr_scale = config.spatial_lr_scale
        
    def init_from_pcd(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        device: str = "cuda"
    ):
        """
        Point Cloud에서 Gaussian 초기화
        
        Args:
            points: [N, 3] 3D 좌표
            colors: [N, 3] RGB 색상 (0-1)
            device: 디바이스
        """
        self.num_gaussians = len(points)
        
        # 위치
        self._xyz = nn.Parameter(
            torch.tensor(points, dtype=torch.float32, device=device)
        )
        
        # SH 특성 (색상)
        if colors is None:
            colors = np.random.rand(self.num_gaussians, 3) * 0.5 + 0.25
        
        # RGB to SH DC component
        fused_color = self._rgb_to_sh(
            torch.tensor(colors, dtype=torch.float32, device=device)
        )
        
        self._features_dc = nn.Parameter(
            fused_color.unsqueeze(1)  # [N, 1, 3]
        )
        
        # 나머지 SH 계수 (0으로 초기화)
        num_sh_rest = (self.max_sh_degree + 1) ** 2 - 1
        self._features_rest = nn.Parameter(
            torch.zeros(self.num_gaussians, num_sh_rest, 3, device=device)
        )
        
        # Opacity
        self._opacity = nn.Parameter(
            torch.logit(torch.ones(self.num_gaussians, 1, device=device) * self.config.opacity_init)
        )
        
        # Scale (점 간 거리 기반 초기화)
        dist = self._compute_nearest_neighbor_dist(points)
        scales = np.log(np.clip(dist * self.config.scale_init, 1e-7, 1e7))
        self._scaling = nn.Parameter(
            torch.tensor(np.tile(scales[:, None], (1, 3)), dtype=torch.float32, device=device)
        )
        
        # Rotation (identity quaternion)
        self._rotation = nn.Parameter(
            torch.tensor([[1, 0, 0, 0]] * self.num_gaussians, dtype=torch.float32, device=device)
        )
        
        print(f"✅ Gaussian 모델 초기화: {self.num_gaussians} points")
    
    def init_random(self, num_points: int, spatial_extent: float = 1.0, device: str = "cuda"):
        """랜덤 초기화"""
        points = np.random.randn(num_points, 3) * spatial_extent
        colors = np.random.rand(num_points, 3)
        self.init_from_pcd(points, colors, device)
    
    def _rgb_to_sh(self, rgb: torch.Tensor) -> torch.Tensor:
        """RGB를 SH DC 계수로 변환"""
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
    
    def _sh_to_rgb(self, sh: torch.Tensor) -> torch.Tensor:
        """SH DC 계수를 RGB로 변환"""
        C0 = 0.28209479177387814
        return sh * C0 + 0.5
    
    def _compute_nearest_neighbor_dist(self, points: np.ndarray, k: int = 3) -> np.ndarray:
        """최근접 이웃 거리 계산"""
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k+1)
        return np.mean(dists[:, 1:], axis=1)
    
    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz
    
    @property
    def features(self) -> torch.Tensor:
        """전체 SH 특성 [N, (sh_deg+1)^2, 3]"""
        return torch.cat([self._features_dc, self._features_rest], dim=1)
    
    @property
    def opacity(self) -> torch.Tensor:
        """Opacity [N, 1] (sigmoid 적용)"""
        return torch.sigmoid(self._opacity)
    
    @property
    def scaling(self) -> torch.Tensor:
        """Scale [N, 3] (exp 적용)"""
        return torch.exp(self._scaling)
    
    @property
    def rotation(self) -> torch.Tensor:
        """정규화된 Quaternion [N, 4]"""
        return torch.nn.functional.normalize(self._rotation, dim=1)
    
    @property
    def covariance(self) -> torch.Tensor:
        """3D Covariance 행렬 계산"""
        # Scale을 대각 행렬로
        S = torch.diag_embed(self.scaling)
        
        # Quaternion을 회전 행렬로
        R = self._quaternion_to_rotation_matrix(self.rotation)
        
        # Covariance = R @ S @ S.T @ R.T
        RS = torch.bmm(R, S)
        cov = torch.bmm(RS, RS.transpose(1, 2))
        
        return cov
    
    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Quaternion [N, 4]를 회전 행렬 [N, 3, 3]으로 변환"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=1).reshape(-1, 3, 3)
        
        return R
    
    def get_param_groups(self) -> list:
        """옵티마이저용 파라미터 그룹"""
        return [
            {"params": [self._xyz], "lr": self.config.position_lr * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": self.config.feature_lr, "name": "features_dc"},
            {"params": [self._features_rest], "lr": self.config.feature_lr / 20.0, "name": "features_rest"},
            {"params": [self._opacity], "lr": self.config.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": self.config.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": self.config.rotation_lr, "name": "rotation"},
        ]
    
    def densify_and_prune(
        self,
        grad_threshold: float,
        opacity_threshold: float,
        scale_threshold: float,
        max_screen_size: float = 20.0,
        xyz_grad: Optional[torch.Tensor] = None
    ):
        """
        Densification & Pruning
        
        - 높은 gradient를 가진 포인트: Clone 또는 Split
        - 낮은 opacity 포인트: Prune
        - 너무 큰 포인트: Prune
        """
        if xyz_grad is None:
            return
        
        # Gradient 크기
        grad_mag = xyz_grad.norm(dim=1)
        
        # Clone (작은 포인트 복제)
        mask_clone = grad_mag > grad_threshold
        mask_clone &= self.scaling.max(dim=1).values < scale_threshold
        
        # Split (큰 포인트 분할)
        mask_split = grad_mag > grad_threshold
        mask_split &= self.scaling.max(dim=1).values >= scale_threshold
        
        # Prune (제거)
        mask_prune = self.opacity.squeeze() < opacity_threshold
        
        # 실제 적용 (간소화 버전)
        if mask_clone.sum() > 0:
            self._clone_points(mask_clone)
        if mask_split.sum() > 0:
            self._split_points(mask_split)
        if mask_prune.sum() > 0:
            self._prune_points(mask_prune)
    
    def _clone_points(self, mask: torch.Tensor):
        """포인트 복제"""
        new_xyz = self._xyz[mask].clone()
        new_xyz += torch.randn_like(new_xyz) * 0.01  # 약간의 노이즈
        
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, self._features_dc[mask]]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, self._features_rest[mask]]))
        self._opacity = nn.Parameter(torch.cat([self._opacity, self._opacity[mask]]))
        self._scaling = nn.Parameter(torch.cat([self._scaling, self._scaling[mask]]))
        self._rotation = nn.Parameter(torch.cat([self._rotation, self._rotation[mask]]))
        
        self.num_gaussians = len(self._xyz)
    
    def _split_points(self, mask: torch.Tensor):
        """포인트 분할"""
        # 간소화: Clone과 동일하게 처리하되 스케일 축소
        new_xyz = self._xyz[mask].clone()
        new_xyz += torch.randn_like(new_xyz) * self.scaling[mask].mean(dim=1, keepdim=True)
        
        new_scaling = self._scaling[mask] - math.log(1.6)  # 스케일 축소
        
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, self._features_dc[mask]]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, self._features_rest[mask]]))
        self._opacity = nn.Parameter(torch.cat([self._opacity, self._opacity[mask]]))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling]))
        self._rotation = nn.Parameter(torch.cat([self._rotation, self._rotation[mask]]))
        
        self.num_gaussians = len(self._xyz)
    
    def _prune_points(self, mask: torch.Tensor):
        """포인트 제거"""
        keep_mask = ~mask
        
        self._xyz = nn.Parameter(self._xyz[keep_mask])
        self._features_dc = nn.Parameter(self._features_dc[keep_mask])
        self._features_rest = nn.Parameter(self._features_rest[keep_mask])
        self._opacity = nn.Parameter(self._opacity[keep_mask])
        self._scaling = nn.Parameter(self._scaling[keep_mask])
        self._rotation = nn.Parameter(self._rotation[keep_mask])
        
        self.num_gaussians = len(self._xyz)
    
    def reset_opacity(self):
        """Opacity 리셋"""
        self._opacity.data = torch.logit(
            torch.min(self.opacity, torch.ones_like(self._opacity) * 0.01)
        )
    
    def save_ply(self, path: Union[str, Path]):
        """PLY 파일로 저장 (의존성 없는 ASCII 형식)"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy().reshape(-1, 3)
        opacities = self._opacity.detach().cpu().numpy().squeeze()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()
        
        # SH DC를 RGB로 변환
        C0 = 0.28209479177387814
        colors = np.clip((f_dc * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
        
        # ASCII PLY 형식으로 저장
        with open(path, 'w') as f:
            # 헤더
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {self.num_gaussians}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float opacity\n")
            f.write("property float scale_0\n")
            f.write("property float scale_1\n")
            f.write("property float scale_2\n")
            f.write("property float rot_0\n")
            f.write("property float rot_1\n")
            f.write("property float rot_2\n")
            f.write("property float rot_3\n")
            f.write("end_header\n")
            
            # 데이터
            for i in range(self.num_gaussians):
                f.write(f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} ")
                f.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} ")
                f.write(f"{opacities[i]:.6f} ")
                f.write(f"{scales[i, 0]:.6f} {scales[i, 1]:.6f} {scales[i, 2]:.6f} ")
                f.write(f"{rotations[i, 0]:.6f} {rotations[i, 1]:.6f} {rotations[i, 2]:.6f} {rotations[i, 3]:.6f}\n")
        
        print(f"💾 PLY 저장: {path} ({self.num_gaussians} gaussians)")
        return path
    
    @classmethod
    def load_ply(cls, path: Union[str, Path], config: GaussianModelConfig, device: str = "cuda"):
        """PLY 파일에서 로드"""
        from plyfile import PlyData
        
        plydata = PlyData.read(str(path))
        vertex = plydata['vertex']
        
        model = cls(config)
        
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        model._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device))
        
        # 기타 속성 로드 (필요시 확장)
        model.num_gaussians = len(xyz)
        
        return model


def build_gaussian_model(config) -> GaussianModel:
    """Config에서 Gaussian 모델 빌드"""
    model_cfg = config.model
    
    gs_config = GaussianModelConfig(
        sh_degree=model_cfg.sh_degree,
        spatial_lr_scale=model_cfg.init.spatial_lr_scale,
        opacity_init=model_cfg.gaussian.opacity_init,
        scale_init=model_cfg.gaussian.scale_init,
        position_lr=config.training.learning_rate.position_lr_init,
        feature_lr=config.training.learning_rate.feature_lr,
        opacity_lr=config.training.learning_rate.opacity_lr,
        scaling_lr=config.training.learning_rate.scaling_lr,
        rotation_lr=config.training.learning_rate.rotation_lr,
    )
    
    return GaussianModel(gs_config)


if __name__ == "__main__":
    # 테스트
    print("=== GaussianModel 테스트 ===")
    
    config = GaussianModelConfig(sh_degree=3)
    model = GaussianModel(config)
    
    # 랜덤 초기화
    model.init_random(1000, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Num Gaussians: {model.num_gaussians}")
    print(f"XYZ shape: {model.xyz.shape}")
    print(f"Features shape: {model.features.shape}")
    print(f"Opacity shape: {model.opacity.shape}")
    print(f"Scaling shape: {model.scaling.shape}")
    print(f"Rotation shape: {model.rotation.shape}")
    
    # PLY 저장
    model.save_ply("/tmp/test_gaussian.ply")
