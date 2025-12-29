#!/usr/bin/env python3
"""
NVS (3D Gaussian Splatting) 렌더링 스크립트
학습된 모델을 불러와 카메라 경로를 따라 영상 생성

사용법:
    # MLflow Run ID로 모델 로드
    python src/inference/render_nvs.py --run-id abc123 --camera-path cameras.json -o output/
    
    # 체크포인트 직접 로드
    python src/inference/render_nvs.py --checkpoint output/point_cloud.ply --camera-path cameras.json
    
    # 360도 자동 생성
    python src/inference/render_nvs.py --run-id abc123 --auto-orbit --num-frames 120
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
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

from src.models.gaussian_model import GaussianModel, GaussianModelConfig


# ============================================
# 카메라 경로 처리
# ============================================
def load_camera_path(json_path: str) -> List[Dict]:
    """
    카메라 경로 JSON 로드
    
    JSON 형식:
    {
        "camera_path": [
            {
                "camera_to_world": [[...], [...], [...], [...]],  # 4x4 행렬
                "fov": 60,  # FOV in degrees
                "aspect": 1.777  # width/height
            },
            ...
        ],
        "render_height": 720,
        "render_width": 1280
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "camera_path" in data:
        return data["camera_path"], data.get("render_width", 1280), data.get("render_height", 720)
    else:
        # 단순 리스트 형식
        return data, 1280, 720


def generate_orbit_cameras(
    num_frames: int = 120,
    radius: float = 3.0,
    height: float = 0.5,
    target: Tuple[float, float, float] = (0, 0, 0),
    fov: float = 60.0
) -> List[Dict]:
    """
    360도 orbit 카메라 경로 생성
    """
    cameras = []
    
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        
        # 카메라 위치
        cam_pos = np.array([
            radius * np.cos(angle) + target[0],
            height + target[1],
            radius * np.sin(angle) + target[2]
        ])
        
        # Look-at 변환
        target_vec = np.array(target)
        up = np.array([0, 1, 0])
        
        forward = target_vec - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        
        # Camera-to-World 변환 (4x4)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        
        cameras.append({
            "camera_to_world": c2w.tolist(),
            "fov": fov,
            "aspect": 16/9
        })
    
    return cameras


def camera_to_view_params(camera_data: Dict, width: int, height: int, device: str = "cuda"):
    """
    카메라 데이터를 렌더링용 파라미터로 변환
    """
    c2w = np.array(camera_data["camera_to_world"])
    fov = camera_data.get("fov", 60)
    
    # World-to-Camera (c2w의 역행렬)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    
    # FOV를 라디안으로
    fov_rad = np.deg2rad(fov)
    FoVx = fov_rad
    FoVy = fov_rad * height / width
    
    return {
        "R": R,
        "T": T,
        "FoVx": FoVx,
        "FoVy": FoVy,
        "width": width,
        "height": height
    }


# ============================================
# 렌더러
# ============================================
class NVSRenderer:
    """NVS 렌더러"""
    
    def __init__(self, gaussians: GaussianModel, device: str = "cuda"):
        self.gaussians = gaussians
        self.device = device
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        
        # gsplat 체크
        try:
            import gsplat
            from gsplat import rasterization
            self.gsplat_available = True
            self.rasterization = rasterization
        except ImportError:
            self.gsplat_available = False
            print("⚠️ gsplat 미설치 - Mock 렌더러 사용")
    
    def render(self, camera_params: Dict) -> np.ndarray:
        """
        단일 뷰 렌더링
        
        Returns:
            [H, W, 3] uint8 이미지
        """
        with torch.no_grad():
            if self.gsplat_available:
                rendered = self._render_gsplat(camera_params)
            else:
                rendered = self._render_mock(camera_params)
        
        # Tensor -> numpy uint8
        img = rendered.cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img
    
    def _render_gsplat(self, camera_params: Dict) -> torch.Tensor:
        """gsplat 렌더링"""
        H, W = camera_params["height"], camera_params["width"]
        
        # View matrix
        R = torch.tensor(camera_params["R"], dtype=torch.float32, device=self.device)
        T = torch.tensor(camera_params["T"], dtype=torch.float32, device=self.device)
        
        view_mat = torch.eye(4, device=self.device)
        view_mat[:3, :3] = R
        view_mat[:3, 3] = T
        
        # K matrix
        fx = W / (2 * np.tan(camera_params["FoVx"] / 2))
        fy = H / (2 * np.tan(camera_params["FoVy"] / 2))
        K = torch.tensor([
            [fx, 0, W/2],
            [0, fy, H/2],
            [0, 0, 1]
        ], device=self.device)
        
        # 렌더링
        try:
            renders, alphas, _ = self.rasterization(
                means=self.gaussians.xyz,
                quats=self.gaussians.rotation,
                scales=self.gaussians.scaling,
                opacities=self.gaussians.opacity.squeeze(-1),
                colors=self.gaussians.features[:, 0, :] * 0.28209479177387814 + 0.5,
                viewmats=view_mat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W,
                height=H,
                sh_degree=0,
                backgrounds=self.background.unsqueeze(0),
            )
            rendered = renders[0]  # [H, W, 3]
        except Exception as e:
            print(f"⚠️ gsplat 렌더링 실패: {e}")
            return self._render_mock(camera_params)
        
        return rendered
    
    def _render_mock(self, camera_params: Dict) -> torch.Tensor:
        """Mock 렌더링"""
        H, W = camera_params["height"], camera_params["width"]
        
        # Gaussian 색상의 가중 평균
        colors = self.gaussians.features[:, 0, :]
        opacities = self.gaussians.opacity.squeeze(-1)
        
        rgb_colors = (colors * 0.28209479177387814 + 0.5).clamp(0, 1)
        weights = opacities.unsqueeze(-1)
        weighted_color = (rgb_colors * weights).sum(dim=0) / (weights.sum() + 1e-8)
        
        # 전체 이미지를 평균색으로
        rendered = weighted_color.view(1, 1, 3).expand(H, W, 3).contiguous()
        
        return rendered


# ============================================
# MinIO 업로드
# ============================================
class MinIOUploader:
    """MinIO 업로드"""
    
    def __init__(self, endpoint: str = None, bucket: str = "mlflow-artifacts"):
        import boto3
        
        self.endpoint = endpoint or os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        self.bucket = bucket
        
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio_secure_password_2024"),
        )
    
    def upload(self, local_path: Path, s3_key: str) -> str:
        """파일 업로드 및 다운로드 링크 반환"""
        self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
        
        # 다운로드 링크 생성 (presigned URL)
        url = self.s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': s3_key},
            ExpiresIn=86400 * 7  # 7일
        )
        
        return url
    
    def get_public_url(self, s3_key: str) -> str:
        """공개 URL (MinIO Console)"""
        return f"{self.endpoint}/{self.bucket}/{s3_key}"


# ============================================
# 비디오 생성
# ============================================
def create_video_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
    codec: str = "libx264",
    crf: int = 23
) -> Path:
    """
    프레임 이미지들로 MP4 비디오 생성
    OpenCV VideoWriter 사용 (ffmpeg fallback)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 프레임 파일 목록
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"프레임 파일이 없습니다: {frames_dir}")
    
    print(f"🎬 비디오 생성 중: {output_path}")
    print(f"   📊 프레임 수: {len(frame_files)}")
    
    # 첫 프레임에서 크기 확인
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # OpenCV VideoWriter로 비디오 생성
    # mp4v 코덱 사용 (대부분의 환경에서 호환)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # .mp4 확장자 보장
    if output_path.suffix.lower() != '.mp4':
        output_path = output_path.with_suffix('.mp4')
    
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print("   ⚠️ OpenCV VideoWriter 실패, ffmpeg 시도...")
        return _create_video_ffmpeg(frames_dir, output_path, fps, codec, crf)
    
    # 프레임 쓰기
    for frame_path in tqdm(frame_files, desc="비디오 생성"):
        frame = cv2.imread(str(frame_path))
        writer.write(frame)
    
    writer.release()
    print(f"   ✅ 비디오 생성 완료 (OpenCV)")
    
    return output_path


def _create_video_ffmpeg(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
    codec: str = "libx264",
    crf: int = 23
) -> Path:
    """ffmpeg로 비디오 생성 (fallback)"""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ✅ 비디오 생성 완료 (ffmpeg)")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"   ❌ ffmpeg 실패: {e}")
        raise RuntimeError("비디오 생성 실패. OpenCV와 ffmpeg 모두 사용 불가.")
    
    return output_path


# ============================================
# 모델 로드
# ============================================
def load_model_from_mlflow(run_id: str, device: str = "cuda") -> GaussianModel:
    """MLflow Run에서 모델 로드"""
    import mlflow
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"📡 MLflow: {tracking_uri}")
    print(f"🔍 Run ID: {run_id}")
    
    client = mlflow.tracking.MlflowClient()
    
    # 아티팩트 다운로드
    artifact_path = client.download_artifacts(run_id, "model")
    print(f"   📥 아티팩트 다운로드: {artifact_path}")
    
    # PLY 파일 찾기
    artifact_dir = Path(artifact_path)
    ply_files = list(artifact_dir.glob("*.ply"))
    
    if not ply_files:
        raise FileNotFoundError(f"Run {run_id}에서 PLY 파일을 찾을 수 없습니다.")
    
    ply_path = ply_files[0]
    return load_model_from_ply(str(ply_path), device)


def load_model_from_ply(ply_path: str, device: str = "cuda") -> GaussianModel:
    """PLY 파일에서 모델 로드"""
    print(f"📦 PLY 로드: {ply_path}")
    
    # PLY 파싱 (ASCII 형식)
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    # 헤더 파싱
    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[-1])
    
    print(f"   📊 Vertices: {num_vertices}")
    
    # 데이터 파싱
    points = []
    colors = []
    opacities = []
    scales = []
    rotations = []
    
    for line in lines[header_end:]:
        parts = line.strip().split()
        if len(parts) < 14:
            continue
        
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
        opacity = float(parts[6])
        s0, s1, s2 = float(parts[7]), float(parts[8]), float(parts[9])
        r0, r1, r2, r3 = float(parts[10]), float(parts[11]), float(parts[12]), float(parts[13])
        
        points.append([x, y, z])
        colors.append([r/255.0, g/255.0, b/255.0])
        opacities.append(opacity)
        scales.append([s0, s1, s2])
        rotations.append([r0, r1, r2, r3])
    
    # Gaussian 모델 생성
    config = GaussianModelConfig()
    model = GaussianModel(config)
    
    model.num_gaussians = len(points)
    
    import torch.nn as nn
    
    model._xyz = nn.Parameter(
        torch.tensor(points, dtype=torch.float32, device=device)
    )
    
    # Colors to SH DC
    colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
    C0 = 0.28209479177387814
    sh_dc = (colors_tensor - 0.5) / C0
    model._features_dc = nn.Parameter(sh_dc.unsqueeze(1))
    
    # 나머지 SH 계수 (0)
    num_sh_rest = (config.sh_degree + 1) ** 2 - 1
    model._features_rest = nn.Parameter(
        torch.zeros(model.num_gaussians, num_sh_rest, 3, device=device)
    )
    
    # Opacity (logit 역변환)
    opacities_tensor = torch.tensor(opacities, dtype=torch.float32, device=device)
    model._opacity = nn.Parameter(
        torch.logit(opacities_tensor.clamp(1e-5, 1-1e-5)).unsqueeze(-1)
    )
    
    # Scales (이미 log space)
    model._scaling = nn.Parameter(
        torch.tensor(scales, dtype=torch.float32, device=device)
    )
    
    # Rotations
    model._rotation = nn.Parameter(
        torch.tensor(rotations, dtype=torch.float32, device=device)
    )
    
    print(f"   ✅ 모델 로드 완료: {model.num_gaussians} gaussians")
    
    return model


# ============================================
# 메인 렌더링 함수
# ============================================
def render_video(
    model: GaussianModel,
    cameras: List[Dict],
    output_dir: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    device: str = "cuda",
    upload_to_minio: bool = True
) -> Dict:
    """
    카메라 경로를 따라 비디오 렌더링
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = NVSRenderer(model, device)
    
    print(f"\n🎬 렌더링 시작: {len(cameras)} 프레임")
    print(f"   📐 해상도: {width}x{height}")
    
    # 프레임 렌더링
    for i, cam_data in enumerate(tqdm(cameras, desc="렌더링")):
        camera_params = camera_to_view_params(cam_data, width, height, device)
        
        frame = renderer.render(camera_params)
        
        # BGR로 변환하여 저장
        frame_path = frames_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # 비디오 생성
    video_name = f"render_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_path = output_dir / video_name
    
    create_video_from_frames(frames_dir, video_path, fps=fps)
    
    result = {
        "video_path": str(video_path),
        "num_frames": len(cameras),
        "resolution": f"{width}x{height}",
        "fps": fps
    }
    
    # MinIO 업로드
    if upload_to_minio:
        try:
            uploader = MinIOUploader()
            s3_key = f"nvs-renders/{video_name}"
            download_url = uploader.upload(video_path, s3_key)
            
            result["s3_key"] = s3_key
            result["download_url"] = download_url
            
            print(f"\n📤 MinIO 업로드 완료")
            print(f"   🔗 다운로드 링크 (7일 유효):")
            print(f"   {download_url}")
        except Exception as e:
            print(f"\n⚠️ MinIO 업로드 실패: {e}")
    
    return result


# ============================================
# CLI
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="NVS 렌더링 (3D Gaussian Splatting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # MLflow Run에서 모델 로드 + 카메라 경로
  python src/inference/render_nvs.py --run-id abc123 --camera-path cameras.json
  
  # 체크포인트 직접 로드 + 360도 자동 생성
  python src/inference/render_nvs.py --checkpoint point_cloud.ply --auto-orbit --num-frames 120
  
  # 고해상도 렌더링
  python src/inference/render_nvs.py --run-id abc123 --auto-orbit --width 1920 --height 1080
        """
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
        help="PLY 체크포인트 경로"
    )
    
    # 카메라 경로
    camera_group = parser.add_mutually_exclusive_group(required=True)
    camera_group.add_argument(
        "--camera-path", "-p",
        type=str,
        help="카메라 경로 JSON 파일"
    )
    camera_group.add_argument(
        "--auto-orbit",
        action="store_true",
        help="360도 orbit 카메라 자동 생성"
    )
    
    # 출력
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output/renders",
        help="출력 디렉토리"
    )
    
    # 렌더링 옵션
    parser.add_argument("--width", type=int, default=1280, help="렌더링 너비")
    parser.add_argument("--height", type=int, default=720, help="렌더링 높이")
    parser.add_argument("--fps", type=int, default=30, help="비디오 FPS")
    parser.add_argument("--num-frames", type=int, default=120, help="Orbit 프레임 수")
    
    # Orbit 옵션
    parser.add_argument("--radius", type=float, default=3.0, help="Orbit 반경")
    parser.add_argument("--height-offset", type=float, default=0.5, help="카메라 높이 오프셋")
    parser.add_argument("--fov", type=float, default=60.0, help="FOV (degrees)")
    
    # 기타
    parser.add_argument("--no-upload", action="store_true", help="MinIO 업로드 안 함")
    parser.add_argument("--device", type=str, default="auto", help="디바이스")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 NVS 렌더링 (3D Gaussian Splatting)")
    print("=" * 60)
    
    # 디바이스
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"📱 Device: {device}")
    
    # 모델 로드
    if args.run_id:
        model = load_model_from_mlflow(args.run_id, device)
    else:
        model = load_model_from_ply(args.checkpoint, device)
    
    # 카메라 경로
    if args.camera_path:
        cameras, width, height = load_camera_path(args.camera_path)
        # CLI 옵션으로 오버라이드 가능
        if args.width != 1280:
            width = args.width
        if args.height != 720:
            height = args.height
    else:
        # 자동 orbit 생성
        cameras = generate_orbit_cameras(
            num_frames=args.num_frames,
            radius=args.radius,
            height=args.height_offset,
            fov=args.fov
        )
        width, height = args.width, args.height
    
    print(f"   📷 카메라: {len(cameras)} 뷰")
    
    # 렌더링
    result = render_video(
        model=model,
        cameras=cameras,
        output_dir=Path(args.output),
        width=width,
        height=height,
        fps=args.fps,
        device=device,
        upload_to_minio=not args.no_upload
    )
    
    print("\n" + "=" * 60)
    print("✅ 렌더링 완료!")
    print("=" * 60)
    print(f"   🎥 비디오: {result['video_path']}")
    print(f"   📊 프레임: {result['num_frames']}")
    print(f"   📐 해상도: {result['resolution']}")
    
    if "download_url" in result:
        print(f"\n🔗 다운로드 링크:")
        print(f"   {result['download_url']}")


if __name__ == "__main__":
    main()
