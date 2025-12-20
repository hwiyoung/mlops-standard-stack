#!/usr/bin/env python3
"""
Novel View Synthesis (NVS) Mock 학습 스크립트
Gaussian Splatting 학습 과정 시뮬레이션 + MLflow 로깅
"""

import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import mlflow
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================
# 1. 가짜 이미지 데이터 생성
# ============================================
def create_sample_images(output_dir: str, num_images: int = 5) -> list:
    """
    COLMAP 스타일의 가짜 이미지 데이터 생성
    
    Args:
        output_dir: 이미지 저장 디렉토리
        num_images: 생성할 이미지 수
    
    Returns:
        생성된 이미지 경로 리스트
    """
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    for i in range(num_images):
        # 랜덤 색상의 그라데이션 배경 생성
        width, height = 640, 480
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 그라데이션 배경
        for y in range(height):
            ratio = y / height
            r = int(50 + 100 * ratio + random.randint(-20, 20))
            g = int(100 + 80 * (1 - ratio) + random.randint(-20, 20))
            b = int(150 + 50 * ratio + random.randint(-20, 20))
            img[y, :] = [max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r))]
        
        # 랜덤 도형 추가 (3D 오브젝트 시뮬레이션)
        num_shapes = random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle'])
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            
            if shape_type == 'circle':
                center = (random.randint(50, width-50), random.randint(50, height-50))
                radius = random.randint(20, 60)
                cv2.circle(img, center, radius, color, -1)
            else:
                pt1 = (random.randint(0, width-100), random.randint(0, height-100))
                pt2 = (pt1[0] + random.randint(30, 100), pt1[1] + random.randint(30, 80))
                cv2.rectangle(img, pt1, pt2, color, -1)
        
        # 카메라 각도 텍스트 추가
        angle = i * (360 // num_images)
        cv2.putText(img, f"View {i+1} ({angle} deg)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 저장
        filename = f"image_{i:04d}.jpg"
        filepath = images_dir / filename
        cv2.imwrite(str(filepath), img)
        image_paths.append(str(filepath))
        
        print(f"   ✅ 생성: {filename}")
    
    return image_paths


# ============================================
# 2. 렌더링 결과 이미지 생성
# ============================================
def generate_render_result(step: int, psnr: float, output_path: str) -> str:
    """
    현재 스텝의 렌더링 결과 이미지 생성
    
    Args:
        step: 현재 학습 스텝
        psnr: 현재 PSNR 값
        output_path: 저장 경로
    
    Returns:
        저장된 이미지 경로
    """
    width, height = 800, 600
    
    # 배경 그라데이션 (학습 진행에 따라 점점 선명해지는 효과)
    clarity = min(1.0, step / 100)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 그라데이션 배경 (파란색 계열)
    for y in range(height):
        ratio = y / height
        r = int(30 + 50 * ratio * clarity)
        g = int(50 + 100 * ratio * clarity)
        b = int(100 + 150 * (1 - ratio) * clarity)
        img[y, :] = [b, g, r]
    
    # 3D 오브젝트 시뮬레이션 (구체들)
    num_objects = 5
    np.random.seed(42)  # 일관된 오브젝트 배치
    
    for i in range(num_objects):
        center_x = int(100 + i * 150)
        center_y = int(200 + np.sin(i * 0.8) * 100)
        radius = int(40 + i * 10)
        
        # 학습 진행에 따라 오브젝트가 선명해짐
        alpha = int(100 + 155 * clarity)
        color = (
            int(200 * clarity + 55),
            int(150 * clarity + 50),
            int(100 * clarity + 30)
        )
        
        cv2.circle(img, (center_x, center_y), radius, color, -1)
        
        # 하이라이트 (3D 효과)
        highlight_offset = radius // 3
        cv2.circle(img, (center_x - highlight_offset, center_y - highlight_offset), 
                   radius // 4, (255, 255, 255), -1)
    
    # 노이즈 추가 (학습 초기에는 많고, 후반에는 적게)
    noise_level = int(50 * (1 - clarity))
    if noise_level > 0:
        noise = np.random.randint(-noise_level, noise_level, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 정보 오버레이
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    cv2.putText(img, f"Gaussian Splatting Training", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Step: {step}/100  |  PSNR: {psnr:.2f} dB", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 진행률 바
    bar_width = int(300 * (step / 100))
    cv2.rectangle(img, (20, height - 40), (320, height - 20), (50, 50, 50), -1)
    cv2.rectangle(img, (20, height - 40), (20 + bar_width, height - 20), (0, 200, 0), -1)
    
    cv2.imwrite(output_path, img)
    return output_path


# ============================================
# 3. 이미지를 MP4 동영상으로 변환
# ============================================
def create_video_from_images(image_paths: list, output_path: str, fps: int = 2) -> str:
    """
    이미지들을 MP4 동영상으로 변환 (브라우저 호환 H.264 코덱)
    
    Args:
        image_paths: 이미지 경로 리스트
        output_path: 출력 비디오 경로
        fps: 프레임 레이트
    
    Returns:
        생성된 비디오 경로
    """
    import subprocess
    import shutil
    
    if not image_paths:
        raise ValueError("이미지 경로가 비어있습니다.")
    
    # 첫 번째 이미지로 크기 확인
    first_img = cv2.imread(image_paths[0])
    height, width = first_img.shape[:2]
    
    # 임시 파일로 먼저 생성
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    
    # 비디오 라이터 설정 (OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            video_writer.write(img)
    
    video_writer.release()
    
    # ffmpeg로 H.264 코덱 변환 (브라우저 호환)
    if shutil.which('ffmpeg'):
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_output,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                output_path
            ], check=True, capture_output=True)
            os.remove(temp_output)
            print(f"   ✅ 동영상 생성 (H.264): {output_path}")
        except subprocess.CalledProcessError:
            # ffmpeg 실패시 원본 사용
            shutil.move(temp_output, output_path)
            print(f"   ⚠️ 동영상 생성 (mp4v 코덱): {output_path}")
    else:
        shutil.move(temp_output, output_path)
        print(f"   ⚠️ ffmpeg 없음, mp4v 코덱 사용: {output_path}")
    
    return output_path


# ============================================
# 4. PSNR 계산 (Mock)
# ============================================
def calculate_mock_psnr(step: int, max_steps: int = 100) -> float:
    """
    Mock PSNR 계산 (학습이 진행될수록 증가)
    실제로는 gt와 rendered 이미지 비교
    """
    # 로그 스케일로 증가하는 PSNR (20 -> 35 dB 범위)
    base_psnr = 20.0
    max_psnr = 35.0
    
    progress = step / max_steps
    psnr = base_psnr + (max_psnr - base_psnr) * (1 - np.exp(-3 * progress))
    
    # 약간의 노이즈 추가
    psnr += random.uniform(-0.5, 0.5)
    
    return psnr


# ============================================
# 5. 메인 학습 함수
# ============================================
def train_nvs_gaussian_splatting(
    data_dir: str = "./data/nvs_sample",
    num_steps: int = 100,
    log_interval: int = 10,
    render_steps: list = None
):
    """
    Gaussian Splatting 학습 시뮬레이션
    
    Args:
        data_dir: 데이터 디렉토리
        num_steps: 총 학습 스텝 수
        log_interval: 로깅 간격
        render_steps: 렌더링 결과 저장할 스텝 리스트
    """
    if render_steps is None:
        render_steps = [50, 100]
    
    print("=" * 60)
    print("🎬 Novel View Synthesis (Gaussian Splatting) 학습 파이프라인")
    print("=" * 60)
    
    # 1. 샘플 이미지 생성
    print("\n📦 샘플 이미지 생성 중 (COLMAP 데이터 구조 시뮬레이션)...")
    image_paths = create_sample_images(data_dir, num_images=5)
    print(f"   총 {len(image_paths)}장의 이미지 생성 완료")
    
    # 2. MLflow 실험 시작
    experiment_name = "nvs-gaussian-splatting"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"gs-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # 파라미터 로깅
        mlflow.log_params({
            "num_steps": num_steps,
            "num_images": len(image_paths),
            "log_interval": log_interval,
            "method": "3D Gaussian Splatting",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "densify_interval": 500,
            "render_resolution": "800x600"
        })
        
        print("\n🚀 Gaussian Splatting 학습 시작...")
        
        # 렌더링 결과 저장 경로
        render_results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in range(1, num_steps + 1):
                # PSNR 계산
                psnr = calculate_mock_psnr(step, num_steps)
                
                # 로깅 간격마다 메트릭 기록
                if step % log_interval == 0:
                    mlflow.log_metrics({
                        "psnr": psnr,
                        "ssim": 0.5 + 0.4 * (step / num_steps),  # Mock SSIM
                        "lpips": 0.5 - 0.4 * (step / num_steps),  # Mock LPIPS (낮을수록 좋음)
                        "num_gaussians": 10000 + step * 500,  # Gaussian 수 증가
                        "train_loss": 0.5 * np.exp(-step / 30) + 0.01  # 감소하는 loss
                    }, step=step)
                    
                    print(f"   Step [{step:3d}/{num_steps}] PSNR: {psnr:.2f} dB")
                
                # 지정된 스텝에서 렌더링 결과 저장
                if step in render_steps:
                    render_path = os.path.join(tmpdir, f"result_step_{step:03d}.jpg")
                    generate_render_result(step, psnr, render_path)
                    render_results.append(render_path)
                    
                    # MLflow Artifact로 업로드
                    mlflow.log_artifact(render_path, artifact_path="renders")
                    print(f"   📸 렌더링 결과 저장: result_step_{step:03d}.jpg")
            
            # 3. 동영상 생성
            print("\n🎥 렌더링 결과 동영상 생성 중...")
            
            # 추가 프레임 생성 (부드러운 동영상을 위해)
            all_frames = []
            for step in range(0, num_steps + 1, 5):  # 5스텝 간격
                psnr = calculate_mock_psnr(step if step > 0 else 1, num_steps)
                frame_path = os.path.join(tmpdir, f"frame_{step:03d}.jpg")
                generate_render_result(step if step > 0 else 1, psnr, frame_path)
                all_frames.append(frame_path)
            
            # MP4 동영상 생성
            video_path = os.path.join(tmpdir, "training_progress.mp4")
            create_video_from_images(all_frames, video_path, fps=2)
            
            # MLflow에 동영상 업로드
            mlflow.log_artifact(video_path, artifact_path="videos")
            print(f"   ✅ 동영상 업로드 완료: training_progress.mp4")
            
            # 최종 메트릭 로깅
            final_psnr = calculate_mock_psnr(num_steps, num_steps)
            mlflow.log_metrics({
                "final_psnr": final_psnr,
                "final_ssim": 0.92,
                "final_lpips": 0.08,
                "total_gaussians": 60000
            })
        
        print("\n" + "=" * 60)
        print("✅ Gaussian Splatting 학습 완료!")
        print("=" * 60)
        print(f"\n📊 최종 결과:")
        print(f"   - Final PSNR: {final_psnr:.2f} dB")
        print(f"   - Total Gaussians: 60,000")
        print(f"\n📌 결과 확인:")
        print(f"   - MLflow UI: http://localhost:5000")
        print(f"   - Experiment: {experiment_name}")


if __name__ == "__main__":
    train_nvs_gaussian_splatting(
        data_dir="./data/nvs_sample",
        num_steps=100,
        log_interval=10,
        render_steps=[50, 100]
    )
