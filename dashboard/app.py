"""
MLOps Web Dashboard
Streamlit 기반 MLOps 통합 대시보드

Tabs:
1. Data Manager - MinIO 데이터 업로드/조회
2. Training Lab - 학습 실행 및 모니터링
3. Model Registry - MLflow 모델 관리
4. Inference - 추론 및 시각화
"""

import os
import sys
import subprocess
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import yaml

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env 로드
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)


# ============================================
# 페이지 설정
# ============================================
st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .status-running {
        color: #ffa500;
        font-weight: bold;
    }
    .status-success {
        color: #00c853;
        font-weight: bold;
    }
    .status-error {
        color: #ff1744;
        font-weight: bold;
    }
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Consolas', 'Monaco', monospace;
        padding: 1rem;
        border-radius: 0.5rem;
        height: 400px;
        overflow-y: auto;
        font-size: 0.85rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 유틸리티 함수
# ============================================
def get_minio_client():
    """MinIO S3 클라이언트 생성"""
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio_secure_password_2024"),
    )


def list_minio_buckets() -> List[str]:
    """MinIO 버킷 목록 조회"""
    try:
        s3 = get_minio_client()
        response = s3.list_buckets()
        return [b["Name"] for b in response.get("Buckets", [])]
    except Exception as e:
        st.error(f"MinIO 연결 실패: {e}")
        return []


def list_minio_objects(bucket: str, prefix: str = "") -> List[Dict]:
    """MinIO 객체 목록 조회"""
    try:
        s3 = get_minio_client()
        paginator = s3.get_paginator('list_objects_v2')
        
        objects = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects.append({
                        "Key": obj["Key"],
                        "Size (KB)": round(obj["Size"] / 1024, 2),
                        "Last Modified": obj["LastModified"].strftime("%Y-%m-%d %H:%M")
                    })
        return objects
    except Exception as e:
        st.error(f"객체 조회 실패: {e}")
        return []


def get_config_files() -> List[str]:
    """configs/ 폴더의 YAML 파일 목록"""
    configs_dir = PROJECT_ROOT / "configs"
    if configs_dir.exists():
        return [f.name for f in configs_dir.glob("*.yaml")]
    return []


def load_config(config_name: str) -> dict:
    """YAML 설정 파일 로드"""
    config_path = PROJECT_ROOT / "configs" / config_name
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_command_async(cmd: List[str], cwd: str = None) -> subprocess.Popen:
    """비동기 명령어 실행"""
    return subprocess.Popen(
        cmd,
        cwd=cwd or str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )


def get_mlflow_runs(experiment_name: str = None, max_results: int = 10) -> pd.DataFrame:
    """MLflow 최근 실험 결과 조회"""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        
        client = mlflow.tracking.MlflowClient()
        
        if experiment_name:
            exp = client.get_experiment_by_name(experiment_name)
            if exp:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=max_results,
                    order_by=["start_time DESC"]
                )
            else:
                return pd.DataFrame()
        else:
            # 모든 실험
            experiments = client.search_experiments()
            exp_ids = [e.experiment_id for e in experiments if e.experiment_id != "0"]
            if not exp_ids:
                return pd.DataFrame()
            runs = client.search_runs(
                experiment_ids=exp_ids,
                max_results=max_results,
                order_by=["start_time DESC"]
            )
        
        data = []
        for run in runs:
            metrics = run.data.metrics
            data.append({
                "Run ID": run.info.run_id[:8] + "...",
                "Full Run ID": run.info.run_id,
                "Name": run.info.run_name or "N/A",
                "Experiment": run.info.experiment_id,
                "Status": run.info.status,
                "PSNR": metrics.get("final_psnr") or metrics.get("psnr", "-"),
                "IoU": metrics.get("val_iou", "-"),
                "Duration": f"{(run.info.end_time - run.info.start_time) / 1000:.0f}s" if run.info.end_time else "-"
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"MLflow 연결 실패: {e}")
        return pd.DataFrame()


# ============================================
# 사이드바
# ============================================
with st.sidebar:
    st.markdown("## 🚀 MLOps Dashboard")
    st.markdown("---")
    
    # 탭 선택
    selected_tab = st.radio(
        "메뉴 선택",
        ["📂 Data Manager", "🔬 Training Lab", "📦 Model Registry", "🔮 Inference"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # 상태 표시
    st.markdown("### 🔌 서비스 상태")
    
    # MinIO 상태
    try:
        buckets = list_minio_buckets()
        st.success(f"✅ MinIO ({len(buckets)} buckets)")
    except:
        st.error("❌ MinIO")
    
    # MLflow 상태
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        experiments = mlflow.tracking.MlflowClient().search_experiments()
        st.success(f"✅ MLflow ({len(experiments)} experiments)")
    except:
        st.error("❌ MLflow")
    
    st.markdown("---")
    st.markdown("##### 🔗 Quick Links")
    st.markdown(f"- [MLflow UI](http://localhost:5000)")
    st.markdown(f"- [MinIO Console](http://localhost:9001)")


# ============================================
# Tab 1: Data Manager
# ============================================
if selected_tab == "📂 Data Manager":
    st.markdown('<h1 class="main-header">📂 Data Manager</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 데이터 업로드")
        
        with st.form("upload_form"):
            project_name = st.text_input("프로젝트 이름", placeholder="my_project")
            local_path = st.text_input("로컬 데이터 경로", placeholder="./data/my_data")
            bucket = st.selectbox("대상 버킷", ["raw-data", "raw-data-nvs", "processed-data"])
            
            upload_btn = st.form_submit_button("🚀 업로드 시작", use_container_width=True)
        
        if upload_btn:
            if project_name and local_path:
                if Path(local_path).exists():
                    with st.spinner("업로드 중..."):
                        cmd = [
                            "python", "scripts/upload_data.py",
                            "--source", local_path,
                            "--bucket", bucket,
                            "--prefix", project_name
                        ]
                        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            st.success("✅ 업로드 완료!")
                            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                        else:
                            st.error(f"❌ 업로드 실패")
                            st.code(result.stderr)
                else:
                    st.error(f"경로가 존재하지 않습니다: {local_path}")
            else:
                st.warning("프로젝트 이름과 경로를 입력해주세요.")
    
    with col2:
        st.subheader("📋 MinIO 데이터 목록")
        
        buckets = list_minio_buckets()
        if buckets:
            selected_bucket = st.selectbox("버킷 선택", buckets)
            prefix_filter = st.text_input("Prefix 필터", placeholder="project/")
            
            if st.button("🔄 새로고침", key="refresh_minio"):
                st.rerun()
            
            objects = list_minio_objects(selected_bucket, prefix_filter)
            if objects:
                df = pd.DataFrame(objects)
                st.dataframe(df, use_container_width=True, height=400)
                st.info(f"총 {len(objects)}개 객체")
            else:
                st.info("객체가 없습니다.")


# ============================================
# Tab 2: Training Lab
# ============================================
elif selected_tab == "🔬 Training Lab":
    st.markdown('<h1 class="main-header">🔬 Training Lab</h1>', unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if "training_process" not in st.session_state:
        st.session_state.training_process = None
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ 학습 설정")
        
        task_type = st.selectbox(
            "Task Type",
            ["Change Detection", "Novel View Synthesis (3DGS)"]
        )
        
        config_files = get_config_files()
        if task_type == "Change Detection":
            default_config = "train_cd.yaml" if "train_cd.yaml" in config_files else config_files[0]
        else:
            default_config = "train_nvs.yaml" if "train_nvs.yaml" in config_files else config_files[0]
        
        selected_config = st.selectbox("Config 파일", config_files, index=config_files.index(default_config) if default_config in config_files else 0)
        
        # 설정 로드 및 편집
        if selected_config:
            config = load_config(selected_config)
            
            st.markdown("##### 주요 파라미터 수정")
            
            if task_type == "Change Detection":
                epochs = st.number_input("Epochs", value=config.get("training", {}).get("epochs", 50), min_value=1)
                batch_size = st.number_input("Batch Size", value=config.get("training", {}).get("batch_size", 8), min_value=1)
                lr = st.number_input("Learning Rate", value=config.get("training", {}).get("optimizer", {}).get("lr", 0.001), format="%.5f")
                
                overrides = f"training.epochs={epochs} training.batch_size={batch_size} training.optimizer.lr={lr}"
                script = "src/training/train_cd.py"
            else:
                iterations = st.number_input("Iterations", value=config.get("training", {}).get("iterations", 30000), min_value=100, step=1000)
                sh_degree = st.number_input("SH Degree", value=config.get("model", {}).get("sh_degree", 3), min_value=0, max_value=3)
                
                overrides = f"training.iterations={iterations} model.sh_degree={sh_degree}"
                script = "src/training/train_nvs.py"
        
        st.markdown("---")
        
        # 학습 실행 버튼
        start_btn = st.button("🚀 학습 시작", use_container_width=True, type="primary")
        stop_btn = st.button("⏹️ 학습 중지", use_container_width=True)
        
        if start_btn:
            cmd = ["python", script, "--config", f"configs/{selected_config}", "-o"] + overrides.split()
            st.session_state.training_process = run_command_async(cmd, str(PROJECT_ROOT))
            st.session_state.training_logs = []
            st.success("학습이 시작되었습니다!")
        
        if stop_btn and st.session_state.training_process:
            st.session_state.training_process.terminate()
            st.session_state.training_process = None
            st.warning("학습이 중지되었습니다.")
    
    with col2:
        st.subheader("📊 실행 로그")
        
        log_container = st.empty()
        status_container = st.empty()
        
        # 프로세스 상태 확인 및 로그 업데이트
        if st.session_state.training_process:
            proc = st.session_state.training_process
            
            # Non-blocking read
            import select
            if proc.poll() is None:
                status_container.markdown('<span class="status-running">🔄 학습 진행 중...</span>', unsafe_allow_html=True)
                
                # 로그 읽기 (non-blocking)
                try:
                    while True:
                        line = proc.stdout.readline()
                        if line:
                            st.session_state.training_logs.append(line.strip())
                            # 최근 100줄만 유지
                            if len(st.session_state.training_logs) > 100:
                                st.session_state.training_logs = st.session_state.training_logs[-100:]
                        else:
                            break
                except:
                    pass
            else:
                if proc.returncode == 0:
                    status_container.markdown('<span class="status-success">✅ 학습 완료!</span>', unsafe_allow_html=True)
                else:
                    status_container.markdown('<span class="status-error">❌ 학습 실패</span>', unsafe_allow_html=True)
                st.session_state.training_process = None
        
        # 로그 표시
        if st.session_state.training_logs:
            log_text = "\n".join(st.session_state.training_logs[-50:])
            log_container.code(log_text, language="bash")
        else:
            log_container.info("학습을 시작하면 로그가 여기에 표시됩니다.")
        
        # 자동 새로고침
        if st.session_state.training_process:
            time.sleep(1)
            st.rerun()


# ============================================
# Tab 3: Model Registry
# ============================================
elif selected_tab == "📦 Model Registry":
    st.markdown('<h1 class="main-header">📦 Model Registry</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 최근 학습 결과")
        
        experiment_filter = st.selectbox(
            "실험 필터",
            ["전체", "satellite-change-detection", "nvs-gaussian-splatting"]
        )
        
        exp_name = None if experiment_filter == "전체" else experiment_filter
        runs_df = get_mlflow_runs(exp_name, max_results=20)
        
        if not runs_df.empty:
            # 표시용 컬럼만 선택
            display_df = runs_df[["Run ID", "Name", "Status", "PSNR", "IoU", "Duration"]]
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # 선택한 Run의 상세 정보
            selected_run = st.selectbox("상세 보기", runs_df["Full Run ID"].tolist())
            if selected_run:
                st.markdown(f"**Full Run ID:** `{selected_run}`")
                st.link_button(
                    "🔗 MLflow에서 보기",
                    f"http://localhost:5000/#/experiments/0/runs/{selected_run}",
                    use_container_width=True
                )
        else:
            st.info("학습 결과가 없습니다.")
    
    with col2:
        st.subheader("🔗 MLflow UI")
        
        mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        
        st.link_button(
            "🌐 MLflow UI 열기",
            mlflow_url,
            use_container_width=True,
            type="primary"
        )
        
        st.markdown("---")
        
        st.markdown("##### 📈 성능 요약")
        
        if not runs_df.empty:
            # PSNR 통계
            psnr_values = runs_df[runs_df["PSNR"] != "-"]["PSNR"].astype(float)
            if len(psnr_values) > 0:
                st.metric("평균 PSNR", f"{psnr_values.mean():.2f} dB")
                st.metric("최고 PSNR", f"{psnr_values.max():.2f} dB")
            
            # IoU 통계
            iou_values = runs_df[runs_df["IoU"] != "-"]["IoU"].astype(float)
            if len(iou_values) > 0:
                st.metric("평균 IoU", f"{iou_values.mean():.4f}")
                st.metric("최고 IoU", f"{iou_values.max():.4f}")


# ============================================
# Tab 4: Inference
# ============================================
elif selected_tab == "🔮 Inference":
    st.markdown('<h1 class="main-header">🔮 Inference</h1>', unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if "inference_result" not in st.session_state:
        st.session_state.inference_result = None
    
    task_type = st.radio(
        "Task Type",
        ["Change Detection", "Novel View Synthesis"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if task_type == "Change Detection":
        st.subheader("🛰️ Change Detection Inference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 모델 선택")
            model_source = st.radio("모델 소스", ["체크포인트 파일", "MLflow Run ID"], horizontal=True)
            
            if model_source == "체크포인트 파일":
                checkpoint = st.text_input("체크포인트 경로", value="checkpoints/final_model.pth")
            else:
                runs_df = get_mlflow_runs("satellite-change-detection", 10)
                if not runs_df.empty:
                    run_id = st.selectbox("Run 선택", runs_df["Full Run ID"].tolist())
                else:
                    run_id = st.text_input("Run ID 입력")
            
            st.markdown("##### 입력 이미지")
            pre_image = st.file_uploader("Pre 이미지 (GeoTIFF)", type=["tif", "tiff"])
            post_image = st.file_uploader("Post 이미지 (GeoTIFF)", type=["tif", "tiff"])
        
        with col2:
            st.markdown("##### 결과")
            
            if st.button("🔮 추론 실행", use_container_width=True, type="primary"):
                if pre_image and post_image:
                    with st.spinner("추론 중..."):
                        # 임시 파일 저장
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            pre_path = Path(tmpdir) / "pre.tif"
                            post_path = Path(tmpdir) / "post.tif"
                            output_dir = Path(tmpdir) / "output"
                            
                            pre_path.write_bytes(pre_image.read())
                            post_path.write_bytes(post_image.read())
                            
                            # 추론 실행
                            if model_source == "체크포인트 파일":
                                cmd = ["python", "src/inference/predict_cd.py",
                                       "--checkpoint", checkpoint,
                                       "--pre", str(pre_path),
                                       "--post", str(post_path),
                                       "-o", str(output_dir)]
                            else:
                                cmd = ["python", "src/inference/predict_cd.py",
                                       "--run-id", run_id,
                                       "--pre", str(pre_path),
                                       "--post", str(post_path),
                                       "-o", str(output_dir)]
                            
                            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                viz_path = output_dir / "visualization.png"
                                if viz_path.exists():
                                    st.image(str(viz_path), caption="추론 결과", use_container_width=True)
                                    st.success("✅ 추론 완료!")
                                else:
                                    st.warning("시각화 파일을 찾을 수 없습니다.")
                            else:
                                st.error("❌ 추론 실패")
                                st.code(result.stderr)
                else:
                    st.warning("Pre/Post 이미지를 모두 업로드해주세요.")
    
    else:  # NVS
        st.subheader("🎬 Novel View Synthesis Rendering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 모델 선택")
            model_source = st.radio("모델 소스", ["PLY 체크포인트", "MLflow Run ID"], horizontal=True, key="nvs_source")
            
            if model_source == "PLY 체크포인트":
                checkpoint = st.text_input("PLY 경로", value="output/nvs/point_cloud.ply")
            else:
                runs_df = get_mlflow_runs("nvs-gaussian-splatting", 10)
                if not runs_df.empty:
                    run_id = st.selectbox("Run 선택", runs_df["Full Run ID"].tolist(), key="nvs_run")
                else:
                    run_id = st.text_input("Run ID 입력", key="nvs_run_input")
            
            st.markdown("##### 렌더링 설정")
            num_frames = st.slider("프레임 수", 30, 240, 60)
            fps = st.slider("FPS", 15, 60, 30)
            resolution = st.selectbox("해상도", ["1280x720", "1920x1080", "640x480"])
            width, height = map(int, resolution.split("x"))
        
        with col2:
            st.markdown("##### 결과")
            
            if st.button("🎬 렌더링 실행", use_container_width=True, type="primary"):
                with st.spinner("렌더링 중... (시간이 걸릴 수 있습니다)"):
                    output_dir = PROJECT_ROOT / "output" / "dashboard_render"
                    
                    if model_source == "PLY 체크포인트":
                        cmd = ["python", "src/inference/render_nvs.py",
                               "--checkpoint", checkpoint,
                               "--auto-orbit",
                               "--num-frames", str(num_frames),
                               "--fps", str(fps),
                               "--width", str(width),
                               "--height", str(height),
                               "-o", str(output_dir)]
                    else:
                        cmd = ["python", "src/inference/render_nvs.py",
                               "--run-id", run_id,
                               "--auto-orbit",
                               "--num-frames", str(num_frames),
                               "--fps", str(fps),
                               "--width", str(width),
                               "--height", str(height),
                               "-o", str(output_dir)]
                    
                    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        # 비디오 파일 찾기
                        video_files = list(output_dir.glob("*.mp4"))
                        if video_files:
                            video_path = video_files[-1]
                            st.video(str(video_path))
                            st.success("✅ 렌더링 완료!")
                            
                            # 다운로드 버튼
                            with open(video_path, "rb") as f:
                                st.download_button(
                                    "📥 비디오 다운로드",
                                    f.read(),
                                    file_name=video_path.name,
                                    mime="video/mp4"
                                )
                        else:
                            st.warning("비디오 파일을 찾을 수 없습니다.")
                    else:
                        st.error("❌ 렌더링 실패")
                        st.code(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)


# ============================================
# 푸터
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.8rem;'>
        🚀 MLOps Standard Stack Dashboard | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
