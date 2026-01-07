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

# 외부 접근을 위한 공인 IP 설정
# 외부 접근을 위한 공인 IP 설정
def get_default_public_ip():
    # 1. 브라우저 접속 기반 감지 (Streamlit 1.34+) - 최우선
    try:
        host = st.context.headers.get("host", "")
        if host:
            if ":" in host:
                ip = host.split(":")[0]
            else:
                ip = host
            # 내부/로컬 주소는 무시하고 실제 IP인 경우만 반환
            if ip not in ["localhost", "127.0.0.1", "mlflow", "minio", "0.0.0.0"]:
                return ip
    except:
        pass

    # 2. 환경변수 확인
    env_ip = os.getenv("PUBLIC_IP")
    if env_ip and env_ip not in ["localhost", "127.0.0.1", "mlflow", "minio"]:
        return env_ip
        
    # 3. 소켓 기반 감지 (서버의 기본 네트워크 IP)
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

# 기본 IP 초기화 (세션 상태 저장)
if "public_ip" not in st.session_state:
    st.session_state.public_ip = get_default_public_ip()

PUBLIC_IP = st.session_state.public_ip
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5000")
MINIO_CONSOLE_PORT = os.getenv("MINIO_CONSOLE_PORT", "9001")
MINIO_API_PORT = os.getenv("MINIO_API_PORT", "9000")

from src.models.gaussian_model import GaussianModel, GaussianModelConfig
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


def get_presigned_url(bucket: str, key: str, expires_in: int = 604800) -> str:
    """MinIO Presigned URL 생성 (외부 IP 반영)"""
    try:
        s3 = get_minio_client()
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expires_in
        )
        
        # 외부 접근을 위해 호스트명 교체 (대시보드 세션 IP 기준)
        public_ip = st.session_state.get("public_ip", "localhost")
        if public_ip not in ["localhost", "127.0.0.1", "mlflow", "minio"]:
            # http://minio:9000/... -> http://PUBLIC_IP:9000/...
            # replace()를 사용하여 정규식 역참조 문제 방지
            url = url.replace("http://minio:9000", f"http://{public_ip}:9000")
            url = url.replace("http://localhost:9000", f"http://{public_ip}:9000")
            
        return url
    except Exception as e:
        st.error(f"링크 생성 실패: {e}")
        return ""


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


def get_data_directories() -> List[str]:
    """./data 디렉토리의 하위 디렉토리 목록"""
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        return [d.name for d in data_dir.iterdir() if d.is_dir()]
    return []


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
        ["📂 Data Manager", "📍 지도 브라우저", "🔬 Training Lab", "📦 Model Registry", "🔮 Inference"],
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
    st.markdown("### 🌐 네트워크 설정")
    
    # 헬프 텍스트 추가
    st.info("💡 다른 컴퓨터에서 접속 중이라면 아래 IP가 서버의 실제 IP인지 확인해주세요.")
    
    new_ip = st.text_input("서버 IP (Server Host)", value=st.session_state.public_ip, help="외부 접속 시 링크가 생성될 IP 주소입니다.")
    if new_ip != st.session_state.public_ip:
        st.session_state.public_ip = new_ip
        st.rerun()

    st.markdown("##### 🔗 Quick Links (미리보기)")
    mlflow_ui_url = f"http://{st.session_state.public_ip}:{MLFLOW_PORT}"
    minio_ui_url = f"http://{st.session_state.public_ip}:{MINIO_CONSOLE_PORT}"
    
    st.markdown(f"- [📊 MLflow UI]({mlflow_ui_url})")
    st.markdown(f"- [📦 MinIO Console]({minio_ui_url})")
    
    if st.session_state.public_ip in ["localhost", "127.0.0.1", "mlflow"]:
        st.warning("⚠️ 현재 로컬/내부 주소로 설정되어 있어 외부 접속 시 링크가 작동하지 않을 수 있습니다.")


# ============================================
# Tab 1: Data Manager
# ============================================
if selected_tab == "📂 Data Manager":
    st.markdown('<h1 class="main-header">📂 Data Manager</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 데이터 업로드")
        st.caption("로컬 폴더를 MinIO 버킷으로 업로드합니다. 대용량 파일도 안정적으로 전송!")
        
        # 로컬 폴더 목록 조회 (/workspace/data 하위)
        try:
            data_root = Path("/workspace/data")
            if data_root.exists():
                # 모든 하위 디렉토리 탐색
                subdirs = sorted([d for d in data_root.rglob("*") if d.is_dir()])
                local_folder_options = [str(d) for d in subdirs]
            else:
                local_folder_options = []
        except:
            local_folder_options = []
            
        with st.form("upload_form"):
            if local_folder_options:
                source_path = st.selectbox("📁 로컬 폴더 선택", local_folder_options, 
                                           help="/workspace/data 하위의 폴더 중 업로드할 대상을 선택하세요.")
            else:
                source_path = st.text_input("📁 로컬 폴더 경로", placeholder="/workspace/data/folder_name", 
                                            help="업로드할 파일들이 있는 로컬 컴퓨터의 폴더 경로")
                
            bucket = st.selectbox("🪣 대상 버킷", ["raw-data", "raw-data-nvs", "processed-data"],
                                  help="MinIO에서 파일을 저장할 버킷")
            prefix = st.text_input("📂 버킷 내 저장 경로", placeholder="project_name/",
                                   help="버킷 안에서 파일들이 저장될 폴더 경로 (보통 폴더명과 동일하게 입력)")
            
            st.markdown("##### ⚙️ 옵션")
            overwrite = st.checkbox("기존 파일 덮어쓰기", value=False, help="이미 존재하는 파일을 덮어씁니다 (기본: 건너뜀)")
            
            upload_btn = st.form_submit_button("🚀 업로드 실행", use_container_width=True, type="primary")
        
        if upload_btn and source_path:
            if not Path(source_path).exists():
                st.error(f"❌ 경로가 존재하지 않습니다: {source_path}")
            else:
                # 파일 수 및 크기 계산
                files = list(Path(source_path).rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                st.info(f"📊 {file_count}개 파일, 총 {total_size / (1024*1024):.1f} MB")
                
                # mc 명령어 구성 (기본적으로 기존 파일은 건너뜀)
                mc_args = ["mc", "mirror"]
                if overwrite:
                    mc_args.append("--overwrite")
                mc_args.extend([f"{source_path}/", f"myminio/{bucket}/{prefix}"])
                
                # 업로드 실행
                progress_bar = st.progress(0, text="업로드 준비 중...")
                log_area = st.empty()
                
                try:
                    import subprocess
                    process = subprocess.Popen(
                        mc_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    output_lines = []
                    uploaded_count = 0
                    
                    for line in process.stdout:
                        output_lines.append(line.strip())
                        if len(output_lines) > 10:
                            output_lines = output_lines[-10:]
                        
                        # 진행률 업데이트 (파일명이 출력될 때마다)
                        if line.strip() and not line.startswith("mc:"):
                            uploaded_count += 1
                            progress = min(uploaded_count / max(file_count, 1), 1.0)
                            progress_bar.progress(progress, text=f"업로드 중... {uploaded_count}/{file_count}")
                        
                        log_area.code("\n".join(output_lines), language="text")
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        progress_bar.progress(1.0, text="✅ 업로드 완료!")
                        st.success(f"✅ {file_count}개 파일 업로드 완료!")
                        
                        # 자동 인덱싱 실행
                        with st.spinner("🔄 지도 브라우저용 인덱싱 중..."):
                            idx_cmd = [
                                "python", "-m", "src.indexer.metadata_extractor",
                                "--bucket", bucket,
                                "--prefix", prefix
                            ]
                            idx_result = subprocess.run(idx_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
                            if idx_result.returncode == 0:
                                st.success("🗺️ 인덱싱 완료! 지도 브라우저 탭에서 확인하세요.")
                            else:
                                st.warning("⚠️ 인덱싱 실패 (수동으로 지도 브라우저에서 실행하세요)")
                    else:
                        st.error("❌ 업로드 중 오류 발생")
                        
                except FileNotFoundError:
                    st.error("❌ mc CLI가 설치되지 않았습니다. 터미널에서 `./scripts/setup_minio_cli.sh` 를 먼저 실행하세요.")
                except Exception as e:
                    st.error(f"❌ 오류: {e}")
        
        st.markdown("---")
        st.subheader("📥 데이터 다운로드")
        st.caption("MinIO 버킷의 데이터를 로컬로 다운로드합니다.")
        
        # 버킷 선택 (폼 외부)
        dl_bucket = st.selectbox("🪣 버킷 선택", ["raw-data", "raw-data-nvs", "processed-data", "mlflow-artifacts"], key="dl_bucket")
        
        # 선택된 버킷의 폴더 목록 조회
        try:
            s3 = get_minio_client()
            paginator = s3.get_paginator("list_objects_v2")
            folders = set()
            for page in paginator.paginate(Bucket=dl_bucket):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # 모든 폴더 경로 추출 (중첩 포함)
                    parts = key.split("/")
                    for i in range(1, len(parts)):
                        folder_path = "/".join(parts[:i]) + "/"
                        # thumbnails 폴더 제외
                        if not folder_path.startswith("thumbnails"):
                            folders.add(folder_path)
            folder_list = sorted(list(folders))
        except:
            folder_list = []
        
        with st.form("download_form"):
            if folder_list:
                dl_prefix = st.selectbox("📂 다운로드할 폴더", folder_list, key="dl_prefix")
            else:
                dl_prefix = st.text_input("📂 버킷 내 경로", placeholder="project/output/", key="dl_prefix_text")
            
            dl_local = st.text_input("💾 로컬 저장 경로", placeholder="/workspace/downloads/", key="dl_local",
                                     help="다운로드한 파일을 저장할 경로 (컨테이너 기준)")
            
            download_btn = st.form_submit_button("📥 다운로드 실행", use_container_width=True, type="primary")
        
        if download_btn and dl_prefix and dl_local:
            # 다운로드 실행
            mc_args = ["mc", "mirror", f"myminio/{dl_bucket}/{dl_prefix}", dl_local]
            
            progress_bar = st.progress(0, text="다운로드 준비 중...")
            log_area = st.empty()
            
            try:
                # 대상 폴더 생성
                Path(dl_local).mkdir(parents=True, exist_ok=True)
                
                process = subprocess.Popen(
                    mc_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output_lines = []
                download_count = 0
                
                for line in process.stdout:
                    output_lines.append(line.strip())
                    if len(output_lines) > 10:
                        output_lines = output_lines[-10:]
                    
                    if line.strip() and not line.startswith("mc:"):
                        download_count += 1
                        progress_bar.progress(min(download_count / 100, 0.99), text=f"다운로드 중... {download_count}개 파일")
                    
                    log_area.code("\n".join(output_lines), language="text")
                
                process.wait()
                
                if process.returncode == 0:
                    progress_bar.progress(1.0, text="✅ 다운로드 완료!")
                    st.success(f"✅ 다운로드 완료! 저장 위치: {dl_local}")
                else:
                    st.error("❌ 다운로드 중 오류 발생")
                    
            except FileNotFoundError:
                st.error("❌ mc CLI가 설치되지 않았습니다.")
            except Exception as e:
                st.error(f"❌ 오류: {e}")
    
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
                st.dataframe(df, use_container_width=True, height=300)
                st.info(f"총 {len(objects)}개 객체")
                
                st.markdown("---")
                st.subheader("📥 개별 파일 다운로드 (임시 링크)")
                
                # 파일 선택용 selectbox
                file_keys = [obj["Key"] for obj in objects]
                selected_file = st.selectbox("다운로드할 파일 선택", file_keys)
                
                if st.button("🔗 임시 다운로드 링크 생성", use_container_width=True):
                    tmp_url = get_presigned_url(selected_bucket, selected_file)
                    if tmp_url:
                        st.success(f"✅ 링크가 생성되었습니다 (7일간 유효)")
                        st.code(tmp_url)
                        st.link_button("🌐 브라우저에서 열기 / 다운로드", tmp_url, use_container_width=True)
            else:
                st.info("객체가 없습니다.")


# ============================================
# Tab 2: 📍 지도 브라우저
# ============================================
elif selected_tab == "📍 지도 브라우저":
    st.markdown('<h1 class="main-header">📍 지도 브라우저</h1>', unsafe_allow_html=True)
    
    # STAC API 설정
    STAC_API_URL = os.getenv("STAC_API_URL", "http://localhost:8080")
    TITILER_URL = os.getenv("TITILER_URL", "http://localhost:8082")
    
    def get_stac_collections():
        """STAC 컬렉션 목록 조회"""
        try:
            import requests
            response = requests.get(f"{STAC_API_URL}/collections", timeout=5)
            if response.status_code == 200:
                return [c["id"] for c in response.json().get("collections", [])]
        except:
            pass
        return []
    
    def search_stac_items(collections=None, limit=500):
        """STAC 검색"""
        try:
            import requests
            params = {"limit": limit}
            if collections:
                params["collections"] = collections
            response = requests.post(f"{STAC_API_URL}/search", json=params, timeout=30)
            if response.status_code == 200:
                return response.json().get("features", [])
        except:
            pass
        return []
    
    def get_stac_item_count(collections=None):
        """STAC 아이템 개수"""
        items = search_stac_items(collections, limit=1000)
        return len(items)
    
    # 레거시 DB 연결 함수
    def get_db_connection():
        import psycopg2
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            user=os.getenv("POSTGRES_USER", "mlflow"),
            password=os.getenv("POSTGRES_PASSWORD", "mlflow123"),
            dbname=os.getenv("POSTGRES_DB", "mlflow"),
        )
    
    # STAC API 사용 가능 여부
    stac_available = len(get_stac_collections()) > 0
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🔍 필터")
        
        if stac_available:
            # STAC 모드: 컬렉션 필터
            collections = get_stac_collections()
            collection_filter = st.selectbox("📁 컬렉션", ["전체"] + collections, key="map_collection_filter")
            
            # 데이터 유형 (컬렉션 기반 자동 설정)
            if collection_filter == "drone-photos":
                data_type_filter = "사진 (photo)"
            elif collection_filter == "orthoimages":
                data_type_filter = "정사영상 (ortho)"
            else:
                data_type_filter = st.selectbox("📷 데이터 유형", ["전체", "사진 (photo)", "정사영상 (ortho)"], key="map_type_filter")
            
            bucket_filter = "전체"  # STAC 모드에서는 사용 안 함
            folder_filter = ""
        else:
            # 레거시 모드
            st.info("⚠️ STAC API 미연결 - 레거시 모드")
            collection_filter = "전체"
            
            # 버킷 필터
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT bucket FROM image_metadata ORDER BY bucket")
                db_buckets = [row[0] for row in cur.fetchall()]
                cur.close()
                conn.close()
            except:
                db_buckets = []
            
            bucket_filter = st.selectbox("📦 버킷", ["전체"] + db_buckets, key="map_bucket_filter")
            folder_filter = ""
            data_type_filter = st.selectbox("📷 데이터 유형", ["전체", "사진 (photo)", "정사영상 (ortho)"])
        
        st.markdown("---")
        
        # 통계 조회
        if stac_available:
            # STAC 모드
            selected_collections = None
            if collection_filter != "전체":
                selected_collections = [collection_filter]
            elif data_type_filter == "사진 (photo)":
                selected_collections = ["drone-photos"]
            elif data_type_filter == "정사영상 (ortho)":
                selected_collections = ["orthoimages"]
            
            filtered_count = get_stac_item_count(selected_collections)
            total_count = get_stac_item_count()
            st.metric("표시 데이터", f"{filtered_count}개", f"전체 {total_count}개 중")
        else:
            # 레거시 모드
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                
                where_clauses = ["1=1"]
                if bucket_filter != "전체":
                    where_clauses.append(f"bucket = '{bucket_filter}'")
                if data_type_filter == "사진 (photo)":
                    where_clauses.append("data_type = 'photo'")
                elif data_type_filter == "정사영상 (ortho)":
                    where_clauses.append("data_type = 'ortho'")
                
                where_sql = " AND ".join(where_clauses)
                
                cur.execute(f"SELECT COUNT(*) FROM image_metadata WHERE {where_sql}")
                filtered_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM image_metadata")
                total_count = cur.fetchone()[0]
                cur.close()
                conn.close()
                
                st.metric("표시 데이터", f"{filtered_count}개", f"전체 {total_count}개 중")
            except Exception as e:
                st.warning(f"DB 오류: {e}")
                filtered_count = 0
        
        st.markdown("---")
        
        # 인덱싱 섹션 (접힘)
        with st.expander("📊 신규 데이터 인덱싱"):
            buckets = list_minio_buckets()
            if buckets:
                idx_bucket = st.selectbox("버킷", buckets, key="idx_bucket")
                idx_prefix = st.text_input("Prefix", key="idx_prefix")
                
                if st.button("🔄 인덱싱 실행", use_container_width=True):
                    with st.spinner("인덱싱 중..."):
                        import subprocess
                        cmd = [
                            "python", "-m", "src.indexer.metadata_extractor",
                            "--bucket", idx_bucket,
                            "--prefix", idx_prefix
                        ]
                        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
                        
                        # 결과 파싱: 새로 추가된 파일 수 확인
                        output = result.stdout + result.stderr
                        new_count = output.count("✅ 인덱싱 완료:")
                        skip_count = output.count("이미 인덱싱됨") if "이미 인덱싱됨" in output else 0
                        
                        if result.returncode == 0:
                            st.success(f"✅ 완료! (신규: {new_count}개)")
                            st.rerun()
                        else:
                            st.error("❌ 실패")
                            st.code(result.stderr[-500:] if result.stderr else "알 수 없는 오류")
    
    with col1:
        st.subheader("🗺️ 지도")
        
        try:
            import folium
            from streamlit_folium import st_folium
            import json
            
            # 데이터 조회
            all_coords = []
            photos = []
            orthos = []
            
            if stac_available:
                # STAC 모드: API 검색
                search_collections = None
                if collection_filter != "전체":
                    search_collections = [collection_filter]
                elif data_type_filter == "사진 (photo)":
                    search_collections = ["drone-photos"]
                elif data_type_filter == "정사영상 (ortho)":
                    search_collections = ["orthoimages"]
                
                items = search_stac_items(collections=search_collections, limit=500)
                
                for item in items:
                    geom = item.get("geometry", {})
                    props = item.get("properties", {})
                    assets = item.get("assets", {})
                    
                    # thumbnail href에서 key 추출 (http://minio:9000/bucket/key 형태)
                    thumb_href = assets.get("thumbnail", {}).get("href", "")
                    thumb_key = ""
                    if thumb_href and "/thumbnails/" in thumb_href:
                        try:
                            # http://minio:9000/raw-data/thumbnails/xxx.jpg -> thumbnails/xxx.jpg
                            parts = thumb_href.split("/")
                            bucket_idx = parts.index("raw-data") if "raw-data" in parts else -1
                            if bucket_idx >= 0:
                                thumb_key = "/".join(parts[bucket_idx + 1:])
                        except:
                            pass
                    
                    if geom.get("type") == "Point":
                        lon, lat = geom["coordinates"]
                        all_coords.append((lat, lon))
                        photos.append({
                            "id": item["id"],
                            "filename": props.get("filename", item["id"]),
                            "bucket": props.get("bucket", "raw-data"),
                            "key": props.get("object_key", ""),
                            "lon": lon,
                            "lat": lat,
                            "file_size": props.get("file_size", 0),
                            "thumb_key": thumb_key or props.get("thumbnail_key", ""),
                        })
                    elif geom.get("type") == "Polygon":
                        bbox = item.get("bbox", [])
                        if len(bbox) >= 4:
                            center_lon = (bbox[0] + bbox[2]) / 2
                            center_lat = (bbox[1] + bbox[3]) / 2
                            all_coords.append((center_lat, center_lon))
                        orthos.append({
                            "id": item["id"],
                            "filename": props.get("filename", item["id"]),
                            "bucket": props.get("bucket", "raw-data"),
                            "key": props.get("object_key", ""),
                            "geometry": geom,
                            "resolution": props.get("proj:resolution", [None])[0] if isinstance(props.get("proj:resolution"), list) else props.get("proj:resolution"),
                            "file_size": props.get("file_size", 0),
                            "thumb_key": thumb_key or props.get("thumbnail_key", ""),
                        })
            else:
                # 레거시 DB 모드
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    
                    where_clauses = ["1=1"]
                    if bucket_filter != "전체":
                        where_clauses.append(f"bucket = '{bucket_filter}'")
                    if data_type_filter == "사진 (photo)":
                        where_clauses.append("data_type = 'photo'")
                    elif data_type_filter == "정사영상 (ortho)":
                        where_clauses.append("data_type = 'ortho'")
                    where_sql = " AND ".join(where_clauses)
                    
                    cur.execute(f"""
                        SELECT id, filename, bucket, object_key, 
                               ST_X(location) as lon, ST_Y(location) as lat,
                               thumbnail_key, file_size
                        FROM image_metadata 
                        WHERE location IS NOT NULL AND {where_sql}
                        LIMIT 500
                    """)
                    for row in cur.fetchall():
                        all_coords.append((row[5], row[4]))
                        photos.append({
                            "id": row[0],
                            "filename": row[1],
                            "bucket": row[2],
                            "key": row[3],
                            "lon": row[4],
                            "lat": row[5],
                            "file_size": row[7] or 0,
                            "thumb_key": row[6],
                        })
                    
                    cur.execute(f"""
                        SELECT id, filename, bucket, object_key,
                               ST_AsGeoJSON(extent) as extent_geojson,
                               ST_X(ST_Centroid(extent)) as clon, ST_Y(ST_Centroid(extent)) as clat,
                               resolution, file_size, thumbnail_key
                        FROM image_metadata 
                        WHERE extent IS NOT NULL AND {where_sql}
                        LIMIT 100
                    """)
                    for row in cur.fetchall():
                        if row[5] and row[6]:
                            all_coords.append((row[6], row[5]))
                        orthos.append({
                            "id": row[0],
                            "filename": row[1],
                            "bucket": row[2],
                            "key": row[3],
                            "geometry": json.loads(row[4]),
                            "resolution": row[7],
                            "file_size": row[8] or 0,
                            "thumb_key": row[9],
                        })
                    
                    cur.close()
                    conn.close()
                except Exception as e:
                    st.warning(f"데이터 로드 실패: {e}")
            
            # 지도 중심 및 줌 계산 (데이터 범위 기반)
            if all_coords:
                lats = [c[0] for c in all_coords]
                lons = [c[1] for c in all_coords]
                center_lat = sum(lats) / len(lats)
                center_lon = sum(lons) / len(lons)
                
                # 범위에 맞는 줌 레벨 계산
                lat_range = max(lats) - min(lats)
                lon_range = max(lons) - min(lons)
                max_range = max(lat_range, lon_range)
                
                if max_range < 0.01:
                    zoom = 15
                elif max_range < 0.1:
                    zoom = 12
                elif max_range < 1:
                    zoom = 10
                elif max_range < 5:
                    zoom = 8
                else:
                    zoom = 6
            else:
                center_lat, center_lon, zoom = 36.5, 127.5, 7  # 기본값 (대한민국)
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
            
            # 사진 마커 추가
            for photo in photos:
                # presigned URL 생성 (STAC/레거시 모두 동일)
                try:
                    original_url = get_presigned_url(photo["bucket"], photo["key"], expires_in=3600) if photo.get("key") else ""
                except:
                    original_url = ""
                
                # 썸네일 URL (thumb_key가 있으면 사용, 없으면 원본)
                thumb_key = photo.get("thumb_key") or photo.get("key", "")
                try:
                    thumb_url = get_presigned_url(photo["bucket"], thumb_key, expires_in=3600) if thumb_key else ""
                except:
                    thumb_url = ""
                
                thumb_html = ""
                if thumb_url:
                    if original_url:
                        thumb_html = f'<a href="{original_url}" target="_blank"><img src="{thumb_url}" style="max-width:200px;max-height:150px;margin-bottom:8px;border-radius:4px;cursor:pointer;" title="클릭하면 원본 열기"></a><br>'
                    else:
                        thumb_html = f'<img src="{thumb_url}" style="max-width:200px;max-height:150px;margin-bottom:8px;border-radius:4px;"><br>'
                
                size_mb = photo["file_size"] / (1024 * 1024)
                popup_html = f"""
                {thumb_html}
                <b>{photo['filename']}</b><br>
                📦 {photo['bucket']}<br>
                💾 {size_mb:.1f} MB
                """
                folium.Marker(
                    location=[photo["lat"], photo["lon"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color="blue", icon="camera", prefix="fa")
                ).add_to(m)
            
            # 정사영상 폴리곤 추가
            for ortho in orthos:
                # presigned URL 생성 (STAC/레거시 모두 동일)
                try:
                    original_url = get_presigned_url(ortho["bucket"], ortho["key"], expires_in=3600) if ortho.get("key") else ""
                except:
                    original_url = ""
                
                # 썸네일 URL (thumb_key가 있으면 사용)
                thumb_key = ortho.get("thumb_key", "")
                try:
                    thumb_url = get_presigned_url(ortho["bucket"], thumb_key, expires_in=3600) if thumb_key else ""
                except:
                    thumb_url = ""
                
                thumb_html = ""
                if thumb_url:
                    if original_url:
                        thumb_html = f'<a href="{original_url}" target="_blank"><img src="{thumb_url}" style="max-width:200px;max-height:150px;margin-bottom:8px;border-radius:4px;cursor:pointer;" title="클릭하면 다운로드"></a><br>'
                    else:
                        thumb_html = f'<img src="{thumb_url}" style="max-width:200px;max-height:150px;margin-bottom:8px;border-radius:4px;"><br>'
                
                res_str = f"{ortho['resolution']:.2f}m" if ortho.get("resolution") else "N/A"
                popup_html = f"""
                {thumb_html}
                <b>{ortho['filename']}</b><br>
                📦 {ortho['bucket']}<br>
                📏 해상도: {res_str}<br>
                💾 {ortho['file_size'] / (1024*1024):.1f} MB
                """
                folium.GeoJson(
                    ortho["geometry"],
                    style_function=lambda x: {
                        "fillColor": "#3388ff",
                        "color": "#3388ff",
                        "weight": 2,
                        "fillOpacity": 0.3
                    },
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)
            
            # 지도 표시
            st_folium(m, width=None, height=600, returned_objects=[])
            
            if not photos and not orthos:
                st.info("📭 표시할 데이터가 없습니다. 오른쪽 패널에서 인덱싱을 실행하세요.")
            
        except ImportError:
            st.error("📦 folium 또는 streamlit-folium이 설치되지 않았습니다.")


# ============================================
# Tab 3: Training Lab
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
            
            st.markdown("##### 📁 데이터 경로 설정")
            data_dirs = get_data_directories()
            
            if task_type == "Change Detection":
                current_data_dir = config.get("data", {}).get("local", {}).get("root_dir", "./data/change_detection")
                # 폴더 이름만 추출 (./data/xxx -> xxx)
                default_data_folder = Path(current_data_dir).name
                
                selected_data_folder = st.selectbox(
                    "학습 데이터 폴더 (./data/)",
                    data_dirs,
                    index=data_dirs.index(default_data_folder) if default_data_folder in data_dirs else 0
                )
                custom_data_path = st.text_input("상세 경로 (직접 입력)", value=f"./data/{selected_data_folder}")
                
                st.markdown("##### 🧠 학습 파라미터 수정")
                epochs = st.number_input("Epochs", value=config.get("training", {}).get("epochs", 50), min_value=1)
                batch_size = st.number_input("Batch Size", value=config.get("training", {}).get("batch_size", 8), min_value=1)
                lr = st.number_input("Learning Rate", value=config.get("training", {}).get("optimizer", {}).get("lr", 0.001), format="%.5f")
                
                overrides = f"data.local.root_dir={custom_data_path} training.epochs={epochs} training.batch_size={batch_size} training.optimizer.lr={lr}"
                script = "src/training/train_cd.py"
            else:
                current_data_path = config.get("data", {}).get("source_path", "./data/nvs_project")
                default_data_folder = Path(current_data_path).name
                
                selected_data_folder = st.selectbox(
                    "학습 데이터 폴더 (./data/)",
                    data_dirs,
                    index=data_dirs.index(default_data_folder) if default_data_folder in data_dirs else 0
                )
                custom_data_path = st.text_input("상세 경로 (직접 입력)", value=f"./data/{selected_data_folder}")
                
                st.markdown("##### 🧠 학습 파라미터 수정")
                iterations = st.number_input("Iterations", value=config.get("training", {}).get("iterations", 30000), min_value=100, step=1000)
                sh_degree = st.number_input("SH Degree", value=config.get("model", {}).get("sh_degree", 3), min_value=0, max_value=3)
                
                overrides = f"data.source_path={custom_data_path} training.iterations={iterations} model.sh_degree={sh_degree}"
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
                    f"http://{st.session_state.public_ip}:{MLFLOW_PORT}/#/experiments/0/runs/{selected_run}",
                    use_container_width=True
                )
        else:
            st.info("학습 결과가 없습니다.")
    
    with col2:
        st.subheader("🔗 MLflow UI")
        
        # 세션 초기화된 IP를 사용 (하드코딩된 PUBLIC_IP 대신)
        mlflow_url = f"http://{st.session_state.public_ip}:{MLFLOW_PORT}"
        
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
                    
                    # 서브프로세스 환경변수에 PUBLIC_IP 전달
                    env = os.environ.copy()
                    env["PUBLIC_IP"] = st.session_state.public_ip
                    
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
                    
                    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300, env=env)
                    
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
