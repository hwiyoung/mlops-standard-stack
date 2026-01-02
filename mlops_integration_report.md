# Strategic MLOps Integration Report

본 보고서는 프로젝트의 현재 상태를 바탕으로 제안된 MLOps 도구들(DVC, PyTorch Lightning 등)의 적용 타당성과 구체적인 활용 방안을 검토한 결과입니다.

---

## 1. DVC (Data Version Control)
**상태**: 현재 MinIO를 저장소로 쓰지만, 데이터 자체의 "버전" 관리는 수동으로 이루어짐.

- **필요성**: ✅ **매우 높음**
- **적용 방안**:
    - **Data-to-Code 연결**: 특정 학습 실험에 쓰인 데이터셋의 상태를 Git 커밋과 연결.
    - **MinIO Remote**: MinIO를 DVC의 원격 저장소(Remote)로 설정하여 `dvc push/pull` 연동.
    - **효과**: "예전에 IoU 0.8 나왔던 데이터셋 그대로 다시 학습해보고 싶다" 할 때 `git checkout` 만으로 데이터 세팅 완료 가능.

---

## 2. PyTorch Lightning
**상태**: `train_cd.py`에 수동 학습 루프, 디바이스 설정, 메트릭 계산 로직이 섞여 있음.

- **필요성**: ✅ **매우 높음**
- **적용 방안**:
    - **Refactoring**: `LightningModule`로 모델/손실함수/최적화 로직 분리.
    - **Boilerplate 제거**: 장치 관리(GPU/CPU/MPS), Mixed Precision(FP16), 로깅(MLflow) 코드를 Lightning이 자동 처리.
    - **효과**: 코드 가독성 2배 향상, 성능 측정 자동화, 멀티 GPU 확장 용이.

---

## 3. Great Expectations
**상태**: 데이터 업로드 및 인덱싱 단계에서 엄격한 품질 검증 로직이 부재함.

- **필요성**: ✅ **추천 (데이터 품질 보증)**
- **적용 방안**:
    - **Data Pipeline Validation**: 인덱싱(`metadata_extractor`) 전후에 데이터 검증 수행.
    - **검증 항목**: "모든 TIF는 읽기 가능한가?", "해상도가 1.0m 이하인가?", "GPS 좌표가 유효한 범위인가?" 등.
    - **효과**: 학습 도중 파일 손상이나 잘못된 데이터로 인해 발생할 수 있는 런타임 에러 사전 차단.

---

## 4. ydata-profiling (formerly pandas-profiling)
**상태**: 본 프로젝트는 이미지 데이터 위주임.

- **필요성**: ⚠️ **보통 (메타데이터 분석용)**
- **적용 방안**:
    - **Metadata EDA**: PostGIS의 `image_metadata` 테이블을 Pandas로 읽어와 분석 보고서 생성.
    - **활용**: 지역별 데이터 분포, 촬영 시기 분포, 용량/해상도 상관관계 등을 한눈에 파악.
    - **효과**: 데이터셋의 전체적인 통계 특성을 파악하여 데이터 수집 전략 수립에 활용 가능.

---

---

## 🚀 통합 로드맵 (Unified Roadmap)

위 도구들의 도입 계획을 기존 서비스 로드맵과 통합하여 다음과 같은 우선순위로 추진합니다.

### 1단계: 인프라 및 기반 다지기 (Core & Engineering)
- [ ] **PyTorch Lightning 전환**: 기존 학습 루프의 구조화 및 GPU 확장성 확보.
- [ ] **DVC 버전 관리**: MinIO를 원격 저장소로 설정하여 데이터-코드 재현성 확보.
- [ ] **실시간 인덱싱**: `watch_bucket.py` 구현을 통한 데이터 업로드 즉시 자동 반영.

### 2단계: 품질 및 시각화 고도화 (Quality & Visualization)
- [ ] **Great Expectations 도입**: 인덱싱 파이프라인에 데이터 무결성 검증 추가.
- [ ] **고급 GIS 도구**: 레이어 투명도, 시계열 비교(Swipe) 툴 등 지도 UI 개선.
- [ ] **3D 가시화**: 지도 위에서 3DGS 결과(Point Cloud)를 직접 렌더링.

### 3단계: 분석 및 확장성 최적화 (Scalability & Ops)
- [ ] **ydata-profiling 분석**: 메타데이터 통계 자동화 및 데이터셋 인사이트 도출.
- [ ] **분산 작업 큐**: Celery/Redis 기반 대규모 데이터 배치 처리.
- [ ] **보안 및 권한**: 사용자 인증(RBAC) 및 프로젝트별 격리.

---

> [!IMPORTANT]
> **우선 권장 사항**: 다음 작업으로 **PyTorch Lightning**을 통한 학습 코드 리팩토링을 시작하여 알고리즘 개발의 생산성을 높이는 것을 추천합니다.
