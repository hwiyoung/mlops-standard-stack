#!/usr/bin/env python3
"""
MLflow + MinIO 통합 테스트 스크립트
- 랜덤 메트릭 로깅
- 텍스트 파일 아티팩트 업로드
"""

import os
import random
import tempfile
from datetime import datetime

import mlflow


def test_mlflow_tracking():
    """MLflow 트래킹 및 아티팩트 업로드 테스트"""
    
    print("=" * 50)
    print("MLflow + MinIO 통합 테스트")
    print("=" * 50)
    
    # 현재 트래킹 URI 확인
    tracking_uri = mlflow.get_tracking_uri()
    print(f"📡 Tracking URI: {tracking_uri}")
    
    # 실험 생성 또는 가져오기
    experiment_name = "mlops-integration-test"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ 새 실험 생성: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"📋 기존 실험 사용: {experiment_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    
    # MLflow run 시작
    with mlflow.start_run(run_name=f"test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"\n🚀 Run 시작: {run_id}")
        
        # ============================================
        # 1. 랜덤 메트릭 로깅
        # ============================================
        print("\n📊 메트릭 로깅 중...")
        
        # 랜덤 하이퍼파라미터 설정
        params = {
            "learning_rate": round(random.uniform(0.001, 0.1), 4),
            "batch_size": random.choice([16, 32, 64, 128]),
            "epochs": random.randint(10, 100),
            "model_type": "UNet",
            "optimizer": random.choice(["Adam", "SGD", "AdamW"])
        }
        mlflow.log_params(params)
        print(f"   ✅ 파라미터 로깅: {params}")
        
        # 랜덤 메트릭 (학습 시뮬레이션)
        for epoch in range(1, 6):
            metrics = {
                "train_loss": round(1.0 - (epoch * 0.15) + random.uniform(-0.05, 0.05), 4),
                "val_loss": round(1.0 - (epoch * 0.12) + random.uniform(-0.08, 0.08), 4),
                "accuracy": round(0.5 + (epoch * 0.08) + random.uniform(-0.03, 0.03), 4),
                "iou_score": round(0.4 + (epoch * 0.1) + random.uniform(-0.05, 0.05), 4)
            }
            mlflow.log_metrics(metrics, step=epoch)
            print(f"   ✅ Epoch {epoch}: loss={metrics['train_loss']:.4f}, acc={metrics['accuracy']:.4f}")
        
        # ============================================
        # 2. 아티팩트 업로드 (MinIO)
        # ============================================
        print("\n📦 아티팩트 업로드 중 (MinIO)...")
        
        # 임시 텍스트 파일 생성
        with tempfile.TemporaryDirectory() as tmpdir:
            # 실험 설정 파일
            config_path = os.path.join(tmpdir, "experiment_config.txt")
            with open(config_path, "w") as f:
                f.write("=" * 50 + "\n")
                f.write("MLOps Integration Test Configuration\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                f.write("Parameters:\n")
                for k, v in params.items():
                    f.write(f"  - {k}: {v}\n")
                f.write("\nThis file was auto-generated to test MinIO artifact storage.\n")
            
            mlflow.log_artifact(config_path)
            print(f"   ✅ 업로드 완료: experiment_config.txt")
            
            # 가상 모델 정보 파일
            model_info_path = os.path.join(tmpdir, "model_info.json")
            with open(model_info_path, "w") as f:
                import json
                model_info = {
                    "model_name": "UNet-ChangeDetection",
                    "input_size": [512, 512],
                    "num_classes": 2,
                    "backbone": "resnet50",
                    "pretrained": True,
                    "created_at": datetime.now().isoformat()
                }
                json.dump(model_info, f, indent=2)
            
            mlflow.log_artifact(model_info_path)
            print(f"   ✅ 업로드 완료: model_info.json")
        
        # ============================================
        # 3. 태그 설정
        # ============================================
        mlflow.set_tags({
            "project": "mlops-standard-stack",
            "task": "change-detection",
            "environment": "development",
            "test_type": "integration"
        })
        print("\n🏷️  태그 설정 완료")
        
        print(f"\n{'=' * 50}")
        print("✅ 테스트 완료!")
        print(f"{'=' * 50}")
        print(f"\n📌 결과 확인:")
        print(f"   - MLflow UI: http://localhost:5000")
        print(f"   - Run ID: {run_id}")
        print(f"   - Experiment: {experiment_name}")


if __name__ == "__main__":
    test_mlflow_tracking()
