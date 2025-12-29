"""
Configuration 모듈
YAML 설정 파일을 로드하고 관리하는 유틸리티
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    YAML 설정 파일을 로드하고 관리하는 클래스
    
    사용법:
        config = Config.from_yaml("configs/train_cd.yaml")
        print(config.training.epochs)
        print(config["model"]["encoder"]["name"])
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Args:
            config_dict: 설정 딕셔너리
        """
        self._config = config_dict
        self._make_attributes(config_dict)
    
    def _make_attributes(self, d: Dict[str, Any], prefix: str = "") -> None:
        """딕셔너리를 객체 속성으로 변환"""
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """
        YAML 파일에서 Config 객체 생성
        
        Args:
            yaml_path: YAML 파일 경로
        
        Returns:
            Config 객체
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """딕셔너리에서 Config 객체 생성"""
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Config를 딕셔너리로 변환"""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        키로 값 조회 (점 표기법 지원)
        
        Args:
            key: 조회할 키 (예: "model.encoder.name")
            default: 기본값
        
        Returns:
            조회된 값 또는 기본값
        """
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        키로 값 설정 (점 표기법 지원)
        
        Args:
            key: 설정할 키 (예: "training.epochs")
            value: 설정할 값
        """
        keys = key.split(".")
        d = self._config
        
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        
        d[keys[-1]] = value
        
        # 속성도 업데이트
        self._make_attributes(self._config)
    
    def merge(self, override_dict: Dict[str, Any]) -> "Config":
        """
        다른 딕셔너리와 병합 (override)
        
        Args:
            override_dict: 덮어쓸 딕셔너리
        
        Returns:
            새로운 Config 객체
        """
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(self._config, override_dict)
        return Config(merged)
    
    def save(self, yaml_path: Union[str, Path]) -> None:
        """
        설정을 YAML 파일로 저장
        
        Args:
            yaml_path: 저장할 경로
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일 접근 지원"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """in 연산자 지원"""
        return key in self._config
    
    def __repr__(self) -> str:
        return f"Config({self._config})"
    
    def __str__(self) -> str:
        return yaml.dump(self._config, default_flow_style=False, allow_unicode=True)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    설정 파일 로드 헬퍼 함수
    
    Args:
        config_path: YAML 파일 경로
    
    Returns:
        Config 객체
    """
    return Config.from_yaml(config_path)


def parse_args_with_config():
    """
    --config 인자를 파싱하는 ArgumentParser 생성
    
    Returns:
        (args, config) 튜플
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--override", "-o",
        type=str,
        nargs="*",
        default=[],
        help="Override config values (e.g., training.epochs=100 model.encoder.name=resnet34)"
    )
    
    args = parser.parse_args()
    
    # Config 로드
    config = load_config(args.config)
    
    # Override 처리
    for override in args.override:
        if "=" in override:
            key, value = override.split("=", 1)
            # 타입 추론
            try:
                value = eval(value)  # int, float, bool, list 등
            except:
                pass  # 문자열로 유지
            config.set(key, value)
    
    return args, config


# 편의를 위한 전역 Config 인스턴스
_global_config: Optional[Config] = None


def get_config() -> Optional[Config]:
    """전역 Config 반환"""
    return _global_config


def set_config(config: Config) -> None:
    """전역 Config 설정"""
    global _global_config
    _global_config = config


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("=== Change Detection Config 테스트 ===")
    print("=" * 60)
    
    config = load_config("configs/train_cd.yaml")
    
    print(f"Experiment name: {config.experiment.name}")
    print(f"Model architecture: {config.model.architecture}")
    print(f"Encoder: {config.model.encoder.name}")
    print(f"Learning rate: {config.training.optimizer.lr}")
    print(f"Epochs: {config.training.epochs}")
    
    print("\n=== 점 표기법 조회 ===")
    print(f"model.encoder.name: {config.get('model.encoder.name')}")
    print(f"data.minio.bucket_raw: {config.get('data.minio.bucket_raw')}")
    
    # NVS Config 테스트
    print("\n" + "=" * 60)
    print("=== NVS (3D Gaussian Splatting) Config 테스트 ===")
    print("=" * 60)
    
    try:
        nvs_config = load_config("configs/train_nvs.yaml")
        
        print(f"Experiment name: {nvs_config.experiment.name}")
        print(f"Method: {nvs_config.experiment.tags.method}")
        print(f"\n[Model]")
        print(f"  SH Degree: {nvs_config.model.sh_degree}")
        print(f"  Opacity Init: {nvs_config.model.gaussian.opacity_init}")
        print(f"\n[Training]")
        print(f"  Iterations: {nvs_config.training.iterations}")
        print(f"  Save Iterations: {nvs_config.training.save_iterations}")
        print(f"\n[Densification]")
        print(f"  Enabled: {nvs_config.training.densification.enabled}")
        print(f"  Interval: {nvs_config.training.densification.interval}")
        print(f"  Start: {nvs_config.training.densification.start_iteration}")
        print(f"  End: {nvs_config.training.densification.end_iteration}")
        print(f"\n[Pipeline]")
        print(f"  Output Dir: {nvs_config.pipeline.output_dir}")
        print(f"  Save Video: {nvs_config.pipeline.save.video}")
    except FileNotFoundError:
        print("NVS config 파일이 없습니다: configs/train_nvs.yaml")
    
    print("\n=== 딕셔너리 변환 ===")
    print(config.to_dict().keys())
