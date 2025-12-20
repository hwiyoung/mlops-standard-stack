"""
변화탐지(Change Detection) 모델 정의
Segmentation Models PyTorch (SMP) 라이브러리 활용
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("segmentation_models_pytorch가 필요합니다: pip install segmentation-models-pytorch")


class ChangeDetectionModel(nn.Module):
    """
    위성 이미지 변화탐지 모델
    Pre/Post 이미지를 입력받아 변화 마스크 예측
    
    지원 아키텍처: unet, fpn, deeplabv3plus, pspnet, manet, linknet, pan
    """
    
    ARCHITECTURES = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "pspnet": smp.PSPNet,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
        "pan": smp.PAN,
    }
    
    def __init__(
        self,
        architecture: str = "unet",
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 6,  # Pre(3) + Post(3)
        num_classes: int = 2,
        activation: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            architecture: 모델 아키텍처 (unet, fpn, deeplabv3plus 등)
            encoder_name: 인코더 백본 (resnet50, efficientnet-b0 등)
            encoder_weights: 사전학습 가중치 (imagenet, None)
            in_channels: 입력 채널 수 (Pre + Post = 6)
            num_classes: 출력 클래스 수 (변화/비변화 = 2)
            activation: 출력 활성화 함수 (None, sigmoid, softmax)
        """
        super().__init__()
        
        self.architecture = architecture.lower()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        if self.architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"지원하지 않는 아키텍처: {architecture}. "
                f"지원 목록: {list(self.ARCHITECTURES.keys())}"
            )
        
        # SMP 모델 생성
        model_class = self.ARCHITECTURES[self.architecture]
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs
        )
        
        print(f"✅ 모델 생성: {architecture.upper()} (encoder={encoder_name}, in={in_channels}, out={num_classes})")
    
    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pre_image: 변화 전 이미지 [B, C, H, W]
            post_image: 변화 후 이미지 [B, C, H, W]
        
        Returns:
            변화 마스크 logits [B, num_classes, H, W]
        """
        # Pre/Post 이미지를 채널 방향으로 결합
        x = torch.cat([pre_image, post_image], dim=1)
        return self.model(x)
    
    def predict(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
        """
        예측 (argmax 적용)
        
        Returns:
            예측 마스크 [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(pre_image, post_image)
            return logits.argmax(dim=1)
    
    def get_encoder_params(self):
        """인코더 파라미터 반환 (fine-tuning용)"""
        return self.model.encoder.parameters()
    
    def get_decoder_params(self):
        """디코더 파라미터 반환"""
        return self.model.decoder.parameters()
    
    def freeze_encoder(self):
        """인코더 가중치 동결"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("🔒 인코더 가중치 동결됨")
    
    def unfreeze_encoder(self):
        """인코더 가중치 해제"""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print("🔓 인코더 가중치 해제됨")


def build_model(config) -> ChangeDetectionModel:
    """
    Config 객체에서 모델 빌드
    
    Args:
        config: Config 객체 또는 딕셔너리
    
    Returns:
        ChangeDetectionModel 인스턴스
    """
    # Config 객체 또는 딕셔너리 처리
    if hasattr(config, 'model'):
        model_cfg = config.model
        arch = model_cfg.architecture
        encoder_name = model_cfg.encoder.name
        encoder_weights = model_cfg.encoder.weights
        in_channels = model_cfg.in_channels
        num_classes = model_cfg.num_classes
        activation = getattr(model_cfg, 'activation', None)
    else:
        model_cfg = config.get("model", config)
        arch = model_cfg.get("architecture", "unet")
        encoder_cfg = model_cfg.get("encoder", {})
        encoder_name = encoder_cfg.get("name", "resnet50")
        encoder_weights = encoder_cfg.get("weights", "imagenet")
        in_channels = model_cfg.get("in_channels", 6)
        num_classes = model_cfg.get("num_classes", 2)
        activation = model_cfg.get("activation", None)
    
    return ChangeDetectionModel(
        architecture=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        activation=activation
    )


# 손실 함수 빌더
def build_loss(config) -> nn.Module:
    """
    Config에서 손실 함수 빌드
    """
    if hasattr(config, 'training'):
        loss_cfg = config.training.loss
        loss_name = loss_cfg.name
        class_weights = getattr(loss_cfg, 'class_weights', None)
    else:
        loss_cfg = config.get("training", {}).get("loss", {})
        loss_name = loss_cfg.get("name", "cross_entropy")
        class_weights = loss_cfg.get("class_weights", None)
    
    if loss_name == "cross_entropy":
        weight = torch.tensor(class_weights) if class_weights else None
        return nn.CrossEntropyLoss(weight=weight)
    
    elif loss_name == "dice":
        return smp.losses.DiceLoss(mode="multiclass")
    
    elif loss_name == "focal":
        return smp.losses.FocalLoss(mode="multiclass")
    
    elif loss_name == "combined":
        ce_loss = nn.CrossEntropyLoss()
        dice_loss = smp.losses.DiceLoss(mode="multiclass")
        
        class CombinedLoss(nn.Module):
            def __init__(self, ce, dice, ce_weight=0.5):
                super().__init__()
                self.ce = ce
                self.dice = dice
                self.ce_weight = ce_weight
            
            def forward(self, pred, target):
                return self.ce_weight * self.ce(pred, target) + (1 - self.ce_weight) * self.dice(pred, target)
        
        return CombinedLoss(ce_loss, dice_loss)
    
    else:
        raise ValueError(f"지원하지 않는 손실 함수: {loss_name}")


if __name__ == "__main__":
    # 테스트
    print("=== 모델 테스트 ===")
    
    model = ChangeDetectionModel(
        architecture="unet",
        encoder_name="resnet34",
        in_channels=6,
        num_classes=2
    )
    
    # 더미 입력
    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 256, 256)
    
    # Forward
    output = model(pre, post)
    print(f"Input: pre={pre.shape}, post={post.shape}")
    print(f"Output: {output.shape}")
    
    # 예측
    pred = model.predict(pre, post)
    print(f"Prediction: {pred.shape}")
