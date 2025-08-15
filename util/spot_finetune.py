#!/usr/bin/env python3
"""
Spot 로봇 GR00T N1.5-3B 파인튜닝 스크립트 (최종 수정 버전)

사용법:
    python spot_finetune.py \
        --dataset_path ./demo_data/spot-lerobot-task0 \
        --output_dir ./spot_finetuned_model \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 1e-5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from dataclasses import dataclass
from transformers import TrainingArguments

# GR00T 관련 import
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.utils.peft import get_lora_model


@dataclass
class SpotFinetuneConfig:
    """Spot 로봇 GR00T 파인튜닝 설정"""
    
    # 데이터셋 설정
    dataset_path: str = "./demo_data/spot-lerobot-task0"
    """Spot 데이터셋 경로"""
    
    # 모델 설정
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """사전 훈련된 모델 경로"""
    output_dir: str = "./spot_finetuned_model"
    """훈련된 모델 저장 디렉토리"""
    
    # 훈련 설정
    num_epochs: int = 3
    """훈련 에포크 수"""
    batch_size: int = 1
    """배치 크기"""
    learning_rate: float = 1e-5
    """학습률"""
    warmup_ratio: float = 0.1
    """워밍업 비율"""
    gradient_accumulation_steps: int = 16
    """그래디언트 누적 스텝 수"""
    
    # 하드웨어 설정
    num_gpus: int = 1
    """사용할 GPU 수"""
    
    # LoRA 설정 (선택사항)
    use_lora: bool = False
    """LoRA 사용 여부"""
    lora_rank: int = 16
    """LoRA 랭크"""
    lora_alpha: int = 32
    """LoRA 알파"""
    lora_dropout: float = 0.1
    """LoRA 드롭아웃"""
    
    # 모델 튜닝 설정
    tune_visual: bool = False
    """비주얼 인코더 튜닝 여부"""
    tune_llm: bool = False
    """언어 모델 튜닝 여부"""
    tune_projector: bool = True
    """프로젝터 튜닝 여부"""
    tune_diffusion_model: bool = True
    """디퓨전 모델 튜닝 여부"""
    
    # 기타 설정
    save_steps: int = 500
    """체크포인트 저장 간격"""
    logging_steps: int = 10
    """로깅 간격"""
    max_steps: int = 5000
    """최대 훈련 스텝 수"""
    
    # 비디오 백엔드
    video_backend: str = "decord"
    """비디오 백엔드 (decord 또는 torchvision_av)"""


class SpotDataConfig:
    """Spot 로봇을 위한 데이터 설정"""
    
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.lin_vel",
        "state.ang_vel", 
        "state.gravity",
        "state.command",
        "state.joint_pos",
        "state.joint_vel",
        "state.prev_action"
    ]
    action_keys = ["action.fl_leg", "action.fr_leg", "action.rl_leg", "action.rr_leg"]
    language_keys = [
        "annotation.human.action.task_description",
    ]
    
    # 관찰 및 액션 인덱스 (실제 데이터에 맞게 수정)
    observation_indices = [0, 1, 2, 3, 4, 5, 6]  # 7개 상태 키 (시간적 오프셋)
    action_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 16개 시간 스텝
    
    # 정규화 모드 설정
    state_normalization_modes = {
        "state.lin_vel": "min_max",
        "state.ang_vel": "min_max", 
        "state.gravity": "min_max",
        "state.command": "min_max",
        "state.joint_pos": "min_max",
        "state.joint_vel": "min_max",
        "state.prev_action": "min_max",
    }
    
    action_normalization_modes = {
        "action.fl_leg": "min_max",
        "action.fr_leg": "min_max",
        "action.rl_leg": "min_max",
        "action.rr_leg": "min_max",
    }
    
    def modality_config(self):
        """모달리티 설정 반환"""
        from gr00t.data.dataset import ModalityConfig
        
        video_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.video_keys,
        )
        
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,  # 16개 시간 스텝 사용
            modality_keys=self.action_keys,
        )
        
        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.language_keys,
        )
        
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs
    
    def transform(self):
        """데이터 변환 설정 반환"""
        from gr00t.experiment.data_config import (
            VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy,
            StateActionToTensor, StateActionTransform, ConcatTransform, GR00TTransform,
            ComposedModalityTransform
        )
        from gr00t.data.transform.base import ModalityTransform
        
        # 올바른 커스텀 텐서 변환 클래스
        class CustomVideoToTensor(ModalityTransform):
            def __init__(self, apply_to):
                super().__init__(apply_to=apply_to)
            
            def apply(self, data):
                import torch
                for key in self.apply_to:
                    if key in data:
                        # numpy → tensor 변환 (해상도 체크 없이)
                        data[key] = torch.from_numpy(data[key]).float() / 255.0
                        data[key] = data[key].permute(0, 3, 1, 2)  # [T, H, W, C] → [T, C, H, W]
                return data
        
        transforms = [
            # 비디오 변환 - 올바른 순서
            CustomVideoToTensor(apply_to=self.video_keys),  # 1. 텐서 변환
            VideoResize(apply_to=self.video_keys, height=256, width=256, interpolation="linear"),  # 2. 리사이즈 (320x240 → 256x256)
            VideoCrop(apply_to=self.video_keys, scale=0.95),  # 3. 크롭 (256x256에서)
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            
            # 상태 변환
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            
            # 액션 변환
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            
            # 연결 변환
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            
            # GR00T 변환
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,  # GR00T 모델 호환성을 위해 32로 설정
            ),
        ]
        
        return ComposedModalityTransform(transforms=transforms)


def main(config: SpotFinetuneConfig):
    """메인 훈련 함수"""
    
    print("="*80)
    print("Spot 로봇 GR00T 파인튜닝 시작")
    print("="*80)
    
    # 1. 데이터 설정
    print("\n1. 데이터 설정 로드 중...")
    data_config = SpotDataConfig()
    modality_configs = data_config.modality_config()
    transforms = data_config.transform()
    
    # 2. 데이터셋 로드
    print(f"\n2. 데이터셋 로드 중: {config.dataset_path}")
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # 새로운 embodiment로 설정
        video_backend=config.video_backend,
    )
    
    print(f"   데이터셋 크기: {len(dataset)} 샘플")
    print(f"   에피소드 수: {len(dataset.trajectory_lengths)}")
    
    # 3. 모델 로드
    print(f"\n3. 모델 로드 중: {config.base_model_path}")
    model = GR00T_N1_5.from_pretrained(
        config.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto" if config.num_gpus > 1 else None,
    )

    # 4. LoRA 설정 (선택사항)
    if config.use_lora:
        print("\n4. LoRA 설정 적용 중...")
        model = get_lora_model(
            model=model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=True,
        )
    
    # 5. 훈련 설정
    print("\n5. 훈련 설정 구성 중...")
    
    # 훈련 인수 설정 - GR00T 스타일로 수정
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name="spot_finetune",
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=1e-5,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        num_train_epochs=config.num_epochs,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=5,
        report_to="tensorboard",
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    
    # 6. 훈련 실행
    print("\n6. 훈련 시작...")
    experiment = TrainRunner(
        model=model,
        training_args=training_args,  # args -> training_args로 수정
        train_dataset=dataset,  # 매개변수 순서 변경
    )
    
    # 훈련 실행
    experiment.train()
    
    print("\n" + "="*80)
    print("✅ Spot 로봇 파인튜닝 완료!")
    print(f"모델 저장 위치: {config.output_dir}")
    print("="*80)


if __name__ == "__main__":
    # 명령행 인수 파싱
    config = tyro.cli(SpotFinetuneConfig)
    
    # 출력 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 메인 함수 실행
    main(config)