#!/usr/bin/env python3
"""
Spot 로봇 GR00T 추론 스크립트

사용법:
    python util/spot_inference.py \
        --model_path ./spot_finetuned_model_corrected \
        --dataset_path ./demo_data/spot-lerobot-task0 \
        --output_dir ./spot_inference_results \
        --num_samples 5
"""

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# GR00T 관련 import
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import VideoToTensor, VideoResize, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform


@dataclass
class SpotInferenceConfig:
    """Spot 로봇 추론 설정"""
    
    # 모델 설정
    model_path: str = "./spot_finetuned_model_corrected"
    """파인튜닝된 모델 경로"""
    
    # 데이터 설정
    dataset_path: str = "./demo_data/spot-lerobot-task0"
    """테스트 데이터셋 경로"""
    output_dir: str = "./spot_inference_results"
    """결과 저장 디렉토리"""
    
    # 추론 설정
    num_samples: int = 5
    """추론할 샘플 수"""
    save_predictions: bool = True
    """예측 결과 저장 여부"""
    visualize: bool = True
    """시각화 여부"""
    
    # 하드웨어 설정
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """사용할 디바이스"""


def create_modality_config():
    """모달리티 설정 생성"""
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.lin_vel", "state.ang_vel", "state.gravity", "state.command", 
                          "state.joint_pos", "state.joint_vel", "state.prev_action"],
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=["action.fl_leg", "action.fr_leg", "action.rl_leg", "action.rr_leg"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }


def create_modality_transform():
    """모달리티 변환 설정 생성 (성공한 버전)"""
    return ComposedModalityTransform(transforms=[
        # 비디오 변환
        VideoToTensor(apply_to=["video.ego_view"]),
        VideoResize(apply_to=["video.ego_view"], height=256, width=256, interpolation="linear"),
        VideoToNumpy(apply_to=["video.ego_view"]),
        
        # 상태 변환
        StateActionToTensor(apply_to=["state.lin_vel", "state.ang_vel", "state.gravity", "state.command", 
                                    "state.joint_pos", "state.joint_vel", "state.prev_action"]),
        StateActionTransform(
            apply_to=["state.lin_vel", "state.ang_vel", "state.gravity", "state.command", 
                     "state.joint_pos", "state.joint_vel", "state.prev_action"],
            normalization_modes={
                "state.lin_vel": "min_max",
                "state.ang_vel": "min_max", 
                "state.gravity": "min_max",
                "state.command": "min_max",
                "state.joint_pos": "min_max",
                "state.joint_vel": "min_max",
                "state.prev_action": "min_max",
            },
        ),
        
        # 액션 변환
        StateActionToTensor(apply_to=["action.fl_leg", "action.fr_leg", "action.rl_leg", "action.rr_leg"]),
        StateActionTransform(
            apply_to=["action.fl_leg", "action.fr_leg", "action.rl_leg", "action.rr_leg"],
            normalization_modes={
                "action.fl_leg": "min_max",
                "action.fr_leg": "min_max",
                "action.rl_leg": "min_max",
                "action.rr_leg": "min_max",
            },
        ),
        
        # 연결 변환
        ConcatTransform(
            video_concat_order=["video.ego_view"],
            state_concat_order=["state.lin_vel", "state.ang_vel", "state.gravity", "state.command", 
                              "state.joint_pos", "state.joint_vel", "state.prev_action"],
            action_concat_order=["action.fl_leg", "action.fr_leg", "action.rl_leg", "action.rr_leg"],
        ),
        
        # GR00T 변환 (모델이 기대하는 형태로 변환)
        GR00TTransform(
            state_horizon=1,  # 단일 시점
            action_horizon=1,  # 단일 시점
            max_state_dim=64,
            max_action_dim=32,
        ),
    ])


def create_policy(config: SpotInferenceConfig):
    """정책 모델 생성"""
    print("정책 모델 생성 중...")
    
    modality_config = create_modality_config()
    modality_transform = create_modality_transform()
    
    policy = Gr00tPolicy(
        model_path=config.model_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=config.device,
    )
    
    return policy


def load_test_dataset(config: SpotInferenceConfig):
    """테스트 데이터셋 로드"""
    print(f"테스트 데이터셋 로드 중: {config.dataset_path}")
    
    modality_config = create_modality_config()
    
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_config,
        transforms=None,  # 변환은 정책에서만 적용
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend="decord",
    )
    
    print(f"데이터셋 크기: {len(dataset)} 샘플")
    return dataset


def run_inference(policy, dataset, config: SpotInferenceConfig):
    """추론 실행"""
    print(f"\n추론 시작 - {config.num_samples}개 샘플")
    
    results = []
    
    for i in range(min(config.num_samples, len(dataset))):
        print(f"\n샘플 {i+1}/{config.num_samples} 처리 중...")
        
        # 데이터 로드
        sample = dataset[i]
        
        # 데이터 구조 확인
        print(f"  샘플 키: {list(sample.keys())}")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"    {key}: type={type(value)}")
        
        # 정책 모델로 추론
        with torch.no_grad():
            try:
                # 데이터를 numpy 배열로 변환 (Gr00tPolicy가 기대하는 형태)
                numpy_sample = {}
                for key, value in sample.items():
                    if isinstance(value, np.ndarray):
                        numpy_sample[key] = value
                    elif isinstance(value, torch.Tensor):
                        numpy_sample[key] = value.cpu().numpy()
                    else:
                        numpy_sample[key] = np.array(value)
                
                # GPU로 이동
                device = "cuda" if torch.cuda.is_available() else "cpu"
                policy.model.to(device)
                
                predicted_actions = policy.get_action(numpy_sample)
                print("  추론 성공!")
                
                # 결과 저장
                result = {
                    "sample_id": i,
                    "predicted_actions": predicted_actions,
                    "error": None,
                }
                
                results.append(result)
                
                # 예측 결과 출력
                if predicted_actions.get("action_pred") is not None:
                    pred_shape = predicted_actions["action_pred"].shape
                    print(f"  예측 액션 shape: {pred_shape}")
                    print(f"  예측 액션 범위: [{predicted_actions['action_pred'].min():.3f}, {predicted_actions['action_pred'].max():.3f}]")
                else:
                    print(f"  예측 액션 키: {list(predicted_actions.keys())}")
                    for key, value in predicted_actions.items():
                        if hasattr(value, 'shape'):
                            print(f"    {key}: shape={value.shape}")
                        else:
                            print(f"    {key}: type={type(value)}")
                
            except Exception as e:
                print(f"  추론 실패: {e}")
                results.append({
                    "sample_id": i,
                    "error": str(e),
                    "predicted_actions": None,
                })
    
    return results


def save_results(results, config: SpotInferenceConfig):
    """결과 저장"""
    if not config.save_predictions:
        return
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # JSON으로 저장
    output_file = os.path.join(config.output_dir, "inference_results.json")
    
    # numpy 배열을 리스트로 변환
    serializable_results = []
    for result in results:
        serializable_result = {
            "sample_id": result["sample_id"],
            "error": result.get("error", None),
        }
        
        # predicted_actions 처리
        if result["predicted_actions"] is not None:
            serializable_actions = {}
            for key, value in result["predicted_actions"].items():
                if hasattr(value, 'cpu'):
                    value = value.cpu()
                if hasattr(value, 'numpy'):
                    value = value.numpy()
                if hasattr(value, 'tolist'):
                    serializable_actions[key] = value.tolist()
                else:
                    serializable_actions[key] = value
            serializable_result["predicted_actions"] = serializable_actions
        else:
            serializable_result["predicted_actions"] = None
        
        serializable_results.append(serializable_result)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n결과 저장됨: {output_file}")


def visualize_results(results, config: SpotInferenceConfig):
    """결과 시각화"""
    if not config.visualize:
        return
    
    print("\n결과 시각화 중...")
    
    for i, result in enumerate(results):
        if result["predicted_actions"] is None or result.get("error") is not None:
            continue
        
        # 액션 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Spot 로봇 액션 예측 - 샘플 {i+1}')
        
        pred_actions = result["predicted_actions"]
        leg_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
        leg_keys = ['action.fl_leg', 'action.fr_leg', 'action.rl_leg', 'action.rr_leg']
        
        for j, (ax, leg_name, leg_key) in enumerate(zip(axes.flat, leg_names, leg_keys)):
            if leg_key in pred_actions:
                action_data = pred_actions[leg_key]
                if hasattr(action_data, 'cpu'):
                    action_data = action_data.cpu()
                if hasattr(action_data, 'numpy'):
                    action_data = action_data.numpy()
                
                if action_data.ndim == 1:
                    ax.plot(action_data, 'r-', label='Predicted', alpha=0.7)
                else:
                    ax.plot(action_data.flatten(), 'r-', label='Predicted', alpha=0.7)
                
                ax.set_title(f'{leg_name} Leg')
                ax.set_xlabel('Dimension')
                ax.set_ylabel('Action Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{leg_name} Leg')
        
        plt.tight_layout()
        
        # 저장
        output_file = os.path.join(config.output_dir, f"action_visualization_sample_{i+1}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  시각화 저장됨: {output_file}")


def main(config: SpotInferenceConfig):
    """메인 추론 함수"""
    
    print("="*80)
    print("Spot 로봇 GR00T 추론 시작")
    print("="*80)
    
    # 1. 정책 모델 생성
    policy = create_policy(config)
    
    # 2. 테스트 데이터셋 로드
    dataset = load_test_dataset(config)
    
    # 3. 추론 실행
    results = run_inference(policy, dataset, config)
    
    # 4. 결과 저장
    save_results(results, config)
    
    # 5. 시각화
    visualize_results(results, config)
    
    print("\n" + "="*80)
    print("✅ Spot 로봇 추론 완료!")
    print(f"결과 저장 위치: {config.output_dir}")
    print("="*80)


if __name__ == "__main__":
    # 명령행 인수 파싱
    import tyro
    config = tyro.cli(SpotInferenceConfig)
    
    # 메인 함수 실행
    main(config) 
