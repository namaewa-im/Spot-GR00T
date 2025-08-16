#!/usr/bin/env python3
"""
Spot 로봇 GR00T 간단한 추론 스크립트
"""

import os
import torch
import numpy as np
import json
from pathlib import Path

# GR00T 관련 import
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP


def main():
    """간단한 추론 실행"""
    
    print("="*80)
    print("Spot 로봇 GR00T 간단한 추론")
    print("="*80)
    
    # 설정
    model_path = "./spot_finetuned_model_corrected"
    dataset_path = "./demo_data/spot-lerobot-task0"
    output_dir = "./spot_inference_results"
    num_samples = 3
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # Spot 데이터에 맞는 간단한 설정
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.transform.base import ComposedModalityTransform
    
    # 간단한 모달리티 설정
    modality_config = {
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
    
    # 기존 GR00T 예제를 참고한 변환 설정
    from gr00t.data.transform import VideoToTensor, VideoResize, VideoToNumpy
    from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.model.transforms import GR00TTransform
    
    modality_transform = ComposedModalityTransform(transforms=[
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
    
    print("정책 모델 생성 중...")
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print("데이터셋 로드 중...")
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=None,  # 변환은 정책에서만 적용
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend="decord",
    )
    
    print(f"데이터셋 크기: {len(dataset)} 샘플")
    
    # 추론 실행
    print(f"\n추론 시작 - {num_samples}개 샘플")
    results = []
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\n샘플 {i+1}/{num_samples} 처리 중...")
        
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
                    "predicted_actions": predicted_actions.get("action_pred", None).cpu().numpy() if predicted_actions.get("action_pred") is not None else None,
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
    
    # 결과 저장
    output_file = os.path.join(output_dir, "inference_results.json")
    
    # numpy 배열을 리스트로 변환
    serializable_results = []
    for result in results:
        serializable_result = {
            "sample_id": result["sample_id"],
            "predicted_actions": result["predicted_actions"].tolist() if result["predicted_actions"] is not None else None,
            "error": result.get("error", None),
        }
        serializable_results.append(serializable_result)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n결과 저장됨: {output_file}")
    print("\n" + "="*80)
    print("✅ Spot 로봇 추론 완료!")
    print("="*80)


if __name__ == "__main__":
    main() 
