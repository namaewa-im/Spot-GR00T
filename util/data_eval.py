#!/usr/bin/env python3
"""
Spot 데이터셋 검증 스크립트
"""

import os
import pandas as pd
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

def validate_spot_dataset(dataset_path="./spot_dataset"):
    """Spot 데이터셋 검증"""
    
    print("=== Spot 데이터셋 검증 시작 ===")
    
    # 1. 폴더 구조 확인
    required_files = [
        "meta/modality.json",
        "meta/tasks.jsonl", 
        "meta/episodes.jsonl",
        "meta/info.json"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(dataset_path, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path} 존재")
        else:
            print(f"❌ {file_path} 없음")
            return False
    
    # 2. 데이터 로딩 테스트
    try:
        data_config = DATA_CONFIG_MAP["spot"]
        modality_configs = data_config.modality_config()
        
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            video_backend="decord",
        )
        
        print(f"✅ 데이터셋 로딩 성공: {len(dataset)} 샘플")
        
        # 3. 첫 번째 샘플 확인
        sample = dataset[0]
        print(f"✅ 샘플 로딩 성공")
        print(f"   - 상태 차원: {sample['state'].shape}")
        print(f"   - 액션 차원: {sample['action'].shape}")
        print(f"   - 비디오 키: {list(sample.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return False

if __name__ == "__main__":
    success = validate_spot_dataset()
    if success:
        print("🎉 데이터셋 검증 완료!")
    else:
        print("💥 데이터셋 검증 실패!")