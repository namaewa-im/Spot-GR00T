# gr00t/experiment/data_config.py에 추가

class SpotDataConfig(BaseDataConfig):
    """Spot 로봇을 위한 데이터 설정"""
    
    video_keys = [
        "video.front",
        "video.rear", 
        "video.hand"
    ]
    
    state_keys = [
        "state.body_linear_velocity",
        "state.body_angular_velocity",
        "state.gravity_direction", 
        "state.teleop_command",
        "state.joint_position_offsets",
        "state.joint_velocities",
        "state.previous_actions"
    ]
    
    action_keys = [
        "action.joint_positions"
    ]
    
    language_keys = ["annotation.human.task_description"]
    
    observation_indices = [0]
    action_indices = list(range(16))
    
    # 정규화 모드 설정
    state_normalization_modes = {
        "state.body_linear_velocity": "min_max",
        "state.body_angular_velocity": "min_max",
        "state.gravity_direction": "min_max",
        "state.teleop_command": "min_max",
        "state.joint_position_offsets": "min_max",
        "state.joint_velocities": "min_max",
        "state.previous_actions": "min_max"
    }
    
    action_normalization_modes = {
        "action.joint_positions": "min_max"
    }
    
    def modality_config(self):
        """모달리티 설정 반환"""
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
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
        transforms = [
            # 비디오 변환
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
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
                max_action_dim=32,
            ),
        ]
        
        return ComposedModalityTransform(transforms=transforms)

# DATA_CONFIG_MAP에 추가
DATA_CONFIG_MAP["spot"] = SpotDataConfig()