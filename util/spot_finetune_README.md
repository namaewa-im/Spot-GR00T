# Spot 로봇 GR00T N1.5-3B 파인튜닝 가이드

이 스크립트는 Spot 로봇 데이터를 사용하여 GR00T N1.5-3B 모델을 파인튜닝하는 도구입니다.

## 📋 요구사항

- Python 3.8+
- PyTorch 2.0+
- GR00T 라이브러리
- CUDA 지원 GPU (권장)

## 🚀 기본 사용법

### 1. 기본 파인튜닝 실행

```bash
python util/spot_finetune.py \
    --dataset_path ./demo_data/spot-lerobot-task0 \
    --output_dir ./spot_finetuned_model \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-5
```

### 2. LoRA를 사용한 효율적인 파인튜닝

```bash
python util/spot_finetune.py \
    --dataset_path ./demo_data/spot-lerobot-task0 \
    --output_dir ./spot_finetuned_model_lora \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 8 \
    --learning_rate 5e-5
```

### 3. 전체 모델 파인튜닝 (고급)

```bash
python util/spot_finetune.py \
    --dataset_path ./demo_data/spot-lerobot-task0 \
    --output_dir ./spot_finetuned_model_full \
    --tune_visual \
    --tune_llm \
    --tune_projector \
    --tune_diffusion_model \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8
```

## 📊 주요 매개변수

### 데이터셋 관련
- `--dataset_path`: Spot 데이터셋 경로 (필수)
- `--video_backend`: 비디오 백엔드 ("decord" 또는 "torchvision_av")

### 모델 관련
- `--base_model_path`: 사전 훈련된 모델 경로 (기본값: "nvidia/GR00T-N1.5-3B")
- `--output_dir`: 훈련된 모델 저장 디렉토리

### 훈련 관련
- `--num_epochs`: 훈련 에포크 수 (기본값: 3)
- `--batch_size`: 배치 크기 (기본값: 4)
- `--learning_rate`: 학습률 (기본값: 1e-5)
- `--warmup_ratio`: 워밍업 비율 (기본값: 0.1)
- `--gradient_accumulation_steps`: 그래디언트 누적 스텝 수 (기본값: 4)

### 모델 튜닝 관련
- `--tune_visual`: 비주얼 인코더 튜닝 여부
- `--tune_llm`: 언어 모델 튜닝 여부
- `--tune_projector`: 프로젝터 튜닝 여부 (기본값: True)
- `--tune_diffusion_model`: 디퓨전 모델 튜닝 여부 (기본값: True)

### LoRA 관련
- `--use_lora`: LoRA 사용 여부
- `--lora_rank`: LoRA 랭크 (기본값: 16)
- `--lora_alpha`: LoRA 알파 (기본값: 32)
- `--lora_dropout`: LoRA 드롭아웃 (기본값: 0.1)

### 기타
- `--save_steps`: 체크포인트 저장 간격 (기본값: 500)
- `--logging_steps`: 로깅 간격 (기본값: 10)
- `--dataloader_num_workers`: 데이터로더 워커 수 (기본값: 4)
- `--report_to`: 훈련 메트릭 보고 대상 ("none", "wandb", "tensorboard")
- `--seed`: 랜덤 시드 (기본값: 42)
- `--resume`: 체크포인트에서 재시작 여부

## 💡 권장 설정

### 메모리가 제한적인 경우 (8GB GPU)
```bash
python util/spot_finetune.py \
    --dataset_path ./demo_data/spot-lerobot-task0 \
    --output_dir ./spot_finetuned_model \
    --use_lora \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5
```

### 충분한 메모리가 있는 경우 (24GB+ GPU)
```bash
python util/spot_finetune.py \
    --dataset_path ./demo_data/spot-lerobot-task0 \
    --output_dir ./spot_finetuned_model \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --tune_projector \
    --tune_diffusion_model
```

### 빠른 실험을 위한 경우
```bash
python util/spot_finetune.py \
    --dataset_path ./demo_data/spot-lerobot-task0 \
    --output_dir ./spot_finetuned_model_quick \
    --num_epochs 1 \
    --batch_size 4 \
    --save_steps 100 \
    --logging_steps 5
```

## 📁 출력 구조

훈련이 완료되면 다음과 같은 구조로 모델이 저장됩니다:

```
spot_finetuned_model/
├── checkpoint-500/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── training_args.bin
│   └── ...
├── checkpoint-1000/
│   └── ...
└── final_model/
    ├── pytorch_model.bin
    ├── config.json
    └── ...
```

## 🔍 모니터링

### 로그 확인
훈련 중에는 다음과 같은 정보가 출력됩니다:
- 데이터셋 로드 상태
- 모델 로드 상태
- 훈련 진행률
- 손실 값
- GPU 메모리 사용량

### 체크포인트 관리
- `save_steps`마다 체크포인트가 저장됩니다
- 최대 5개의 체크포인트가 유지됩니다 (`save_total_limit=5`)
- `--resume` 플래그로 이전 체크포인트에서 재시작할 수 있습니다

## ⚠️ 주의사항

1. **메모리 사용량**: 배치 크기와 모델 튜닝 설정에 따라 GPU 메모리 사용량이 크게 달라집니다.
2. **데이터셋 경로**: 올바른 Spot 데이터셋 경로를 지정해야 합니다.
3. **GPU 가용성**: CUDA 지원 GPU가 필요합니다.
4. **저장 공간**: 훈련된 모델은 수 GB의 저장 공간이 필요할 수 있습니다.

## 🐛 문제 해결

### 메모리 부족 오류
- 배치 크기를 줄이세요
- LoRA를 사용하세요
- 그래디언트 누적 스텝을 늘리세요

### 데이터셋 로드 오류
- 데이터셋 경로가 올바른지 확인하세요
- 데이터셋 구조가 예상과 일치하는지 확인하세요

### 모델 로드 오류
- 인터넷 연결을 확인하세요 (HuggingFace 모델 다운로드)
- 충분한 디스크 공간이 있는지 확인하세요

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. GR00T 라이브러리 버전
2. PyTorch 버전
3. CUDA 버전
4. GPU 드라이버 버전 