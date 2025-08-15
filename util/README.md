## util/hf_downloader.sh 사용법
```bash
./util/hf_downloader.sh --url "namaewa-im/spot-lerobot-task0" --output_dir "./demo_data"
./util/hf_downloader.sh --url "https://huggingface.co/datasets/namaewa-im/spot-lerobot-task0" --output_dir "./demo_data"
```

## util/dataset_inspector.py 사용법
```bash
python util/dataset_inspector.py --dataset_path ./demo_data/spot-lerobot-task0
```
#### 상세 정보와 함께
```bash
python util/dataset_inspector.py --dataset_path ./demo_data/spot-lerobot-task0 --verbose
```

## util/video_validator.py 사용법
```bash
python3 util/video_validator.py demo_data/spot-lerobot-task0 --video-key ego_view --backend opencv
```

## scripts/spot_finetune.py 사용법
