#!/bin/bash

# 허깅페이스 데이터셋 다운로더
# 사용법: ./hf_downloader.sh --url <데이터셋_URL> --output_dir <출력_디렉토리>

set -e

# 기본값 설정
URL=""
OUTPUT_DIR=""
VERBOSE=false

# 도움말 함수
show_help() {
    echo "허깅페이스 데이터셋 다운로더"
    echo ""
    echo "사용법:"
    echo "  $0 --url <데이터셋_URL> --output_dir <출력_디렉토리> [옵션]"
    echo ""
    echo "옵션:"
    echo "  --url <URL>           다운로드할 허깅페이스 데이터셋 URL (필수)"
    echo "  --output_dir <DIR>    데이터셋을 저장할 디렉토리 (필수)"
    echo "  --verbose             상세한 출력 표시"
    echo "  --help                이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0 --url https://huggingface.co/datasets/username/dataset_name --output_dir ./data"
    echo "  $0 --url username/dataset_name --output_dir ./data"
}

# 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            URL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 필수 인수 확인
if [[ -z "$URL" ]]; then
    echo "오류: --url 옵션이 필요합니다."
    show_help
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "오류: --output_dir 옵션이 필요합니다."
    show_help
    exit 1
fi

# URL에서 데이터셋 이름 추출
if [[ "$URL" == https://huggingface.co/datasets/* ]]; then
    DATASET_NAME=$(echo "$URL" | sed 's|https://huggingface.co/datasets/||')
elif [[ "$URL" == https://huggingface.co/* ]]; then
    DATASET_NAME=$(echo "$URL" | sed 's|https://huggingface.co/||')
else
    DATASET_NAME="$URL"
fi

# 데이터셋 이름에서 마지막 부분만 추출 (username/dataset_name -> dataset_name)
DATASET_FOLDER_NAME=$(basename "$DATASET_NAME")

# 실제 저장할 디렉토리 경로 생성 (output_dir/dataset_name)
FINAL_OUTPUT_DIR="$OUTPUT_DIR/$DATASET_FOLDER_NAME"

# 출력 디렉토리 생성
if [[ "$VERBOSE" == true ]]; then
    echo "출력 디렉토리 생성: $FINAL_OUTPUT_DIR"
fi
mkdir -p "$FINAL_OUTPUT_DIR"

# Python 스크립트 생성
PYTHON_SCRIPT=$(cat << 'EOF'
import os
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset

def download_dataset(dataset_name, output_dir, verbose=False):
    try:
        if verbose:
            print(f"데이터셋 다운로드 시작: {dataset_name}")
            print(f"출력 디렉토리: {output_dir}")
        
        # 데이터셋 다운로드
        dataset_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        if verbose:
            print(f"데이터셋이 성공적으로 다운로드되었습니다: {dataset_path}")
        
        # 데이터셋 로드 테스트
        try:
            dataset = load_dataset(dataset_name)
            if verbose:
                print(f"데이터셋 정보:")
                print(f"  - 분할: {list(dataset.keys())}")
                for split_name, split_data in dataset.items():
                    print(f"  - {split_name}: {len(split_data)} 개 샘플")
        except Exception as e:
            if verbose:
                print(f"데이터셋 로드 테스트 중 경고: {e}")
        
        return True
        
    except Exception as e:
        print(f"오류: 데이터셋 다운로드 실패 - {e}")
        return False

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    output_dir = sys.argv[2]
    verbose = sys.argv[3].lower() == 'true'
    
    success = download_dataset(dataset_name, output_dir, verbose)
    sys.exit(0 if success else 1)
EOF
)

# 임시 Python 스크립트 파일 생성
TEMP_SCRIPT=$(mktemp)
echo "$PYTHON_SCRIPT" > "$TEMP_SCRIPT"

# Python 스크립트 실행
if [[ "$VERBOSE" == true ]]; then
    echo "허깅페이스 데이터셋 다운로드 중..."
    echo "데이터셋: $DATASET_NAME"
    echo "출력 디렉토리: $FINAL_OUTPUT_DIR"
fi

python3 "$TEMP_SCRIPT" "$DATASET_NAME" "$FINAL_OUTPUT_DIR" "$VERBOSE"

# 임시 파일 정리
rm -f "$TEMP_SCRIPT"

if [[ "$VERBOSE" == true ]]; then
    echo "다운로드 완료!"
    echo "데이터셋 위치: $FINAL_OUTPUT_DIR"
fi
