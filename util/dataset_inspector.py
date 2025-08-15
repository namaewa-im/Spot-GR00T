#!/usr/bin/env python3
"""
허깅페이스 데이터셋 및 특별한 데이터 구조 분석 도구
사용법: python dataset_inspector.py --dataset_path <데이터셋_경로> [옵션]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import glob
import numpy as np

try:
    from datasets import load_from_disk, Dataset, DatasetDict
    from datasets.utils.logging import set_verbosity_error
    set_verbosity_error()  # 불필요한 로그 숨기기
except ImportError:
    print("오류: datasets 라이브러리가 필요합니다. 'pip install datasets'로 설치하세요.")
    sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
    """numpy 배열과 기타 객체를 JSON 직렬화할 수 있도록 하는 인코더"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'dtype'):
            return str(obj)
        return super().default(obj)


class DatasetInspector:
    def __init__(self, dataset_path: str, verbose: bool = False):
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        
    def inspect_dataset(self) -> Dict[str, Any]:
        """데이터셋을 분석하고 결과를 반환"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {self.dataset_path}")
        
        # 먼저 일반 허깅페이스 데이터셋인지 확인
        try:
            dataset = load_from_disk(str(self.dataset_path))
            return self._analyze_hf_dataset(dataset)
        except Exception as e:
            # 일반 데이터셋이 아니면 특별한 구조로 분석
            if self.verbose:
                print(f"lerobot 데이터셋이므로 특별한 구조로 분석합니다: {e}")
            return self._analyze_special_structure()
    
    def _analyze_hf_dataset(self, dataset) -> Dict[str, Any]:
        """일반 허깅페이스 데이터셋 분석"""
        result = {
            "dataset_path": str(self.dataset_path),
            "dataset_type": "HuggingFace Dataset",
            "splits": {},
            "total_samples": 0,
            "features": {},
            "file_size": self._get_directory_size(),
            "sample_data": {}
        }
        
        if isinstance(dataset, DatasetDict):
            # 여러 분할이 있는 경우
            for split_name, split_dataset in dataset.items():
                split_info = self._analyze_split(split_dataset, split_name)
                result["splits"][split_name] = split_info
                result["total_samples"] += split_info["num_samples"]
                result["features"][split_name] = split_info["features"]
                result["sample_data"][split_name] = split_info["sample_data"]
        else:
            # 단일 데이터셋인 경우
            split_info = self._analyze_split(dataset, "main")
            result["splits"]["main"] = split_info
            result["total_samples"] = split_info["num_samples"]
            result["features"]["main"] = split_info["features"]
            result["sample_data"]["main"] = split_info["sample_data"]
        
        return result
    
    def _analyze_special_structure(self) -> Dict[str, Any]:
        """특별한 데이터 구조 분석 (parquet + 메타데이터)"""
        result = {
            "dataset_path": str(self.dataset_path),
            "dataset_type": "Special Structure (Parquet + Metadata)",
            "file_size": self._get_directory_size(),
            "structure": {},
            "metadata": {},
            "data_files": {},
            "sample_data": {}
        }
        
        # 디렉토리 구조 분석
        result["structure"] = self._analyze_directory_structure()
        
        # 메타데이터 파일 분석
        result["metadata"] = self._analyze_metadata_files()
        
        # 데이터 파일 분석
        result["data_files"] = self._analyze_data_files()
        
        # 샘플 데이터 추출
        result["sample_data"] = self._extract_sample_data()
        
        return result
    
    def _analyze_directory_structure(self) -> Dict[str, Any]:
        """디렉토리 구조 분석"""
        structure = {}
        
        for item in self.dataset_path.iterdir():
            if item.is_dir():
                structure[item.name] = {
                    "type": "directory",
                    "contents": [f.name for f in item.iterdir() if f.is_file()]
                }
            else:
                structure[item.name] = {
                    "type": "file",
                    "size": f"{item.stat().st_size / 1024:.1f} KB"
                }
        
        return structure
    
    def _analyze_metadata_files(self) -> Dict[str, Any]:
        """메타데이터 파일 분석"""
        metadata = {}
        
        # info.json 분석
        info_file = self.dataset_path / "meta" / "info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                metadata["info.json"] = {
                    "keys": list(info_data.keys()),
                    "content": info_data
                }
            except Exception as e:
                metadata["info.json"] = {"error": str(e)}
        
        # modality.json 분석
        modality_file = self.dataset_path / "meta" / "modality.json"
        if modality_file.exists():
            try:
                with open(modality_file, 'r', encoding='utf-8') as f:
                    modality_data = json.load(f)
                metadata["modality.json"] = {
                    "keys": list(modality_data.keys()),
                    "content": modality_data
                }
            except Exception as e:
                metadata["modality.json"] = {"error": str(e)}
        
        # episodes.jsonl 분석
        episodes_file = self.dataset_path / "meta" / "episodes.jsonl"
        if episodes_file.exists():
            try:
                episodes = []
                with open(episodes_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        episodes.append(json.loads(line.strip()))
                metadata["episodes.jsonl"] = {
                    "num_episodes": len(episodes),
                    "sample_episodes": episodes[:3] if episodes else []
                }
            except Exception as e:
                metadata["episodes.jsonl"] = {"error": str(e)}
        
        # tasks.jsonl 분석
        tasks_file = self.dataset_path / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            try:
                tasks = []
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        tasks.append(json.loads(line.strip()))
                metadata["tasks.jsonl"] = {
                    "num_tasks": len(tasks),
                    "sample_tasks": tasks[:3] if tasks else []
                }
            except Exception as e:
                metadata["tasks.jsonl"] = {"error": str(e)}
        
        return metadata
    
    def _analyze_data_files(self) -> Dict[str, Any]:
        """데이터 파일 분석"""
        data_files = {}
        
        # parquet 파일 찾기
        parquet_files = list(self.dataset_path.rglob("*.parquet"))
        data_files["parquet_files"] = {
            "count": len(parquet_files),
            "files": []
        }
        
        total_rows = 0
        total_size = 0
        
        for parquet_file in parquet_files[:5]:  # 처음 5개 파일만 분석
            try:
                df = pd.read_parquet(parquet_file)
                file_info = {
                    "name": parquet_file.name,
                    "path": str(parquet_file.relative_to(self.dataset_path)),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "size": f"{parquet_file.stat().st_size / 1024:.1f} KB",
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
                }
                data_files["parquet_files"]["files"].append(file_info)
                total_rows += len(df)
                total_size += parquet_file.stat().st_size
            except Exception as e:
                data_files["parquet_files"]["files"].append({
                    "name": parquet_file.name,
                    "error": str(e)
                })
        
        data_files["parquet_files"]["total_rows"] = total_rows
        data_files["parquet_files"]["total_size"] = f"{total_size / 1024 / 1024:.1f} MB"
        
        return data_files
    
    def _extract_sample_data(self) -> Dict[str, Any]:
        """샘플 데이터 추출"""
        sample_data = {}
        
        # 첫 번째 parquet 파일에서 샘플 추출
        parquet_files = list(self.dataset_path.rglob("*.parquet"))
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                sample_data["first_file"] = {
                    "filename": parquet_files[0].name,
                    "sample_rows": df.head(3).to_dict('records')
                }
            except Exception as e:
                sample_data["first_file"] = {"error": str(e)}
        
        return sample_data
    
    def _analyze_split(self, dataset: Dataset, split_name: str) -> Dict[str, Any]:
        """개별 분할 분석"""
        split_info = {
            "num_samples": len(dataset),
            "features": {},
            "sample_data": {}
        }
        
        # 피처 정보 분석
        for feature_name, feature in dataset.features.items():
            feature_info = {
                "dtype": str(feature.dtype),
                "type": type(feature).__name__
            }
            
            # 특별한 피처 타입 처리
            if hasattr(feature, 'num_classes'):
                feature_info["num_classes"] = feature.num_classes
            if hasattr(feature, 'names'):
                feature_info["class_names"] = feature.names
            
            split_info["features"][feature_name] = feature_info
        
        # 샘플 데이터 추출 (처음 3개)
        if len(dataset) > 0:
            sample_indices = min(3, len(dataset))
            samples = dataset.select(range(sample_indices))
            
            for i in range(sample_indices):
                sample = samples[i]
                sample_data = {}
                for key, value in sample.items():
                    # 긴 텍스트나 리스트는 잘라서 표시
                    if isinstance(value, str) and len(value) > 100:
                        sample_data[key] = value[:100] + "..."
                    elif isinstance(value, list) and len(value) > 10:
                        sample_data[key] = str(value[:10]) + "..."
                    else:
                        sample_data[key] = value
                
                split_info["sample_data"][f"sample_{i+1}"] = sample_data
        
        return split_info
    
    def _get_directory_size(self) -> str:
        """디렉토리 크기 계산"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        # 크기를 읽기 쉬운 형태로 변환
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"
    
    def print_summary(self, result: Dict[str, Any]):
        """요약 정보 출력"""
        print("=" * 60)
        print("📊 데이터셋 구성 분석 결과")
        print("=" * 60)
        print(f"📁 경로: {result['dataset_path']}")
        print(f"📦 타입: {result['dataset_type']}")
        print(f"📏 크기: {result['file_size']}")
        
        if result['dataset_type'] == "HuggingFace Dataset":
            self._print_hf_summary(result)
        else:
            self._print_special_summary(result)
    
    def _print_hf_summary(self, result: Dict[str, Any]):
        """허깅페이스 데이터셋 요약 출력"""
        print(f"📊 총 샘플 수: {result['total_samples']:,}개")
        print()
        
        # 분할 정보
        print("🔍 분할 정보:")
        for split_name, split_info in result["splits"].items():
            print(f"  • {split_name}: {split_info['num_samples']:,}개 샘플")
        print()
        
        # 피처 정보 (첫 번째 분할 기준)
        first_split = list(result["splits"].keys())[0]
        print(f"📋 피처 정보 ({first_split}):")
        for feature_name, feature_info in result["features"][first_split].items():
            dtype = feature_info["dtype"]
            feature_type = feature_info["type"]
            print(f"  • {feature_name}: {dtype} ({feature_type})")
            
            # 추가 정보가 있으면 표시
            if "num_classes" in feature_info:
                print(f"    - 클래스 수: {feature_info['num_classes']}")
            if "class_names" in feature_info:
                print(f"    - 클래스명: {feature_info['class_names']}")
        print()
        
        # 샘플 데이터
        print("📝 샘플 데이터:")
        for split_name, sample_data in result["sample_data"].items():
            print(f"  • {split_name}:")
            for sample_key, sample in sample_data.items():
                print(f"    {sample_key}:")
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 50:
                        print(f"      {key}: {value[:50]}...")
                    else:
                        print(f"      {key}: {value}")
                print()
    
    def _print_special_summary(self, result: Dict[str, Any]):
        """특별한 구조 요약 출력"""
        print()
        
        # 디렉토리 구조
        print("📁 디렉토리 구조:")
        for name, info in result["structure"].items():
            if info["type"] == "directory":
                print(f"  • {name}/ (디렉토리)")
                for content in info["contents"][:5]:  # 처음 5개만 표시
                    print(f"    - {content}")
                if len(info["contents"]) > 5:
                    print(f"    ... 외 {len(info['contents']) - 5}개 파일")
            else:
                print(f"  • {name} ({info['size']})")
        print()
        
        # 메타데이터 정보
        if result["metadata"]:
            print("📋 메타데이터 정보:")
            for meta_name, meta_info in result["metadata"].items():
                if "error" not in meta_info:
                    if "num_episodes" in meta_info:
                        print(f"  • {meta_name}: {meta_info['num_episodes']}개 에피소드")
                    elif "num_tasks" in meta_info:
                        print(f"  • {meta_name}: {meta_info['num_tasks']}개 태스크")
                    elif "keys" in meta_info:
                        print(f"  • {meta_name}: {len(meta_info['keys'])}개 키")
                else:
                    print(f"  • {meta_name}: 오류 - {meta_info['error']}")
            print()
        
        # 데이터 파일 정보
        if "parquet_files" in result["data_files"]:
            parquet_info = result["data_files"]["parquet_files"]
            print("📊 데이터 파일 정보:")
            print(f"  • Parquet 파일 수: {parquet_info['count']}개")
            print(f"  • 총 행 수: {parquet_info['total_rows']:,}개")
            print(f"  • 총 크기: {parquet_info['total_size']}")
            
            if parquet_info["files"]:
                print(f"  • 첫 번째 파일: {parquet_info['files'][0]['name']}")
                print(f"    - 행 수: {parquet_info['files'][0]['rows']:,}개")
                print(f"    - 컬럼: {', '.join(parquet_info['files'][0]['columns'][:5])}")
                if len(parquet_info['files'][0]['columns']) > 5:
                    print(f"    ... 외 {len(parquet_info['files'][0]['columns']) - 5}개 컬럼")
            print()
        
        # 샘플 데이터
        if result["sample_data"]:
            print("📝 샘플 데이터:")
            for key, data in result["sample_data"].items():
                if "error" not in data:
                    print(f"  • {key}:")
                    for i, row in enumerate(data["sample_rows"][:2]):  # 처음 2개 행만
                        print(f"    행 {i+1}:")
                        for col, val in list(row.items())[:5]:  # 처음 5개 컬럼만
                            if isinstance(val, str) and len(val) > 30:
                                print(f"      {col}: {val[:30]}...")
                            else:
                                print(f"      {col}: {val}")
                        if len(row) > 5:
                            print(f"      ... 외 {len(row) - 5}개 컬럼")
                        print()
    
    def print_detailed(self, result: Dict[str, Any]):
        """상세 정보 출력 (JSON 형태)"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔍 상세 정보 (JSON)")
            print("=" * 60)
            print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyEncoder))


def main():
    parser = argparse.ArgumentParser(
        description="허깅페이스 데이터셋 및 특별한 데이터 구조 분석 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python dataset_inspector.py --dataset_path ./data/dataset_name
  python dataset_inspector.py --dataset_path ./data/dataset_name --verbose
  python dataset_inspector.py --dataset_path ./data/dataset_name --json
        """
    )
    
    parser.add_argument(
        "--dataset_path", 
        required=True,
        help="분석할 데이터셋 경로"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="상세한 정보 출력"
    )
    
    parser.add_argument(
        "--json", 
        action="store_true",
        help="JSON 형태로만 출력"
    )
    
    args = parser.parse_args()
    
    try:
        # 데이터셋 분석
        inspector = DatasetInspector(args.dataset_path, args.verbose)
        result = inspector.inspect_dataset()
        
        if args.json:
            # JSON 형태로만 출력
            print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        else:
            # 요약 정보 출력
            inspector.print_summary(result)
            
            # 상세 정보 출력 (verbose 옵션이 있을 때)
            if args.verbose:
                inspector.print_detailed(result)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 