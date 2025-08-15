#!/usr/bin/env python3
"""
비디오 파일 유효성 검증 스크립트

이 스크립트는 데이터셋의 비디오 파일들이 유효한 프레임을 가지고 있는지 확인합니다.
각 mp4 파일의 프레임 수, 해상도, FPS, 코덱 등을 검사하고 문제가 있는 파일을 보고합니다.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("Warning: Decord not available. Install with: pip install decord")

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    print("Warning: PyAV not available. Install with: pip install av")


class VideoValidator:
    """비디오 파일 유효성 검증 클래스"""
    
    def __init__(self, backend: str = "decord"):
        """
        Args:
            backend: 비디오 백엔드 ("decord", "opencv", "pyav")
        """
        self.backend = backend
        self.valid_backends = ["decord", "opencv", "pyav"]
        
        if backend not in self.valid_backends:
            raise ValueError(f"Backend must be one of {self.valid_backends}")
        
        # 백엔드별 가용성 확인
        if backend == "decord" and not DECORD_AVAILABLE:
            raise ImportError("Decord is not available. Install with: pip install decord")
        elif backend == "opencv" and not CV2_AVAILABLE:
            raise ImportError("OpenCV is not available. Install with: pip install opencv-python")
        elif backend == "pyav" and not PYAV_AVAILABLE:
            raise ImportError("PyAV is not available. Install with: pip install av")
    
    def validate_video(self, video_path: str) -> Dict:
        """
        단일 비디오 파일 검증
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            검증 결과 딕셔너리
        """
        result = {
            "path": video_path,
            "valid": False,
            "error": None,
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration": 0.0,
            "codec": None,
            "file_size_mb": 0.0
        }
        
        try:
            # 파일 크기 확인
            file_size = os.path.getsize(video_path)
            result["file_size_mb"] = file_size / (1024 * 1024)
            
            if file_size == 0:
                result["error"] = "Empty file"
                return result
            
            # 백엔드별 검증
            if self.backend == "decord":
                result = self._validate_with_decord(video_path, result)
            elif self.backend == "opencv":
                result = self._validate_with_opencv(video_path, result)
            elif self.backend == "pyav":
                result = self._validate_with_pyav(video_path, result)
                
        except Exception as e:
            result["error"] = str(e)
            result["valid"] = False
        
        return result
    
    def _validate_with_decord(self, video_path: str, result: Dict) -> Dict:
        """Decord를 사용한 비디오 검증"""
        try:
            vr = decord.VideoReader(video_path)
            
            result["frame_count"] = len(vr)
            result["fps"] = vr.get_avg_fps()
            result["width"] = vr[0].shape[1]
            result["height"] = vr[0].shape[0]
            result["duration"] = result["frame_count"] / result["fps"] if result["fps"] > 0 else 0
            
            # 첫 번째와 마지막 프레임 읽기 테스트
            first_frame = vr[0].asnumpy()
            last_frame = vr[-1].asnumpy()
            
            if first_frame is not None and last_frame is not None:
                result["valid"] = True
            else:
                result["error"] = "Failed to read frames"
                
        except Exception as e:
            result["error"] = f"Decord error: {str(e)}"
        
        return result
    
    def _validate_with_opencv(self, video_path: str, result: Dict) -> Dict:
        """OpenCV를 사용한 비디오 검증"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                result["error"] = "Failed to open video file"
                return result
            
            result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result["fps"] = cap.get(cv2.CAP_PROP_FPS)
            result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result["duration"] = result["frame_count"] / result["fps"] if result["fps"] > 0 else 0
            
            # 첫 번째와 마지막 프레임 읽기 테스트
            ret1, first_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, result["frame_count"] - 1)
            ret2, last_frame = cap.read()
            
            cap.release()
            
            if ret1 and ret2 and first_frame is not None and last_frame is not None:
                result["valid"] = True
            else:
                result["error"] = "Failed to read frames"
                
        except Exception as e:
            result["error"] = f"OpenCV error: {str(e)}"
        
        return result
    
    def _validate_with_pyav(self, video_path: str, result: Dict) -> Dict:
        """PyAV를 사용한 비디오 검증"""
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            result["frame_count"] = stream.frames
            result["fps"] = float(stream.average_rate) if stream.average_rate else 0
            result["width"] = stream.width
            result["height"] = stream.height
            result["codec"] = stream.codec_context.name
            result["duration"] = float(stream.duration * stream.time_base) if stream.duration else 0
            
            # 첫 번째와 마지막 프레임 읽기 테스트
            container.seek(0)
            first_frame = next(container.decode(video=0))
            
            container.seek(stream.duration - 1 if stream.duration else 0)
            frames = list(container.decode(video=0))
            last_frame = frames[-1] if frames else None
            
            container.close()
            
            if first_frame is not None and last_frame is not None:
                result["valid"] = True
            else:
                result["error"] = "Failed to read frames"
                
        except Exception as e:
            result["error"] = f"PyAV error: {str(e)}"
        
        return result
    
    def validate_dataset(self, dataset_path: str, video_key: str = "ego_view") -> Dict:
        """
        데이터셋의 모든 비디오 파일 검증
        
        Args:
            dataset_path: 데이터셋 경로
            video_key: 비디오 키 (예: "ego_view", "front_view" 등)
            
        Returns:
            검증 결과 요약
        """
        dataset_path = Path(dataset_path)
        videos_path = dataset_path / "videos" / "chunk-000" / video_key
        
        if not videos_path.exists():
            return {
                "error": f"Videos path not found: {videos_path}",
                "total_files": 0,
                "valid_files": 0,
                "invalid_files": 0,
                "results": []
            }
        
        # 모든 mp4 파일 찾기
        video_files = list(videos_path.glob("*.mp4"))
        
        if not video_files:
            return {
                "error": f"No MP4 files found in {videos_path}",
                "total_files": 0,
                "valid_files": 0,
                "invalid_files": 0,
                "results": []
            }
        
        print(f"Found {len(video_files)} video files in {videos_path}")
        print(f"Using backend: {self.backend}")
        
        # 각 비디오 파일 검증
        results = []
        valid_count = 0
        invalid_count = 0
        
        for video_file in tqdm(video_files, desc="Validating videos"):
            result = self.validate_video(str(video_file))
            results.append(result)
            
            if result["valid"]:
                valid_count += 1
            else:
                invalid_count += 1
        
        # 통계 계산
        if results:
            valid_results = [r for r in results if r["valid"]]
            if valid_results:
                avg_frame_count = sum(r["frame_count"] for r in valid_results) / len(valid_results)
                avg_fps = sum(r["fps"] for r in valid_results) / len(valid_results)
                avg_duration = sum(r["duration"] for r in valid_results) / len(valid_results)
                avg_file_size = sum(r["file_size_mb"] for r in valid_results) / len(valid_results)
            else:
                avg_frame_count = avg_fps = avg_duration = avg_file_size = 0
        else:
            avg_frame_count = avg_fps = avg_duration = avg_file_size = 0
        
        return {
            "dataset_path": str(dataset_path),
            "video_key": video_key,
            "backend": self.backend,
            "total_files": len(video_files),
            "valid_files": valid_count,
            "invalid_files": invalid_count,
            "success_rate": valid_count / len(video_files) if video_files else 0,
            "average_frame_count": avg_frame_count,
            "average_fps": avg_fps,
            "average_duration": avg_duration,
            "average_file_size_mb": avg_file_size,
            "results": results
        }
    
    def print_summary(self, summary: Dict):
        """검증 결과 요약 출력"""
        print("\n" + "="*80)
        print("VIDEO VALIDATION SUMMARY")
        print("="*80)
        
        if "error" in summary:
            print(f"❌ Error: {summary['error']}")
            return
        
        print(f"📁 Dataset: {summary['dataset_path']}")
        print(f"🎥 Video Key: {summary['video_key']}")
        print(f"🔧 Backend: {summary['backend']}")
        print(f"📊 Total Files: {summary['total_files']}")
        print(f"✅ Valid Files: {summary['valid_files']}")
        print(f"❌ Invalid Files: {summary['invalid_files']}")
        print(f"📈 Success Rate: {summary['success_rate']:.2%}")
        
        if summary['valid_files'] > 0:
            print(f"\n📊 AVERAGE STATISTICS (Valid Files Only):")
            print(f"   🎞️  Frame Count: {summary['average_frame_count']:.1f}")
            print(f"   ⏱️  FPS: {summary['average_fps']:.2f}")
            print(f"   ⏰ Duration: {summary['average_duration']:.2f}s")
            print(f"   💾 File Size: {summary['average_file_size_mb']:.2f} MB")
        
        # 문제가 있는 파일들 출력
        invalid_results = [r for r in summary['results'] if not r['valid']]
        if invalid_results:
            print(f"\n❌ INVALID FILES ({len(invalid_results)}):")
            for result in invalid_results:
                print(f"   📁 {Path(result['path']).name}: {result['error']}")
        
        # 유효한 파일들의 샘플 출력
        valid_results = [r for r in summary['results'] if r['valid']]
        if valid_results:
            print(f"\n✅ SAMPLE VALID FILES (First 5):")
            for result in valid_results[:5]:
                filename = Path(result['path']).name
                print(f"   📁 {filename}: {result['frame_count']} frames, "
                      f"{result['width']}x{result['height']}, {result['fps']:.1f} FPS")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="비디오 파일 유효성 검증")
    parser.add_argument("dataset_path", help="데이터셋 경로")
    parser.add_argument("--video-key", default="ego_view", help="비디오 키 (기본값: ego_view)")
    parser.add_argument("--backend", default="decord", choices=["decord", "opencv", "pyav"], 
                       help="비디오 백엔드 (기본값: decord)")
    parser.add_argument("--output", help="결과를 JSON 파일로 저장")
    parser.add_argument("--verbose", action="store_true", help="상세한 출력")
    
    args = parser.parse_args()
    
    try:
        # 비디오 검증기 생성
        validator = VideoValidator(backend=args.backend)
        
        # 데이터셋 검증
        summary = validator.validate_dataset(args.dataset_path, args.video_key)
        
        # 결과 출력
        validator.print_summary(summary)
        
        # JSON 파일로 저장
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")
        
        # 상세한 결과 출력
        if args.verbose:
            print(f"\n📋 DETAILED RESULTS:")
            for result in summary['results']:
                filename = Path(result['path']).name
                status = "✅" if result['valid'] else "❌"
                print(f"{status} {filename}: {result}")
        
        # 종료 코드
        if summary.get('invalid_files', 0) > 0:
            sys.exit(1)  # 문제가 있는 파일이 있으면 에러 코드 반환
        else:
            sys.exit(0)  # 모든 파일이 유효하면 성공 코드 반환
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 