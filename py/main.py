import argparse
import os
import logging
import matplotlib.pyplot as plt
import json

from config.settings import Settings
from config.logging_config import setup_logging
from core.detector import HoleDefectDetector
from core.image_processor import ImageProcessor
from core.utils import save_results_to_json, save_visualizations

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='구멍 결함 검출 시스템')
    
    # 기본 옵션
    parser.add_argument('--image', required=True, help='분석할 이미지 경로')
    parser.add_argument('--roi', required=False, help='ROI 정보가 담긴 JSON 파일 경로')
    parser.add_argument('--config', required=False, help='설정 파일 경로')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    parser.add_argument('--save', action='store_true', help='결과 저장 활성화')
    parser.add_argument('--visualize', action='store_true', help='시각화 결과 표시')
    
    return parser.parse_args()

def load_roi_from_json(roi_path):
    """JSON 파일에서 ROI 정보 로드"""
    try:
        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
        
        # ROI 데이터 유효성 검사
        if not isinstance(roi_data, list):
            logging.error("ROI 데이터는 리스트 형태여야 합니다.")
            return None
            
        return roi_data
    except Exception as e:
        logging.error(f"ROI 파일 로드 실패: {e}")
        return None

def get_default_roi(image_shape):
    """기본 ROI 생성 (이미지 중앙)"""
    h, w = image_shape[:2]
    center_x, center_y = w // 2, h // 2
    size = min(w, h) // 4
    
    # 중앙에 사각형 ROI 생성
    roi = [
        [(center_x - size, center_y - size),
         (center_x + size, center_y - size),
         (center_x + size, center_y + size),
         (center_x - size, center_y + size)]
    ]
    
    logging.info("기본 ROI 생성됨 (이미지 중앙)")
    return roi

def main():
    """메인 함수"""
    # 인자 파싱
    args = parse_args()
    
    # 설정 로드
    settings = Settings(args.config)
    
    # 로깅 설정
    debug_mode = args.debug or settings.get('debug_mode', False)
    setup_logging(
        log_dir=settings.get('log_dir', 'logs'),
        debug_mode=debug_mode
    )
    
    # 이미지 처리기 생성
    img_processor = ImageProcessor()
    
    # 이미지 로드
    try:
        image = img_processor.load_image(args.image)
        logging.info(f"이미지 로드 성공: {args.image}")
    except Exception as e:
        logging.error(f"이미지 로드 실패: {e}")
        return
    
    # ROI 로드
    roi_list = None
    if args.roi:
        roi_list = load_roi_from_json(args.roi)
    
    # ROI가 없으면 기본 ROI 생성
    if not roi_list:
        roi_list = get_default_roi(image.shape)
    
    # 검출 설정
    detector_config = {
        'debug_mode': debug_mode,
        'mode_min': settings.get('mode_min'),
        'mode_max': settings.get('mode_max'),
        'secondary_peak_ratio': settings.get('secondary_peak_ratio'),
        'min_dist': settings.get('min_dist'),
        'param1': settings.get('param1'),
        'param2': settings.get('param2'),
        'min_radius': settings.get('min_radius'),
        'max_radius': settings.get('max_radius')
    }
    
    # 결함 검출기 생성
    detector = HoleDefectDetector(config=detector_config)
    
    # 결과 저장
    results = []
    
    # 각 ROI 분석
    for idx, roi in enumerate(roi_list):
        logging.info(f"ROI {idx} 분석 시작")
        
        # ROI 분석
        result = detector.analyze_roi(image, roi, idx)
        results.append(result)
        
        # 결과 출력
        if result['is_defective']:
            logging.warning(f"ROI {idx}에서 결함 감지: {', '.join(result['reasons'])}")
        else:
            logging.info(f"ROI {idx} 정상")
    
    # 결과 저장
    if args.save:
        json_path = save_results_to_json(results)
        logging.info(f"결과가 {json_path}에 저장되었습니다.")
        
        if debug_mode:
            viz_paths = save_visualizations(results)
            if viz_paths:
                logging.info(f"시각화 결과가 저장되었습니다: {', '.join(viz_paths)}")
    
    # 시각화
    if args.visualize:
        for result in results:
            if result.get('visualization') is not None:
                plt.figure()
                result['visualization'].show()
    
    logging.info("분석 완료")
    
    return results

if __name__ == "__main__":
    main()
