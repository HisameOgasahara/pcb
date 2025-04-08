import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

import sys
import os
# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import HoleDefectDetector
from core.image_processor import ImageProcessor
from config.settings import Settings
from config.logging_config import setup_logging
from core.utils import save_results_to_json, save_visualizations

def main():
    # 설정 로드
    settings = Settings()
    
    # 로깅 설정
    setup_logging(
        log_dir=settings.get('log_dir'),
        debug_mode=settings.get('debug_mode')
    )
    
    st.title("구멍 결함 검출 시스템")
    
    # 사이드바 설정
    st.sidebar.header("설정")
    debug_mode = st.sidebar.checkbox("디버그 모드", value=settings.get('debug_mode'))
    
    # 이미지 업로드
    uploaded_file = st.sidebar.file_uploader("이미지 선택", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    # ROI 설정
    st.sidebar.subheader("ROI 설정")
    roi_count = st.sidebar.number_input("ROI 개수", min_value=1, max_value=10, value=1)
    
    # 검출 설정
    st.sidebar.subheader("검출 설정")
    mode_min = st.sidebar.slider("Mode 최소값", 0, 50, settings.get('mode_min'))
    mode_max = st.sidebar.slider("Mode 최대값", mode_min, 100, settings.get('mode_max'))
    secondary_peak_ratio = st.sidebar.slider("부 피크 비율", 0.1, 1.0, settings.get('secondary_peak_ratio'))
    
    # 원 검출 설정
    st.sidebar.subheader("원 검출 설정")
    min_radius = st.sidebar.slider("최소 반지름", 1, 50, settings.get('min_radius'))
    max_radius = st.sidebar.slider("최대 반지름", min_radius, 100, settings.get('max_radius'))
    
    # 설정 저장
    if st.sidebar.button("설정 저장"):
        config = {
            'debug_mode': debug_mode,
            'mode_min': mode_min,
            'mode_max': mode_max,
            'secondary_peak_ratio': secondary_peak_ratio,
            'min_radius': min_radius,
            'max_radius': max_radius
        }
        settings.save_settings(config)
        st.sidebar.success("설정이 저장되었습니다.")
    
    # 메인 화면
    if uploaded_file is not None:
        # 이미지 표시
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="입력 이미지", use_column_width=True)
        
        # ROI 입력
        rois = []
        for i in range(roi_count):
            st.subheader(f"ROI {i+1} 좌표 입력")
            col1, col2 = st.columns(2)
            
            with col1:
                x1 = st.number_input(f"ROI {i+1} - X1", 0, image.shape[1], 100)
                y1 = st.number_input(f"ROI {i+1} - Y1", 0, image.shape[0], 100)
            
            with col2:
                x2 = st.number_input(f"ROI {i+1} - X2", x1, image.shape[1], x1 + 100)
                y2 = st.number_input(f"ROI {i+1} - Y2", y1, image.shape[0], y1 + 100)
            
            rois.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        
        # 분석 실행
        if st.button("결함 검출 실행"):
            # 설정 준비
            config = {
                'debug_mode': debug_mode,
                'mode_min': mode_min,
                'mode_max': mode_max,
                'secondary_peak_ratio': secondary_peak_ratio,
                'min_radius': min_radius,
                'max_radius': max_radius
            }
            
            # 검출기 초기화
            detector = HoleDefectDetector(config=config)
            
            # 결과 저장
            results = []
            
            # 각 ROI 분석
            for idx, roi in enumerate(rois):
                st.subheader(f"ROI {idx+1} 분석 결과")
                
                # ROI 분석
                result = detector.analyze_roi(image, roi, idx)
                results.append(result)
                
                # 결함 여부 표시
                if result['is_defective']:
                    st.error(f"결함 감지: {', '.join(result['reasons'])}")
                else:
                    st.success("정상")
                
                # 통계 정보 표시
                st.write("픽셀 통계:")
                stats_to_show = ["Mean", "Median", "Mode", "Std Dev", "Has Secondary Peak"]
                for stat in stats_to_show:
                    if stat in result['stats']:
                        if isinstance(result['stats'][stat], (int, float)):
                            st.write(f"{stat}: {result['stats'][stat]:.2f}")
                        else:
                            st.write(f"{stat}: {result['stats'][stat]}")
                
                # 시각화 표시 (디버그 모드에서만)
                if debug_mode and result.get('visualization') is not None:
                    st.pyplot(result['visualization'])
            
            # 결과 저장
            if results:
                json_path = save_results_to_json(results)
                st.info(f"결과가 {json_path}에 저장되었습니다.")
                
                if debug_mode:
                    viz_paths = save_visualizations(results)
                    if viz_paths:
                        st.info(f"시각화 결과가 저장되었습니다.")

if __name__ == "__main__":
    main()
