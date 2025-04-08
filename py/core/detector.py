import numpy as np
import cv2
from scipy import stats
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class HoleDefectDetector:
    """구멍 결함 검출 클래스"""
    
    def __init__(self, config=None):
        """
        초기화 함수
        
        Args:
            config (dict, optional): 설정 딕셔너리
        """
        self.config = config or {}
        # 기본 설정값 정의
        self.default_config = {
            'mode_min': 10,
            'mode_max': 17,
            'secondary_peak_ratio': 0.6,
            'min_dist': 20,
            'param1': 50,
            'param2': 30,
            'min_radius': 5,
            'max_radius': 50,
            'debug_mode': False
        }
    
    def get_config(self, key):
        """설정값을 가져오는 함수"""
        return self.config.get(key, self.default_config.get(key))
    
    def analyze_circle_pixels(self, circle_pixels):
        """원 내부 픽셀 분석"""
        logger.debug(f"Analyzing {len(circle_pixels)} pixels")
        
        # 기본 통계 계산
        stats_dict = {
            "Mean": np.mean(circle_pixels),
            "Median": np.median(circle_pixels),
            "Mode": stats.mode(circle_pixels)[0],
            "Std Dev": np.std(circle_pixels),
            "Variance": np.var(circle_pixels),
            "Min": np.min(circle_pixels),
            "Max": np.max(circle_pixels),
            "Q1": np.percentile(circle_pixels, 25),
            "Q3": np.percentile(circle_pixels, 75),
            "Skewness": stats.skew(circle_pixels),
            "Kurtosis": stats.kurtosis(circle_pixels),
            "Dark pixels ratio (<50)": np.sum(circle_pixels < 50) / len(circle_pixels),
            "Bright pixels ratio (>150)": np.sum(circle_pixels > 150) / len(circle_pixels)
        }
        
        # 히스토그램 분석
        hist, bin_edges = np.histogram(circle_pixels, bins=50, range=(0, 255))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 피크 찾기
        peak_indices = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(circle_pixels) * 0.05:
                peak_indices.append(i)
        
        # 피크 값과 높이
        peaks = [(bin_centers[i], hist[i]) for i in peak_indices]
        
        # 모드에 가장 가까운 피크 찾기
        mode_bin_index = np.argmin(np.abs(bin_centers - stats_dict["Mode"]))
        mode_peak_height = hist[mode_bin_index]
        
        # 중요한 부 피크 확인
        has_significant_secondary_peak = False
        secondary_peak_info = ""
        secondary_peak_ratio = self.get_config('secondary_peak_ratio')
        
        for peak_val, peak_height in peaks:
            # 모드 가까운 피크는 무시
            if abs(peak_val - stats_dict["Mode"]) < 10:
                continue
                
            # 모드 피크의 일정 비율 이상인 피크가 있는지 확인
            if peak_height > mode_peak_height * secondary_peak_ratio:
                has_significant_secondary_peak = True
                secondary_peak_info = f"Secondary peak at {peak_val:.1f} with height {peak_height}"
                break
        
        # 다중 피크 정보 저장
        stats_dict["Has Secondary Peak"] = has_significant_secondary_peak
        stats_dict["Secondary Peak Info"] = secondary_peak_info
        
        # 히스토그램 데이터 저장
        histogram_data = {
            'hist': hist,
            'bin_centers': bin_centers,
            'peaks': peaks
        }
        
        return stats_dict, histogram_data
    
    def detect_defect(self, stats_dict):
        """결함 탐지 로직"""
        is_defective = False
        defect_reasons = []
        
        # 로깅
        logger.debug(f"Analyzing stats: {stats_dict}")

        # 검사 1: 모드 값 범위 확인
        mode_min = self.get_config('mode_min')
        mode_max = self.get_config('mode_max')
        if not (mode_min <= stats_dict["Mode"] <= mode_max):
            is_defective = True
            defect_reasons.append(f"Mode value {stats_dict['Mode']:.2f} out of normal range ({mode_min}-{mode_max})")

        # 검사 2: 중요한 부 피크 확인
        if stats_dict["Has Secondary Peak"]:
            is_defective = True
            defect_reasons.append(f"Multiple peaks detected: {stats_dict['Secondary Peak Info']}")

        return is_defective, defect_reasons
    
    def visualize_results(self, roi_idx, roi_color, center, radius, is_defective, 
                         defect_reasons, stats_dict, circle_pixels, histogram_data):
        """결과 시각화"""
        plt.figure(figsize=(12, 4))
        
        # 이미지에 원 표시
        plt.subplot(1, 2, 1)
        title = f'ROI {roi_idx} - {"DEFECTIVE" if is_defective else "NORMAL"}'
        if is_defective:
            title += f'\nReasons: {", ".join(defect_reasons)}'
        plt.title(title, color='red' if is_defective else 'green', fontsize=10)
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        circle_color = (255, 0, 0) if is_defective else (0, 255, 0)
        cv2.circle(roi_rgb, (center[0], center[1]), radius, circle_color, 2)
        plt.imshow(roi_rgb)
        plt.axis('off')
        
        # 히스토그램 시각화
        plt.subplot(1, 2, 2)
        hist_title = f'ROI {roi_idx} - Pixel Value Distribution\n'
        hist_title += f'Mode: {stats_dict["Mode"]:.1f}, Mean: {stats_dict["Mean"]:.1f}, '
        hist_title += f'Median: {stats_dict["Median"]:.1f}, StdDev: {stats_dict["Std Dev"]:.1f}'
        if stats_dict["Has Secondary Peak"]:
            hist_title += "\nMultiple peaks detected!"
        plt.title(hist_title, fontsize=10)
        
        # 히스토그램 플롯
        hist = histogram_data['hist']
        bin_centers = histogram_data['bin_centers']
        peaks = histogram_data['peaks']
        
        n, bins, patches = plt.hist(circle_pixels, bins=50, range=(0, 255), color='blue', alpha=0.7)
        
        # 피크 하이라이트
        for peak_val, peak_height in peaks:
            bin_idx = np.argmin(np.abs(bin_centers - peak_val))
            if abs(peak_val - stats_dict["Mode"]) < 10:
                # 메인 피크
                patches[bin_idx].set_facecolor('green')
            else:
                # 부 피크
                patches[bin_idx].set_facecolor('red')
        
        plt.axvline(x=150, color='r', linestyle='--', label='Threshold (150)')
        plt.axvline(x=stats_dict["Mean"], color='g', linestyle='--', label='Mean')
        plt.axvline(x=stats_dict["Median"], color='y', linestyle='--', label='Median')
        plt.axvline(x=stats_dict["Mode"], color='purple', linestyle='--', label='Mode')
        plt.xlabel('Pixel Value')
        plt.ylabel('Pixel Count')
        plt.legend()
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def analyze_roi(self, image, roi_points, roi_idx):
        """ROI 분석"""
        from .image_processor import ImageProcessor
        
        # 이미지 처리기 초기화
        img_proc = ImageProcessor()
        
        # ROI 추출
        roi_color, roi_gray, (x_min, y_min, x_max, y_max) = img_proc.extract_roi(image, roi_points)
        
        # 가우시안 블러 적용
        roi_blurred = img_proc.apply_gaussian_blur(roi_gray)
        
        # 원 검출
        circles = img_proc.detect_circles(
            roi_blurred,
            min_dist=self.get_config('min_dist'),
            param1=self.get_config('param1'),
            param2=self.get_config('param2'),
            min_radius=self.get_config('min_radius'),
            max_radius=self.get_config('max_radius')
        )
        
        if circles is None:
            logger.warning(f"No circles found in ROI {roi_idx}")
            return {
                'roi_idx': roi_idx,
                'is_defective': False,
                'reasons': ["No circles detected"],
                'stats': {},
                'visualization': None
            }
            
        # 가장 큰 원 찾기
        largest_circle = circles[0][0]
        
        # 원 중심과 반지름
        center_x, center_y, radius = largest_circle
        
        # 원형 마스크 생성
        mask = img_proc.create_circular_mask(roi_gray.shape, (center_x, center_y), radius)
        
        # 원 내부 픽셀 추출
        circle_pixels = roi_gray[mask == 255]
        
        # 픽셀 분석
        stats_dict, histogram_data = self.analyze_circle_pixels(circle_pixels)
        
        # 결함 검출
        is_defective, defect_reasons = self.detect_defect(stats_dict)
        
        # 결과 로깅
        if is_defective:
            logger.warning(f"ROI {roi_idx} is DEFECTIVE: {', '.join(defect_reasons)}")
        else:
            logger.info(f"ROI {roi_idx} is NORMAL")
        
        # 통계 로깅
        for stat_name, stat_value in stats_dict.items():
            if isinstance(stat_value, (int, float)):
                logger.debug(f"{stat_name}: {stat_value:.2f}")
            else:
                logger.debug(f"{stat_name}: {stat_value}")
        
        # 결과 시각화 (디버그 모드에서만)
        visualization = None
        if self.get_config('debug_mode'):
            visualization = self.visualize_results(
                roi_idx, roi_color, (center_x, center_y), radius, 
                is_defective, defect_reasons, stats_dict, 
                circle_pixels, histogram_data
            )
        
        # 결과 반환
        result = {
            'roi_idx': roi_idx,
            'roi_info': {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            },
            'circle_info': {
                'center': (center_x, center_y),
                'radius': radius
            },
            'is_defective': is_defective,
            'reasons': defect_reasons,
            'stats': stats_dict,
            'visualization': visualization
        }
        
        return result
