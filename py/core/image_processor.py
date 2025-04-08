import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """이미지 처리를 위한 클래스"""
    
    @staticmethod
    def load_image(image_path):
        """이미지 로드"""
        logger.debug(f"Loading image from {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image from {image_path}")
            raise ValueError(f"Could not load the image from {image_path}")
        return image
    
    @staticmethod
    def extract_roi(image, roi_points):
        """ROI 추출"""
        # ROI 경계 계산
        x_min = min(p[0] for p in roi_points)
        y_min = min(p[1] for p in roi_points)
        x_max = max(p[0] for p in roi_points)
        y_max = max(p[1] for p in roi_points)
        
        # ROI 추출 및 그레이스케일 변환
        roi_color = image[y_min:y_max, x_min:x_max].copy()
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        
        return roi_color, roi_gray, (x_min, y_min, x_max, y_max)
    
    @staticmethod
    def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
        """가우시안 블러 적용"""
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def detect_circles(image, dp=1, min_dist=20, param1=50, param2=30, 
                      min_radius=5, max_radius=50):
        """원 검출"""
        circles = cv2.HoughCircles(
            image, 
            cv2.HOUGH_GRADIENT, 
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is None:
            logger.warning("No circles detected")
            return None
            
        # uint16으로 변환
        return np.uint16(np.around(circles))
    
    @staticmethod
    def create_circular_mask(image_shape, center, radius):
        """원형 마스크 생성"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return mask
    
    @staticmethod
    def draw_circle_on_image(image, center, radius, color=(0, 255, 0), thickness=2):
        """이미지에 원 그리기"""
        img_copy = image.copy()
        cv2.circle(img_copy, center, radius, color, thickness)
        return img_copy
