import os
import json

class Settings:
    """설정 관리 클래스"""
    
    def __init__(self, config_file=None):
        """
        초기화 함수
        
        Args:
            config_file (str, optional): 설정 파일 경로
        """
        # 기본 설정
        self.default_settings = {
            # 디버깅 설정
            'debug_mode': False,
            
            # 로깅 설정
            'log_level': 'INFO',
            'log_dir': 'logs',
            
            # 이미지 처리 설정
            'image_dir': 'data/images',
            
            # 검출 설정
            'mode_min': 10,
            'mode_max': 17,
            'secondary_peak_ratio': 0.6,
            'min_dist': 20,
            'param1': 50,
            'param2': 30,
            'min_radius': 5,
            'max_radius': 50,
            
            # 결과 설정
            'results_dir': 'results',
            'visualizations_dir': 'results/visualizations'
        }
        
        # 사용자 설정 로드
        self.user_settings = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.user_settings = json.load(f)
    
    def get(self, key, default=None):
        """설정값 가져오기"""
        # 사용자 설정에서 먼저 찾기
        if key in self.user_settings:
            return self.user_settings[key]
        
        # 기본 설정에서 찾기
        if key in self.default_settings:
            return self.default_settings[key]
        
        # 기본값 반환
        return default
    
    def save_settings(self, settings_dict, config_file='config/user_settings.json'):
        """설정 저장하기"""
        # 설정 디렉토리 확인
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # 현재 설정에 새 설정 병합
        self.user_settings.update(settings_dict)
        
        # JSON 파일로 저장
        with open(config_file, 'w') as f:
            json.dump(self.user_settings, f, indent=4)
        
        return config_file
