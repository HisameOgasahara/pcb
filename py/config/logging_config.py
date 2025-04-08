import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir='logs', log_level=logging.INFO, debug_mode=False):
    """로깅 설정"""
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 레벨 설정
    if debug_mode:
        log_level = logging.DEBUG
    
    # 로그 포맷 설정
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 추가
    log_file = os.path.join(log_dir, 'hole_detection.log')
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # 디버그 모드 메시지
    if debug_mode:
        logging.info("Debug mode is enabled")
    
    return root_logger
