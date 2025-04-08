import os
import logging
import json
from datetime import datetime

def save_results_to_json(results, output_dir="results"):
    """결과를 JSON 파일로 저장"""
    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON 파일 경로
    json_path = os.path.join(output_dir, f"detection_results_{timestamp}.json")
    
    # 시각화 객체는 직렬화할 수 없으므로 제거
    serializable_results = []
    for result in results:
        result_copy = result.copy()
        if 'visualization' in result_copy:
            del result_copy['visualization']
        serializable_results.append(result_copy)
    
    # JSON 저장
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    logging.info(f"Results saved to {json_path}")
    return json_path

def save_visualizations(results, output_dir="results/visualizations"):
    """시각화 결과 저장"""
    # 시각화 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_paths = []
    for result in results:
        if result.get('visualization') is not None:
            # 저장 경로
            fig_path = os.path.join(output_dir, f"roi_{result['roi_idx']}_{timestamp}.png")
            
            # 그림 저장
            result['visualization'].savefig(fig_path)
            saved_paths.append(fig_path)
            
            logging.info(f"Visualization saved to {fig_path}")
    
    return saved_paths
