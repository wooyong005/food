# # services/yolo_detector.py
# # YOLO 모델 로딩 및 음식 박스 추출 함수 등 설계
# def detect_food_objects(image_path):
#     # TODO: YOLOv5 등으로 객체 검출
#     return ["cropped_image1.jpg", "cropped_image2.jpg"]

import torch
from PIL import Image
import cv2
import numpy as np

def run_inference(image_path: str, model_path: str = 'yolov5/runs/train/yolov5m_food8/weights/last.pt', conf_threshold: float = 0.25):
    """
    YOLOv5 모델로 이미지 추론 수행 후 결과 출력 및 저장, 이미지 창으로 보여주기
    
    Args:
        image_path (str): 추론할 이미지 파일 경로
        model_path (str): 학습된 모델 파일 경로
        conf_threshold (float): confidence threshold (기본값 0.25)
    """
    # 1. 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.conf = conf_threshold  # confidence threshold 설정
    
    # 2. 이미지 열기
    img = Image.open(image_path)
    
    # 3. 추론 수행
    results = model(img)
    
    # 4. 결과 출력
    results.print()
    
    # 5. 결과 이미지 저장 (runs/detect/exp 등)
    results.save()
    
    # 6. 결과 이미지 화면에 표시
    result_img = np.squeeze(results.render())  # 결과 이미지 배열
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)  # RGB->BGR 변환(OpenCV용)
    
    cv2.imshow('YOLOv5 Inference', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 테스트용 직접 실행 시
    test_image = 'test.jpg'
    run_inference(test_image)
