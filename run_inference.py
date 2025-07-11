# run_inference.py
import argparse
import torch
from PIL import Image
import cv2
import numpy as np
import pandas as pd

def run_inference(
    image_path: str,
    model_path: str,
    conf_thres: float,
    iou_thres: float,
    img_size: int
):
    # 1. 모델 로드
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path=model_path, force_reload=False
    )
    model.conf = conf_thres
    model.iou  = iou_thres
    print(f"[INFO] Loaded model with classes: {model.names}\n")

    # 2. 이미지 로드 & 원본 사이즈 출력
    img = Image.open(image_path).convert('RGB')
    print(f"[INFO] Loaded image '{image_path}' size: {img.size}")

    # 3. 추론: !size 옵션으로 resize 강제
    results = model(img, size=img_size)
    results.print()  # YOLOv5 기본 출력

    # 4. raw tensor 와 pandas DataFrame 출력
    dets = results.xyxy[0]  # tensor(x1, y1, x2, y2, conf, cls)
    print(f"[DEBUG] Raw detections tensor:\n{dets}\n")

    df = results.pandas().xyxy[0]
    print(f"[DEBUG] Pandas DataFrame of detections:\n{df}\n")

    if dets.shape[0] == 0:
        print("[WARN] No objects detected. → image에 검출 가능한 객체가 있는지, threshold를 더 낮춰보세요.")
    else:
        print("[INFO] Detected objects:")
        for idx, row in df.iterrows():
            print(f" - {row['name']} @ {row['confidence']:.3f} "
                  f"({row['xmin']:.0f},{row['ymin']:.0f})→({row['xmax']:.0f},{row['ymax']:.0f})")
    
    # 5. render() 후 OpenCV로 보기
    annotated = results.render()[0]  # RGB numpy
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 Inference', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. 결과 저장
    results.save()  # runs/detect/expN/

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YOLOv5 Inference with debug")
    p.add_argument('--image', type=str, required=True,
                   help='Path to input image')
    p.add_argument('--model', type=str, default='yolov5/runs/train/yolov5m_food8/weights/last.pt',
                   help='Path to .pt model')
    p.add_argument('--conf', type=float, default=0.1,
                   help='Confidence threshold')
    p.add_argument('--iou', type=float, default=0.45,
                   help='NMS IoU threshold')
    p.add_argument('--img-size', type=int, default=640,
                   help='Inference image size (pixels)')
    args = p.parse_args()

    run_inference(
        image_path=args.image,
        model_path=args.model,
        conf_thres=args.conf,
        iou_thres=args.iou,
        img_size=args.img_size
    )
