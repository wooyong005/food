import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os

# 1. YOLOv11 모델 정의 import (YOLOv11 레포에 맞게 경로/클래스명 수정)
# from models.yolo import Model  # 예시, 실제 YOLOv11 레포에 맞게 import

def load_model(weights_path, device='cpu'):
    # 2. YOLOv11 모델 정의 및 config 파일 경로를 실제 환경에 맞게 수정
    # model = Model(cfg='yolov11.yaml').to(device)
    # checkpoint = torch.load(weights_path, map_location=device)
    # model.load_state_dict(checkpoint['model'].state_dict())
    # model.eval()
    # return model

    # 아래는 예시 코드입니다. 실제 YOLOv11 레포의 모델 로딩 방식을 참고해서 수정하세요!
    model = torch.load(weights_path, map_location=device, weights_only=False)
    model.eval()
    return model

def preprocess_image(img_path, device):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((640, 640))  # YOLOv11 입력 크기에 맞게 수정
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    return img_tensor

def detect_image(model, img_path, device='cpu', conf_thres=0.25):
    img_tensor = preprocess_image(img_path, device)
    with torch.no_grad():
        results = model(img_tensor)
    # 결과 후처리 및 시각화 (YOLOv11 레포에 맞게 수정)
    print(f'Predictions for {img_path}:', results)
    # 결과 저장 등 추가 구현 필요

def detect_images_in_folder(model, folder_path, device='cpu', conf_thres=0.25):
    img_extensions = ['.jpg', '.jpeg', '.png']
    for img_file in os.listdir(folder_path):
        if Path(img_file).suffix.lower() in img_extensions:
            img_path = os.path.join(folder_path, img_file)
            print(f'Processing {img_path} ...')
            detect_image(model, img_path, device, conf_thres)

def detect_video(model, video_path, device='cpu', conf_thres=0.25):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # YOLOv11 입력 크기에 맞게 수정
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            results = model(img_tensor)
        print('Video frame prediction:', results)
        # 결과 시각화/저장 추가 가능
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='C:/Users/user/Desktop/airecipe/model/runs/train/train/weights/best.pt', help='model weights path')
    parser.add_argument('--source', type=str, default='C:/Users/user/Desktop/airecipe/dataset/train/images', help='image or video file path')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    source = Path(args.source)
    if source.is_dir():
        detect_images_in_folder(model, str(source), args.device, args.conf)
    elif source.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        detect_image(model, str(source), args.device, args.conf)
    elif source.suffix.lower() in ['.mp4', '.avi', '.mov']:
        detect_video(model, str(source), args.device, args.conf)
    else:
        print('지원하지 않는 파일 형식 또는 경로입니다.')

if __name__ == '__main__':
    main()
