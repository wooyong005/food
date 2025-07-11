from ultralytics import YOLO

def main():
    # 모델 불러오기 (yolov8n.pt: nano 버전, 빠르고 가벼움)
    model = YOLO("yolov8n.pt")

    # 학습 시작
    model.train(
        data="data.yaml",          # data.yaml 경로
        epochs=100,                # 에폭 수
        imgsz=640,                 # 이미지 크기
        name="food10_yolov8n",     # 결과 저장 폴더 이름
        batch=16,                  # 배치 사이즈 (메모리에 따라 조정 가능)
        workers=4,                 # 데이터 로딩 스레드 수 (CPU 코어 수에 따라 조정)
        device=0                   # GPU 사용 (0), CPU 강제 사용시 'cpu'
    )

if __name__ == "__main__":
    main()
    
#추론

