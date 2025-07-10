import os
import pandas as pd
from PIL import Image

def yolo_to_csv(images_dir, labels_dir, class_file):
    rows = []
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            base_name = label_file[:-4]
            # jpg, png 등 이미지 확장자 중 하나 찾기
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                temp_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
            if image_path is None:
                continue

            img = Image.open(image_path)
            w, h = img.size

            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, bw, bh = map(float, line.strip().split())
                    xmin = int((x - bw / 2) * w)
                    ymin = int((y - bh / 2) * h)
                    xmax = int((x + bw / 2) * w)
                    ymax = int((y + bh / 2) * h)
                    rows.append([os.path.basename(image_path), xmin, ymin, xmax, ymax, class_names[int(cls)]])

    df = pd.DataFrame(rows, columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    df.to_csv("labels.csv", index=False)
    print("labels.csv 파일이 생성되었습니다!")

# 변수 설정
images_dir = "./images/train"
labels_dir = "./labels/train"
class_file = "classes/classes.txt"

# 실행
yolo_to_csv(images_dir, labels_dir, class_file)
