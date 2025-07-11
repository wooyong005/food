import os
import random
import shutil

def find_image_label_pairs(source_dir):
    pairs = []
    for folder_name in os.listdir(source_dir):
        if not folder_name.endswith("_image_processed"):
            continue

        # 라벨링 폴더 이름 찾기 (ex: bibimbap_image_labeling0)
        label_folder_name = None
        for sub_name in os.listdir(source_dir):
            if sub_name.startswith(folder_name.replace("_image_processed", "_image_labeling")):
                label_folder_name = sub_name
                break

        if label_folder_name is None:
            print(f"[경고] 라벨 폴더를 찾을 수 없습니다: {folder_name}")
            continue

        image_folder = os.path.join(source_dir, folder_name)
        label_folder = os.path.join(source_dir, label_folder_name)

        for file in os.listdir(image_folder):
            name, ext = os.path.splitext(file)
            label_file = f"{name}.txt"
            image_path = os.path.join(image_folder, file)
            label_path = os.path.join(label_folder, label_file)

            if os.path.exists(label_path):
                pairs.append((image_path, label_path))

    return pairs


def split_yolo_dataset_recursive(
    source_dir="raw_data",
    output_dir="dataset",
    train_ratio=0.8,
    seed=42,
    log_file="split_log.txt"
):
    # 현재 실행 중인 스크립트의 절대경로 기준
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(current_dir, source_dir)
    output_dir = os.path.join(current_dir, output_dir)
    log_file = os.path.join(current_dir, log_file)

    random.seed(seed)

    all_pairs = find_image_label_pairs(source_dir)
    total = len(all_pairs)
    print(f"[INFO] 총 유효한 이미지-라벨 쌍: {total}개")

    if total == 0:
        print("[ERROR] 유효한 이미지-라벨 쌍이 없습니다.")
        return

    random.shuffle(all_pairs)
    split_idx = int(total * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    def copy_pairs(pairs, mode):
        img_out = os.path.join(output_dir, "images", mode)
        label_out = os.path.join(output_dir, "labels", mode)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

        for img_path, label_path in pairs:
            shutil.copy(img_path, os.path.join(img_out, os.path.basename(img_path)))
            shutil.copy(label_path, os.path.join(label_out, os.path.basename(label_path)))

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    summary = f"[분할 완료]\n" \
              f"총 샘플 수: {total}\n" \
              f"Train: {len(train_pairs)}개\n" \
              f"Val: {len(val_pairs)}개\n" \
              f"데이터 저장 위치: {output_dir}/images/train,val / labels/train,val"
    print(summary)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    split_yolo_dataset_recursive()


