import os
from PIL import Image

def preprocess_all_images_in_folders(base_path=".", output_suffix="_processed", size=(244, 244)):
    folder_list = [f for f in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, f)) and f.endswith("_image")]

    for folder in folder_list:
        input_folder = os.path.join(base_path, folder)
        output_folder = os.path.join(base_path, folder + output_suffix)
        os.makedirs(output_folder, exist_ok=True)

        count = 1
        for filename in os.listdir(input_folder):
            if count > 50:  # ✅ 폴더당 최대 50장만 처리
                break

            filepath = os.path.join(input_folder, filename)
            try:
                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                    img = img.resize(size)
                    label = folder.replace("_image", "")
                    save_path = os.path.join(output_folder, f"{label}{count}.jpg")
                    img.save(save_path)
                    print(f"✅ 저장 완료: {save_path}")
                    count += 1
            except Exception as e:
                print(f"⛔ 처리 실패: {filename} ({e})")

# 실행
preprocess_all_images_in_folders()

# from PIL import Image ← 이 줄을 사용하려면 아래 명령어로 Pillow 라이브러리를 설치해야 합니다.
#    pip install pillow    ← 위 명령어를 터미널(CMD)에 입력해 주세요.

#  이 코드는 현재 파이썬 파일이 있는 폴더 내부의 모든 이미지 폴더(이름이 *_image로 끝나는)를 자동으로 처리합니다.


#  처리 방식:
#    - 폴더 이름 뒤에 "_processed"를 붙여 새로운 폴더를 생성하고,
#    - 모든 이미지를 244x244 크기로 통일하며,
#    - 확장자를 .jpg로 저장합니다.
#    - 파일 이름은 폴더 이름 기반으로 자동 생성됩니다 (예: bulgogi1.jpg, bibimbap1.jpg 등).
#    - 코딩에 따라 폴더에 있는 이미지 50개 제한으로 처리합니다.


#  주의사항:
#    - 이미지 처리 중 오류가 발생할 수도 있으므로, 원본 이미지 폴더를 반드시 백업해 두시길 권장합니다.