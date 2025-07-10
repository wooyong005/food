import time
import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO

# 유튜브 썸네일 도메인 필터링
YOUTUBE_DOMAINS = ["ytimg.com", "youtube.com", "img.youtube.com", "i.ytimg.com"]

def search_and_save_images(query, max_results=500, folder_name="images"):
    os.makedirs(folder_name, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_results)
        saved = 0

        for i, result in enumerate(results, 1):
            url = result["image"]
            # 유튜브 도메인 필터링
            if any(domain in url for domain in YOUTUBE_DOMAINS):
                print(f"⛔ 유튜브 썸네일 필터링: {url}")
                continue

            try:
                res = requests.get(url, timeout=10)
                img = Image.open(BytesIO(res.content)).convert("RGB")

                ext = url.split('.')[-1].split('?')[0][:4]
                filename = f"{folder_name}/img_{saved+1}.{ext}"
                img.save(filename)
                print(f"✅ 저장 완료: {filename}")
                saved += 1

                if saved >= max_results:
                    break

            except Exception as e:
                print(f"⛔ 저장 실패: {url} ({e})")

            time.sleep(0.5)  # 요청 사이 딜레이 (속도 조절)

# 예시 실행
search_and_save_images("저장받을거", max_results=500, folder_name="저장받을거_image")
