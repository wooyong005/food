from duckduckgo_search import DDGS
import requests, os

def search_and_save_images(query, max_results=200, folder_name="images"):
    os.makedirs(folder_name, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_results)
        for i, result in enumerate(results, 1):
            url = result["image"]
            try:
                ext = url.split('.')[-1].split('?')[0][:4]
                filename = f"{folder_name}/img_{i}.{ext}"
                res = requests.get(url, timeout=10)
                with open(filename, 'wb') as f:
                    f.write(res.content)
                print(f"✅ 저장 완료: {filename}")
            except Exception as e:
                print(f"⛔ 저장 실패 ({e})")

# 예시 실행
search_and_save_images("된장찌개", max_results=200, folder_name="된장찌개이이_이미지")