import requests

API_KEY = 'AIzaSyCh26s4aSUQgjF_u5ksLIVwYzh3-HjH_os'
CSE_ID = '41e86263bd8d3482d'

def is_youtube_thumbnail(url: str) -> bool:
    # 유튜브 썸네일의 도메인을 필터링
    return any(domain in url for domain in [
        "ytimg.com",
        "youtube.com",
        "img.youtube.com",
        "i.ytimg.com"
    ])

def search_images_without_youtube(query, max_images=30):
    seen_urls = set()
    results = []
    start_index = 1

    while len(results) < max_images and start_index <= 90:
        params = {
            'q': query,
            'cx': CSE_ID,
            'key': API_KEY,
            'searchType': 'image',
            'num': 10,
            'start': start_index
        }
        res = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        data = res.json()

        items = data.get('items', [])
        for item in items:
            img_url = item['link']
            if is_youtube_thumbnail(img_url):
                print(f"⛔ 유튜브 썸네일 제외: {img_url}")
                continue
            if img_url not in seen_urls:
                seen_urls.add(img_url)
                results.append(img_url)
                print(f"✅ 사용 이미지: {img_url}")
                if len(results) >= max_images:
                    break
        start_index += 10

    return results

# 예시 실행
# image_links = search_images_without_youtube("불고기", max_images=1000)

import os

def save_images(image_links, folder_name="images"):
    os.makedirs(folder_name, exist_ok=True)

    for idx, url in enumerate(image_links, start=1):
        try:
            response = requests.get(url, timeout=10)
            ext = url.split('.')[-1].split('?')[0][:4]  # 확장자 추정
            filename = f"{folder_name}/img_{idx}.{ext}"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ 저장 완료: {filename}")
        except Exception as e:
            print(f"⛔ 저장 실패: {url} ({e})")
            
            
all_links = set()
keywords = ["제육볶음", "jeyuk bokkeum", "korean spicy pork stir fry", "spicy pork bulgogi"]

for kw in keywords:
    links = search_images_without_youtube(kw, max_images=1000)
    all_links.update(links)

save_images(list(all_links), folder_name="제육볶음_이미지")