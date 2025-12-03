import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

def top3_keywords_from_context(text: str, top_n: int = 3,
                                      ngram_range=(1,1),
                                      max_features: int = 1000) -> List[str]:
    """
    주어진 영어 텍스트(context)에서 TF-IDF를 이용해 상위 top_n 키워드를 반환하는 단일 함수.
    모든 처리 단계는 아래 주석에서 차근차근 설명합니다.
    
    Args:
        text: 분석할 영문 텍스트(하나의 문서).
        top_n: 반환할 키워드 개수 (기본 3).
        ngram_range: 단어 단위 n-gram 범위 (기본 (1,1) -> unigram).
        max_features: TF-IDF 벡터화시 고려할 최대 피처 수 (성능 조정용).
    Returns:
        키워드 문자열 리스트 (top_n, 내림차순 중요도).
    """

    # 1) 입력 검증: 비어있으면 빈 리스트 반환
    if not text or not text.strip():
        return []

    # 2) TF-IDF 벡터라이저 준비
    #    - stop_words='english'로 기본 영어 불용어 제거
    #    - ngram_range로 단어/바이그램 등 선택 가능
    #    - max_features는 희귀/빈번한 단어 수 조절 (메모리/속도)
    vectorizer = TfidfVectorizer(stop_words='english',
                                 ngram_range=ngram_range,
                                 max_features=max_features)

    # 3) 단일 문서를 리스트로 래핑해서 fit_transform 수행
    #    (sklearn은 문서 리스트를 입력으로 받음)
    tfidf_matrix = vectorizer.fit_transform([text])

    # 4) 피처(토큰) 이름 목록 얻기
    feature_names = vectorizer.get_feature_names_out()

    # 5) TF-IDF 행 벡터(문서 1개)에서 각 피처의 점수 추출
    #    tfidf_matrix는 희소행렬이므로 toarray로 변환해 접근
    scores = tfidf_matrix.toarray()[0]

    # 6) (인덱스, 점수)쌍을 만들어 점수 기준으로 내림차순 정렬
    #    0 점수(불필요 토큰)는 걸러냄
    idx_score_pairs = [(i, score) for i, score in enumerate(scores) if score > 0]
    idx_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 7) 상위 top_n 인덱스 선택, 인덱스를 토큰으로 매핑
    top_indices = [i for i, _ in idx_score_pairs[:top_n]]
    top_keywords = [feature_names[i] for i in top_indices]

    # 8) 결과 반환 (리스트)
    return top_keywords

# 웹 드라이버
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

url ='https://huggingface.co/papers/week/2025-W47'
driver.get(url)
time.sleep(3)
print('브라우져가 성공적으로 열렸습니다.')

for i in range(1,106):
    # 문서 제목 클릭 : title
    title = driver.find_element(By.XPATH,f'/html/body/div[1]/main/div[2]/section/div[2]/article[{i}]/div[2]/div/div[2]/h3/a')
    title.click()
    time.sleep(2)

    # 페이지 소스 가져오기
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    huggingface_url = driver.current_url
    print(f'허깅페이스 링크 : {huggingface_url}')
    
    paper_name = soup.select_one('body > div > main > div > section.pt-8.border-gray-100.md\:col-span-7.sm\:pb-16.lg\:pb-24.relative > div > div.pb-10.md\:pt-3 > h1').get_text(strip=True)
    print(f'제목 : {paper_name}')

    # href 값 추출
    github_url = None   # 기본값

    # 페이지 내의 모든 <a> 태그 탐색
    links = driver.find_elements(By.CSS_SELECTOR, "section a")
    for link in links:
        href = link.get_attribute("href")
        if href and "github.com" in href:
            github_url = href
            break  # 첫 번째 GitHub URL만 사용
    if github_url is None:
        print("⚠ GitHub 링크 없음")
    else:
        print(f"깃허브 링크 : {github_url}")
        
    upvote = soup.select_one('body > div > main > div > section.pt-8.border-gray-100.md\:col-span-5.pt-6.lg\:pt-28.pb-24.md\:pl-6.md\:border-l > div.hidden.flex-wrap.items-start.gap-2.md\:flex > div > div > a > div > div').get_text(strip=True)
    print(f'좋아요 갯수 : {upvote}')
    
    # context (본문 내용) 추출
    context = soup.select_one('body > div > main > div > section.pt-8.border-gray-100.md\:col-span-7.sm\:pb-16.lg\:pb-24.relative > div > div.pb-8.pr-4.md\:pr-16 > div > p').get_text(strip=True)
    print(f'내용 : {context[:40]}')

    
    # Top 3 키워드 추출 : 자주 언급 되는 단어
    tags = top3_keywords_from_context(context)
    print(f'Top 3 키워드 : {top3_keywords_from_context(context)}')
    
    # JSON 구조 만들기
    paper_json = {
        "context": context,
        "metadata": {
            "paper_name": paper_name,
            "github_url": github_url,
            "huggingface_url": huggingface_url,
            "upvote": upvote,
            "tags": tags
        }
    }
    # 파일로 저장
    with open(f'doc2547{str(i).zfill(3)}.json', 'w', encoding='utf-8') as f:
        json.dump(paper_json, f, ensure_ascii=False, indent=2)
    driver.back()
    time.sleep(2)
    
    print(f"데이터가 doc2547{str(i).zfill(3)}.json 에 저장되었습니다.")
driver.quit()
