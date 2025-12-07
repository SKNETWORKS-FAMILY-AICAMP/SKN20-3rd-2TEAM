import os
import json
import re
import time
import requests
from typing import List, Dict

from bs4 import BeautifulSoup
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from keybert import KeyBERT

# NLTK 불용어 다운로드
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("[PREPARE] NLTK 데이터 다운로드 중...")
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

# HTTP 헤더 설정
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"
}

# 불용어 로드
nltk_stopwords = set(stopwords.words('english'))
# 도메인 특화 불용어 추가 (선택적)
custom_stopwords = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "been", "be",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "we", "our", "their", "they", "it", "its", "which", "who", "when",
    "where", "why", "how", "what", "if", "than", "such", "into", "through",
    # 논문에서 너무 많이 나오는 단어들
    "paper", "propose", "present", "show", "demonstrate", "using", "used",
    "approach", "method", "model", "based", "results", "work",
    "task", "tasks", "result", "results", "data"
}

all_stopwords = nltk_stopwords.union(custom_stopwords)


# KeyBERT를 사용하여 keyword 추출
def extract_keywords(text: str, top_n: int = 3) -> List[str]:
    """
    KeyBERT를 사용한 키워드 추출

    Args:
        text: 논문 Abstract
        top_n: 추출할 키워드 개수

    Returns:
        키워드 리스트
    """
    if not text or len(text.split()) < 10:
        return [f"keyword{i+1}" for i in range(top_n)]

    try:
        # KeyBERT 모델 초기화
        kw_model = KeyBERT()

        # 키워드 추출 (불용어 제거)
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),   # 1~2단어 키워드
            stop_words=list(all_stopwords),
            top_n=top_n,
            use_maxsum=True     # 다양성 확보
        )

        # (keyword, score) 튜플에서 keyword만 추출
        result = [kw[0] for kw in keywords]

        # 부족한 키워드는 기본값으로 채우기
        while len(result) < top_n:
            result.append(f"keyword{len(result)+1}")

        return result[:top_n]

    except Exception as e:
        print(f"[WARNING]  키워드 추출 실패: {e}")
        return [f"keyword{i+1}" for i in range(top_n)]


def get_with_retry(url: str, max_retries: int = 3):
    """
    재시도 로직이 포함된 HTTP 요청

    Args:
        url: 요청 URL
        max_retries: 최대 재시도 횟수

    Returns:
        requests.Response 또는 None
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)

            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                print("[ERROR]  429 에러 (Too Many Requests), 대기 중...")
                time.sleep(5)

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[FATAL] 요청 실패: {e}")

        time.sleep(2)

    return None


def fetch_weekly_papers(year: int, week: int) -> List[Dict[str, str]]:
    """
    HuggingFace DailyPapers Weekly 페이지에서 논문 목록 추출

    Args:
        year (int): 연도
        week (int): 주차 (1~52)

    Returns:
        List[Dict[str, str]]: 논문 URL과 제목 리스트
    """
    week_str = f"{year}-W{week:02d}"
    weekly_url = f"https://huggingface.co/papers/week/{week_str}"

    print(f"\n[FETCH] {week_str} 논문 목록 가져오는 중...")

    response = get_with_retry(weekly_url)
    if response is None:
        print(f"[FATAL] 페이지 로드 실패: {weekly_url}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    # 논문 링크 추출 (CSS Selector: a.line-clamp-3)
    paper_links = []
    for link in soup.select("a.line-clamp-3"):
        href = link.get("href")
        title = link.get_text(strip=True)

        if href:
            full_url = f"https://huggingface.co{href}"
            paper_links.append({"title": title, "url": full_url})

    print(f"[CHECK] 논문 {len(paper_links)}개 발견")
    return paper_links


def fetch_paper_details(paper_url: str) -> Dict[str, any]:
    """
    개별 논문 상세 페이지에서 Abstract, Authors, GitHub URL, Upvote 추출

    Args:
        paper_url (str): _description_

    Returns:
        Dict: 논문 상세 정보
            - context(abstract) (str): 논문 초록 (여러 <p> 태그 결합)
            - github_url (str): GitHub 레포지토리 URL (없으면 빈 문자열)
            - authors (List[str]): 저자들
            - upvote (int): 추천 수 (추출 실패 시 0)
    """
    response = get_with_retry(paper_url)
    if response is None:
        return {"context": "", "authors": [], "github_url": "", "upvote": 0}

    soup = BeautifulSoup(response.content, "html.parser")

    # Abstract 추출 (여러 <p> 태그를 결합)
    abstract_section = soup.select_one("section div")
    abstract = ""
    if abstract_section:
        paragraphs = abstract_section.find_all("p")
        abstract = " ".join([p.get_text(strip=True) for p in paragraphs])

    # Authors 추출
    authors = []
    author_links = soup.select(
        "div.relative.flex.flex-wrap.items-center.gap-2.text-base.leading-tight a"
    )
    for link in author_links:
        author_name = link.get_text(strip=True)
        if author_name and "huggingface.co" not in author_name:
            authors.append(author_name)

    # GitHub URL 추출 (선택적)
    github_link = soup.select_one('a[href*="github.com"]')
    github_url = github_link["href"] if github_link else ""

    # Upvote 추출 (숫자 확인 필요)
    upvote = 0
    upvote_elem = soup.select_one("div.font-semibold.text-orange-500")
    if upvote_elem:
        upvote_text = upvote_elem.get_text(strip=True)
        # 숫자만 추출 (예: "123" 또는 "123 upvotes")
        upvote_match = re.search(r"\d+", upvote_text)
        if upvote_match:
            upvote = int(upvote_match.group())

    return {
        "context": abstract,
        "authors": authors,
        "github_url": github_url,
        "upvote": upvote,
    }


def save_paper_json(paper_data: Dict, year: int, week: int, index: int) -> str:
    """
    논문 데이터를 JSON 파일로 저장

    Args:
        paper_data: 논문 데이터 딕셔너리
            - title (str): 논문 제목
            - context (str): 논문 초록
            - authors (List[str]): 저자 List
            - github_url (str): GitHub URL
            - huggingface_url (str): HuggingFace 논문 URL
            - upvote (int): 추천 수
            - tags (List[str]): tag List
        year: 연도 (예: 2025)
        week: 주차 (1~52)
        index: 논문 번호 (0부터 시작)

    Returns:
        str: 저장된 파일 ID (예: doc2545001)

    File Structure:
        01_data/documents/{year}/{year}-W{week}/doc{YY}{ww}{NNN}.json

    JSON Format:
        {
          "context": "Abstract 텍스트...",
          "metadata": {
            "title": "논문 제목",
            "authors": ["저자1", "저자2", ...],
            "publiction_year": year (int)
            "github_url": "GitHub URL",
            "huggingface_url": "HuggingFace URL",
            "upvote": 123,
            "tags": ["keyword1", "keyword2", "keyword3"]
          }
        }
    """
    week_str = f"{year}-W{week:02d}"

    # 파일명 생성: doc{YY}{ww}{NNN}.json
    doc_id = f"doc{year % 100:02d}{week:02d}{index+1:03d}"
    filename = f"{doc_id}.json"

    # 디렉토리 생성
    save_dir = f"01_data/documents/{year}/{week_str}"
    os.makedirs(save_dir, exist_ok=True)

    # JSON 데이터 구조
    document = {
        "context": paper_data["context"],
        "metadata": {
            "title": paper_data["title"],
            "authors": paper_data["authors"],
            "publication_year": year,
            "github_url": paper_data["github_url"],
            "huggingface_url": paper_data["huggingface_url"],
            "upvote": paper_data["upvote"],
            "tags": paper_data["tags"],
        },
    }

    # JSON 파일 저장
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    return doc_id


# 메인 함수
def crawl_weekly_papers(year: int, week: int):
    """
    특정 주차의 HuggingFace DailyPapers 크롤링

    Args:
        year: 연도
        week: 주차
    """
    week_str = f"{year}-W{week:02d}"
    print(f"\n{'='*60}")
    print(f"[START] {week_str} 크롤링 시작")
    print(f"{'='*60}")

    # 1. Weekly 페이지에서 논문 목록 추출
    papers = fetch_weekly_papers(year, week)

    if not papers:
        print(f"[WARNING]  {week_str}에 논문이 없습니다.")
        return

    # 2. 각 논문 처리
    success_count = 0
    fail_count = 0

    for index, paper_info in enumerate(tqdm(papers, desc="[CRAWLING] 논문 처리")):
        paper_url = paper_info["url"]
        paper_title = paper_info["title"]

        # 논문 상세 정보 추출
        details = fetch_paper_details(paper_url)
        time.sleep(2)  # 서버 부하 방지

        if not details["context"]:
            fail_count += 1
            continue

        # KeyBERT로 키워드 추출
        keywords = extract_keywords(details["context"], top_n=3)

        # 데이터 저장
        paper_data = {
            "title": paper_title,
            "context": details["context"],
            "authors": details["authors"],
            "github_url": details["github_url"],
            "huggingface_url": paper_url,
            "upvote": details["upvote"],
            "tags": keywords,
        }

        try:
            doc_id = save_paper_json(paper_data, year, week, index)
            success_count += 1
        except Exception as e:
            print(f"\n[FATAL] {doc_id} 저장 실패: {e}")
            fail_count += 1

        # Error 429 방지 - 40개마다 휴식
        if (index + 1) % 40 == 0 and index + 1 < len(papers):
            print(f"\n[BREAK] {index+1}개 처리 완료, 160초 휴식...")
            time.sleep(160)

    # 3. 통계 출력
    print(f"\n{'='*60}")
    print(f"[END] {week_str} 크롤링 완료")
    print(f"   총 논문: {len(papers)}개")
    print(f"   성공: {success_count}개")
    print(f"   실패: {fail_count}개")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=== HuggingFace DailyPapers 크롤러 ===\n")

    # 크롤링 실행 예시 (2025년 45~49주차)
    for week in range(45, 50):
        try:
            crawl_weekly_papers(year=2025, week=week)
        except Exception as e:
            print(f"\n[FATAL] W{week:02d} 크롤링 실패: {e}")

    print("\n[COMPLETE] 모든 크롤링 완료!")
