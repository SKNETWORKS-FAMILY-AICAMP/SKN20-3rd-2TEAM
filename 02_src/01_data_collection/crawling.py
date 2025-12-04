"""
HuggingFace DailyPapers 크롤링 및 키워드 추출 모듈

주요 기능:
1. HuggingFace Weekly Papers 크롤링
2. 논문 Abstract에서 자동 키워드 추출 (Lemmatization → TF-IDF)
3. JSON 형식으로 데이터 저장

키워드 추출 알고리즘:
- Step 1: 토큰화 (3글자 이상 영문)
- Step 2: WordNet Lemmatization (단어 기본형 변환)
- Step 3: NLTK + 커스텀 불용어 제거
- Step 4: TF-IDF 벡터화 (unigram + bigram)
- Step 5: 중복 키워드 필터링 (긴 키워드에 포함된 짧은 키워드 제거)
- Step 6: 상위 3개 키워드 반환

Version: 2.0
Author: SKN20-3rd-2TEAM
"""

import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from datetime import datetime
import re
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

def get_with_retry(
    url: str,
    max_retries: int = 6,      
    initial_wait: float = 2.0, # 첫 실패 이후 대기
    backoff: float = 6.0,
    timeout: float = 12.0,
):
    wait = initial_wait

    for attempt in range(1, max_retries + 1):
        logging.info(f"[HTTP] GET {url} (try {attempt}/{max_retries})")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
        except Exception as e:
            logging.warning(f"[HTTP] 예외, retry... {e}")
            time.sleep(wait)
            wait *= backoff
            continue

        # 429: 너무 자주 요청했다는 뜻 → 짧게 몇 번만 재시도
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            wait_time = float(ra) if ra else wait
            logging.warning(f"[429] Too Many Requests, sleep {wait_time:.1f}s 후 재시도")
            time.sleep(wait_time)
            wait *= backoff
            continue

        # 정상 응답
        if 200 <= resp.status_code < 300:
            return resp

        # 기타 에러 코드
        logging.warning(f"[HTTP {resp.status_code}] retry..")
        time.sleep(wait)
        wait *= backoff

    logging.error(f"[FAIL] 요청 실패(최대 재시도 초과): {url}")
    return None



# TF-IDF 및 NLTK
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK 불용어 다운로드 (최초 1회)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# 설정
TAG_COUNT = 3
CRAWLER_VERSION = "2.0"

def extract_keywords_tfidf_nltk(text: str, top_n: int = 3) -> List[str]:
    """
    Lemmatization → TF-IDF 기반 키워드 추출

    전처리 후 TF-IDF를 적용하여 더 정확한 키워드를 추출합니다.

    처리 순서:
        1. 토큰화: 3글자 이상 영문 단어만 추출
        2. Lemmatization: WordNetLemmatizer로 단어 기본형 변환
           - "training" → "train"
           - "models" → "model"
        3. 불용어 제거: NLTK stopwords + 논문 도메인 특화 불용어
        4. TF-IDF 벡터화: unigram(1-gram) + bigram(2-gram)
        5. 키워드 추출: TF-IDF 점수 상위 N개 선택
        6. 중복 필터링: 긴 키워드에 포함된 짧은 키워드 제거
           - ["attention", "self attention"] → ["self attention"]

    Args:
        text: 논문 Abstract 텍스트
        top_n: 추출할 키워드 개수 (기본값: 3)

    Returns:
        List[str]: TF-IDF 점수가 높은 상위 N개 키워드 (lemmatized & filtered)

    Examples:
        >>> abstract = "We trained multiple neural networks using transformers."
        >>> extract_keywords_tfidf_nltk(abstract, top_n=3)
        ['transformer', 'neural network', 'train']

    Note:
        - 10단어 미만의 짧은 텍스트는 기본 키워드 반환
        - 전처리 후 텍스트가 비어있으면 기본 키워드 반환
    """
    # 1. 텍스트 검증
    if not text or len(text.split()) < 10:
        logging.warning("[WARNING] Abstract가 너무 짧음 (< 10 words), 기본 키워드 반환")
        return [f"keyword{i+1}" for i in range(top_n)]

    try:
        lemmatizer = WordNetLemmatizer()

        # 2. NLTK 영어 불용어 로드
        nltk_stopwords = set(stopwords.words('english'))

        # 3. 도메인 특화 불용어 추가 (선택적)
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

        # 4. 텍스트 전처리: Lemmatization + NLTK 불용어 제거
        # 4-1. 토큰화 (3글자 이상 영문만)
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # 4-2. Lemmatization 적용
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # 4-3. 불용어 제거
        filtered_tokens = [token for token in lemmatized_tokens if token not in all_stopwords]

        # 4-4. 전처리된 텍스트 재구성
        preprocessed_text = ' '.join(filtered_tokens)

        if not preprocessed_text.strip():
            logging.warning("[WARNING] 전처리 후 텍스트가 비어있음, 기본 키워드 반환")
            return [f"keyword{i+1}" for i in range(top_n)]

        logging.debug(f"[PREPROCESSED] Original length: {len(text.split())} → Filtered: {len(filtered_tokens)}")

        # 5. 전처리된 텍스트로 TF-IDF Vectorizer 생성
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),              # 1~2그램 (unigram + bigram)
            max_features=1000,               # 어휘 크기 제한
            lowercase=False,                 # 이미 소문자 처리됨
            token_pattern=r'\b[a-z]{3,}\b'   # 3글자 이상 영문만
        )

        # 6. TF-IDF 행렬 생성
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # 7. TF-IDF 점수로 정렬 (내림차순)
        idx_score_pairs = [(i, score) for i, score in enumerate(scores) if score > 0]

        if not idx_score_pairs:
            logging.warning("[WARNING] TF-IDF 점수가 모두 0, 기본 키워드 반환")
            return [f"keyword{i+1}" for i in range(top_n)]

        idx_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in idx_score_pairs[:top_n]]

        # 8. 상위 N개 키워드 추출 (이미 lemmatization이 적용된 상태)
        keywords = [feature_names[i] for i in top_indices]

        logging.info(f"[KEYWORDS_RAW] {keywords} (lemmatized)")

        # 8-1. 중복 제거: 더 긴 키워드에 포함되는 짧은 키워드 제거
        # 예: ["attention", "self attention"] → ["self attention"] (attention 제거)
        filtered_keywords = []
        keywords_sorted = sorted(keywords, key=len, reverse=True)  # 긴 것부터 정렬

        for keyword in keywords_sorted:
            # 현재 키워드가 이미 선택된 더 긴 키워드에 포함되는지 확인
            is_substring = False
            for selected in filtered_keywords:
                # 단어 경계 체크: "attention"이 "self attention"에 단어로 포함되는지
                if keyword in selected.split():
                    is_substring = True
                    logging.debug(f"[REMOVED] '{keyword}' (포함됨: '{selected}')")
                    break

            if not is_substring:
                filtered_keywords.append(keyword)

        # 원래 순서(TF-IDF 점수 순) 유지
        keywords = [kw for kw in keywords if kw in filtered_keywords]

        logging.info(f"[KEYWORDS_FILTERED] {keywords} (중복 제거 완료)")

        # 9. 부족한 키워드는 기본값으로 채우기
        while len(keywords) < top_n:
            keywords.append(f"keyword{len(keywords)+1}")

        return keywords[:top_n]

    except Exception as e:
        logging.error(f"[ERROR] TF-IDF 키워드 추출 실패: {e}")
        return [ f"keyword{i+1}" for i in range(top_n) ]


def fetch_weekly_papers(year: int, week: int) -> List[Dict[str, str]]:
    """
    HuggingFace DailyPapers Weekly 페이지에서 논문 목록 추출

    Args:
        year: 연도 (예: 2025)
        week: 주차 (1~52)

    Returns:
        List[Dict]: 논문 URL과 제목 리스트
    """
    week_str = f"{year}-W{week:02d}"
    weekly_url = f"https://huggingface.co/papers/week/{week_str}"

    logging.info(f"[FETCH] Weekly 페이지 요청: {weekly_url}")

    try:
        response = get_with_retry(weekly_url, timeout=10)
        if response is None:
            logging.error(f"[ERROR] Weekly 페이지 요청 최대 재시도 초과: {weekly_url}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')

        # 논문 링크 추출 (CSS Selector: a.line-clamp-3)
        paper_links = []
        for link in soup.select('a.line-clamp-3'):
            href = link.get('href')
            title = link.get_text(strip=True)

            if href:
                full_url = f"https://huggingface.co{href}"
                paper_links.append({
                    'title': title
                    , 'url': full_url
                })

        logging.info(f"[SUCCESS] 논문 {len(paper_links)}개 발견")
        return paper_links

    except requests.exceptions.RequestException as e:
        logging.error(f"[ERROR] Weekly 페이지 요청 실패: {e}")
        return []


def fetch_paper_details(paper_url: str) -> Dict[str, any]:
    """
    개별 논문 상세 페이지에서 Abstract, GitHub URL, Upvote 추출

    Args:
        paper_url: 논문 상세 페이지 URL (예: https://huggingface.co/papers/2511.18538)

    Returns:
        Dict: 논문 상세 정보
            - abstract (str): 논문 초록 (여러 <p> 태그 결합)
            - github_url (str): GitHub 레포지토리 URL (없으면 빈 문자열)
            - upvote (int): 추천 수 (추출 실패 시 0)

    Note:
        - 네트워크 오류 시 빈 딕셔너리 반환 (모든 값 빈 문자열/0)
        - Upvote CSS Selector는 HuggingFace 페이지 구조 변경 시 조정 필요
    """
    try:
        response = get_with_retry(paper_url, timeout=10)
        if response is None:
            logging.error(f"[ERROR] 논문 상세 페이지 요청 실패(최대 재시도 초과): {paper_url}")
            return {"abstract": "", "github_url": "", "upvote": 0}


        soup = BeautifulSoup(response.content, 'html.parser')

        # Abstract 추출 (여러 <p> 태그를 결합)
        abstract_section = soup.select_one('section div')
        abstract = ""
        if abstract_section:
            paragraphs = abstract_section.find_all('p')
            abstract = ' '.join([ p.get_text(strip=True) for p in paragraphs ])

        # GitHub URL 추출 (선택적)
        github_link = soup.select_one('a[href*="github.com"]')
        github_url = github_link['href'] if github_link else ""

        # Upvote 추출 (CSS Selector는 실제 페이지 구조에 따라 조정 필요)
        upvote = 0
        upvote_elem = soup.select_one('div.font-semibold.text-orange-500')  # 실제 클래스명 확인 필요
        if upvote_elem:
            upvote_text = upvote_elem.get_text(strip=True)
            # 숫자만 추출 (예: "123" 또는 "123 upvotes")
            upvote_match = re.search(r'\d+', upvote_text)
            if upvote_match:
                upvote = int(upvote_match.group())

        return {
            'abstract': abstract
            , 'github_url': github_url
            , 'upvote': upvote
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"[ERROR] 논문 상세 페이지 요청 실패: {paper_url} - {e}")
        return {
            'abstract': ""
            , 'github_url': ""
            ,'upvote': 0
        }


def save_paper_json(paper_data: Dict, year: int, week: int, index: int) -> str:
    """
    논문 데이터를 JSON 파일로 저장

    Args:
        paper_data: 논문 데이터 딕셔너리
            - title (str): 논문 제목
            - abstract (str): 논문 초록
            - github_url (str): GitHub URL
            - huggingface_url (str): HuggingFace 논문 URL
            - upvote (int): 추천 수
            - tags (List[str]): 키워드 리스트 (3개)
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
            "paper_name": "논문 제목",
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
        "context": paper_data['abstract']
        , "metadata": {
            "paper_name": paper_data['title']
            , "github_url": paper_data['github_url']
            , "huggingface_url": paper_data['huggingface_url']
            , "upvote": paper_data['upvote']
            , "tags": paper_data['tags']
        }
    }

    # JSON 파일 저장
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    logging.info(f"[SAVED] {filename}")
    return doc_id


def crawl_weekly_papers(year: int, week: int):
    """
    특정 주차의 HuggingFace DailyPapers 크롤링 메인 함수

    전체 크롤링 파이프라인:
        1. Weekly 페이지에서 논문 목록 추출
        2. 각 논문 상세 페이지에서 Abstract, GitHub URL, Upvote 수집
        3. Abstract에서 Lemmatization → TF-IDF 기반 키워드 추출
        4. JSON 파일로 저장 (01_data/documents/{year}/{year}-W{week}/)

    Args:
        year: 연도 (예: 2025)
        week: 주차 (1~52)

    Returns:
        None (성공/실패 통계는 로그로 출력)

    Logs:
        - [START]: 크롤링 시작
        - [FETCH]: Weekly 페이지 요청
        - [KEYWORDS]: 추출된 키워드
        - [SAVED]: JSON 파일 저장 완료
        - [DONE]: 크롤링 완료 통계 (총 논문 수, 성공, 실패)

    Note:
        - Abstract가 없는 논문은 자동 스킵
        - 저장 실패 시 에러 로그 기록 후 다음 논문으로 진행
    """
    week_str = f"{year}-W{week:02d}"
    logging.info(f"\n{'='*60}")
    logging.info(f"[START] {week_str} 크롤링 시작")
    logging.info(f"{'='*60}")

    # 1. Weekly 페이지에서 논문 목록 추출
    papers = fetch_weekly_papers(year, week)

    if not papers:
        logging.warning(f"[WARNING] {week_str}에 논문이 없습니다.")
        return

    # 2. 각 논문 처리
    success_count = 0
    fail_count = 0

    for index, paper_info in enumerate(papers):

        # 🔥 여기에 있어야 함! (바깥 루프 안 / 내부 for문 없음)
        if index > 0 and index % 40 == 0:
            logging.info(f"[COOLDOWN] {index}개 처리 완료 → 160초 휴식")
            time.sleep(160)

        paper_url = paper_info['url']
        paper_title = paper_info['title']

        logging.info(f"\n[{index+1}/{len(papers)}] 처리 중: {paper_title}")

        # 2-1. 논문 상세 정보 추출
        details = fetch_paper_details(paper_url)
        time.sleep(2.0)

        if not details['abstract']:
            logging.warning(f"  [SKIP] Abstract 없음: {paper_title}")
            fail_count += 1
            continue

        # 2-2. TF-IDF + NLTK로 키워드 추출
        keywords = extract_keywords_tfidf_nltk(details['abstract'], top_n=TAG_COUNT)
        logging.info(f"  [KEYWORDS] {keywords}")

        # 2-3. 데이터 저장
        paper_data = {
            'title': paper_title
            , 'abstract': details['abstract']
            , 'github_url': details['github_url']
            , 'huggingface_url': paper_url
            , 'upvote': details['upvote']
            ,'tags': keywords
        }

        try:
            doc_id = save_paper_json(paper_data, year, week, index)
            success_count += 1
            logging.info(f"  [SUCCESS] {doc_id} 저장 완료")
        except Exception as e:
            logging.error(f"  [ERROR] 저장 실패: {e}")
            fail_count += 1

    # 3. 통계 출력
    logging.info(f"\n{'='*60}")
    logging.info(f"[DONE] {week_str} 크롤링 완료")
    logging.info(f"  - 총 논문 수: {len(papers)}")
    logging.info(f"  - 성공: {success_count}")
    logging.info(f"  - 실패: {fail_count}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    """메인 실행 함수"""
    # 로깅 설정
    log_dir = "././01_data/logs"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            # 로그 파일 이름은 datetime 사용
            logging.FileHandler(f"{log_dir}/crawling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # 크롤링 실행 (예시: 2025년 48주차)
    try:
        crawl_weekly_papers(year=2025, week=45)
    except Exception as e:
        logging.error(f"[ERROR] W{45:02d} 크롤링 실패: {e}")

    # 최신 데이터 크롤링 실행
    # 배치 돌 때 사용
    # try:
    #     current_date = datetime.now()
    #     current_year, current_week, _ = current_date.isocalendar()
    #     crawl_weekly_papers(year=current_year, week=current_week)
    # except Exception as e:
    #     logging.error(f"[ERROR] 최신 데이터 크롤링 실패: {e}")