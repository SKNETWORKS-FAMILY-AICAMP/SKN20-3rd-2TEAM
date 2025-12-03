"""
HuggingFace DailyPapers Weekly 크롤링 스크립트
- 주차별 논문 데이터를 JSON 형식으로 수집
- TF-IDF 기반 키워드 자동 추출
- 로깅 및 에러 처리 포함
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# 로깅 설정
def setup_logging(week_str: str) -> None :
    """크롤링 로그 설정"""
    log_dir = "././01_data/logs"
    os.makedirs(log_dir, exist_ok = True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/crawling_{week_str}_{timestamp}.log"

    logging.basicConfig(
        level = logging.INFO
        , format='%(asctime)s [%(levelname)s] %(message)s'
        , handlers = [
            logging.FileHandler(log_file, encoding = 'utf-8')
            , logging.StreamHandler()
        ]
    )

    logging.info(f"[START] 크롤링 시작: {week_str}")

# HTTP 요청 재시도 로직
def fetch_with_retry(url: str, max_retries: int = 3, backoff: int = 2) -> requests.Response :
    """재시도 로직이 포함된 HTTP 요청"""
    for attempt in range(max_retries) :
        try :
            response = requests.get(url, timeout = 10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e :
            if attempt < max_retries - 1 :
                wait_time = backoff ** attempt
                logging.warning(f"[WARNING] 요청 실패 ({attempt+1}/{max_retries}), {wait_time}초 후 재시도... URL: {url}")
                time.sleep(wait_time)
            else :
                logging.error(f"[FAILED] 최대 재시도 횟수 초과: {url}")
                raise Exception(f"최대 재시도 횟수 초과: {url}") from e

# URL 생성
def get_week_url(year: int = None, week: int = None) -> tuple[str, str] :
    """
    주차별 HuggingFace Weekly Papers URL 생성

    Args:
        year: 연도 (None이면 현재 연도)
        week: 주차 (None이면 현재 주차)

    Returns:
        (week_str, url) 튜플
    """
    if year is None or week is None :
        now = datetime.now()
        year = year or now.year
        week = week or now.isocalendar()[1]

    week_str = f"{year}-W{week:02d}"
    url = f"https://huggingface.co/papers/week/{week_str}"

    return week_str, url

# Step 1: 논문 목록 추출
def fetch_paper_urls(weekly_url: str) -> List[Dict[str, str]] :
    """
    Weekly 페이지에서 모든 논문 URL 추출

    Args:
        weekly_url: HuggingFace weekly papers URL

    Returns:
        [{"title": "...", "url": "..."}, ...] 리스트
    """
    logging.info(f"[STATUS] 논문 목록 추출 중: {weekly_url}")

    response = fetch_with_retry(weekly_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    paper_links = []
    for link in soup.select('a.line-clamp-3') :
        href = link.get('href')
        title = link.get_text(strip = True)

        if href and title:
            full_url = f"https://huggingface.co{href}"
            paper_links.append({ "title": title, "url": full_url })

    logging.info(f"[COUNT] 총 {len(paper_links)}개 논문 발견")
    return paper_links

# Step 2: 논문 상세 정보 추출
def fetch_paper_details(paper_url: str, paper_title: str) -> Optional[Dict] :
    """
    개별 논문 상세 페이지에서 정보 추출

    Args:
        paper_url: 논문 상세 페이지 URL
        paper_title: 논문 제목

    Returns:
        {"abstract": "...", "github_url": "...", "upvote": int} 또는 None
    """
    try :
        response = fetch_with_retry(paper_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Abstract 추출 (여러 <p> 태그를 하나로 결합)
        abstract = ""
        abstract_section = soup.select_one('section div')
        if abstract_section :
            paragraphs = abstract_section.find_all('p')
            abstract = ' '.join([ p.get_text(strip=True) for p in paragraphs ])

        if not abstract or len(abstract) < 50:
            logging.warning(f"[WARNING] Abstract가 너무 짧거나 없음: {paper_title}")
            return None

        # GitHub URL 추출 (선택적)
        github_link = soup.select_one('a[href*="github.com"]')
        github_url = github_link['href'] if github_link else ""

        # Upvote 추출
        # XPath: /html/body/div/main/div/section[1]/div/div[1]/div[3]/div/a/div/div
        # outerHTML: <div class="font-semibold text-orange-500">-</div>
        # upvote가 0이면 '-'로 표시됨
        upvote = 0

        # font-semibold 클래스를 가진 div 요소에서 upvote 추출
        upvote_elem = soup.select_one('div.font-semibold.text-orange-500')

        if upvote_elem :
            text = upvote_elem.get_text(strip = True)

            # '-'는 upvote가 0임을 의미
            if text == '-' :
                upvote = 0
            else :
                # 숫자만 추출
                digits = ''.join(filter(str.isdigit, text))
                if digits :
                    upvote = int(digits)

        return {
            "abstract": abstract
            , "github_url": github_url
            , "huggingface_url": paper_url
            , "upvote": upvote
            , "title": paper_title
        }

    except Exception as e :
        logging.error(f"[FAILED] 논문 상세 정보 추출 실패: {paper_title} - {e}")
        return None

# Step 3: 키워드 추출
def extract_top_keywords(text: str, top_n: int = 3) -> List[str] :
    """
    TF-IDF 기반 상위 N개 키워드 추출

    Args:
        text: 추출 대상 텍스트 (Abstract)
        top_n: 추출할 키워드 개수

    Returns:
        키워드 리스트
    """
    try :
        # 텍스트가 너무 짧으면 기본값 반환
        if len(text.split()) < 10 :
            return ["keyword1", "keyword2", "keyword3"][:top_n]

        vectorizer = TfidfVectorizer(
            max_features = 100
            , stop_words = 'english'
            , ngram_range = (1, 2)  # 1-gram과 2-gram 모두 고려
        )
        tfidf_matrix = vectorizer.fit_transform([text])

        # 상위 N개 단어 추출
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = np.argsort(scores)[-top_n:][::-1]

        keywords = [ feature_names[i] for i in top_indices ]

        # 키워드가 부족하면 기본값으로 채우기
        while len(keywords) < top_n :
            keywords.append(f"keyword{len(keywords)+1}")

        return keywords[:top_n]

    except Exception as e :
        logging.warning(f"[WARNING] 키워드 추출 실패, 기본값 사용: {e}")
        return ["keyword1", "keyword2", "keyword3"][:top_n]

# Step 4: JSON 파일 저장
def save_document_json(paper_data: Dict, week_str: str, index: int) -> tuple[str, str] :
    """
    논문 데이터를 JSON 파일로 저장

    Args:
        paper_data: 논문 데이터 딕셔너리
        week_str: 주차 문자열 (예: "2025-W49")
        index: 논문 인덱스 (0부터 시작)

    Returns:
        (doc_id, doc_filename) 튜플
    """
    # 파일명 생성: doc{YY}{ww}{NNN}.json
    year = week_str[:4]
    week = week_str.split('-W')[1]
    doc_id = f"doc{year[2:]}{week}{index+1:03d}"
    doc_filename = f"{doc_id}.json"

    # 디렉토리 생성 (연도별 폴더 구조)
    save_dir = f"././01_data/documents/{year}/{week_str}"
    os.makedirs(save_dir, exist_ok = True)

    # JSON 구조 생성 (CLAUDE.md의 간소화된 스키마 사용)
    document = {
        "context": paper_data['abstract']
        , "metadata": {
            "paper_name": paper_data['title']
            , "github_url": paper_data.get('github_url', "")
            , "huggingface_url": paper_data['huggingface_url']
            , "upvote": paper_data['upvote']
            , "tags": paper_data['tags']  # 리스트 형태로 저장
        }
    }

    # JSON 파일 저장
    file_path = os.path.join(save_dir, doc_filename)
    with open(file_path, 'w', encoding = 'utf-8') as f :
        json.dump(document, f, ensure_ascii = False, indent = 2)

    logging.info(f"[SUCCESS] JSON 저장: {doc_filename}")
    return doc_id, doc_filename

# Step 5: CSV 메타데이터 저장
def save_metadata_csv(papers_data: List[Dict], week_str: str) -> str :
    """
    수집된 논문 메타데이터를 CSV 인덱스로 저장

    Args:
        papers_data: 논문 데이터 리스트
        week_str: 주차 문자열

    Returns:
        CSV 파일 경로
    """
    csv_records = []

    for paper in papers_data :
        csv_records.append({
            'doc_id': paper['doc_id']
            , 'paper_name': paper['metadata']['paper_name']
            , 'doc_file': f"{paper['doc_id']}.json"
            , 'github_url': paper['metadata']['github_url']
            , 'huggingface_url': paper['metadata']['huggingface_url']
            , 'upvote': paper['metadata']['upvote']
            , 'tags': json.dumps(paper['metadata']['tags'], ensure_ascii = False)
        })

    df = pd.DataFrame(csv_records)

    # 연도 추출
    year = week_str[:4]
    csv_path = f"././01_data/documents/{year}/{week_str}/docs_info.csv"
    df.to_csv(csv_path, index = False, encoding = 'utf-8-sig')

    logging.info(f"[SUCCESS] CSV 저장 완료: {csv_path} ({len(csv_records)}개 논문)")
    return csv_path

# 메인 크롤링 함수
def crawl_weekly_papers(year: int = None, week: int = None) -> None :
    """
    주차별 HuggingFace Papers 크롤링 메인 함수

    Args:
        year: 연도 (None이면 현재 연도)
        week: 주차 (None이면 현재 주차)
    """
    # 1. URL 생성 및 로깅 설정
    week_str, weekly_url = get_week_url(year, week)
    setup_logging(week_str)

    logging.info(f"[INFO] 대상 주차: {week_str}")
    logging.info(f"[INFO] URL: {weekly_url}")

    start_time = time.time()

    try :
        # 2. 논문 목록 추출
        paper_links = fetch_paper_urls(weekly_url)

        if not paper_links :
            logging.warning("[WARNING] 논문 목록이 비어있습니다.")
            return

        # 3. 각 논문 상세 정보 수집 및 저장
        saved_papers = []
        failed_count = 0

        for idx, paper_link in enumerate(paper_links) :
            logging.info(f"\n[STATUS] [{idx+1}/{len(paper_links)}] 처리 중: {paper_link['title'][:50]}...")

            # 상세 정보 추출
            details = fetch_paper_details(paper_link['url'], paper_link['title'])

            if not details :
                failed_count += 1
                continue

            # 키워드 추출
            keywords = extract_top_keywords(details['abstract'], top_n = 3)
            details['tags'] = keywords

            # JSON 파일 저장
            doc_id, doc_filename = save_document_json(details, week_str, idx)

            # CSV용 데이터 저장
            saved_papers.append({
                'doc_id': doc_id
                , 'context': details['abstract']
                , 'metadata': {
                    'paper_name': details['title']
                    , 'github_url': details['github_url']
                    , 'huggingface_url': details['huggingface_url']
                    , 'upvote': details['upvote']
                    , 'tags': keywords
                }
            })

            # 요청 간 딜레이 (Rate Limiting 방지)
            time.sleep(1)

        # 4. CSV 메타데이터 저장
        if saved_papers :
            save_metadata_csv(saved_papers, week_str)

        # 5. 통계 출력
        elapsed_time = time.time() - start_time
        logging.info("\n" + "=" * 60)
        logging.info(f"[SUCCESS] 크롤링 완료!")
        logging.info(f"[STATS] 통계:")
        logging.info(f"   - 총 논문 수: {len(paper_links)}")
        logging.info(f"   - 성공: {len(saved_papers)}")
        logging.info(f"   - 실패: {failed_count}")
        logging.info(f"   - 소요 시간: {elapsed_time:.2f}초")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"[FATAL] 크롤링 중 치명적 오류 발생: {e}", exc_info = True)
        raise

# 실행
if __name__ == "__main__" :
    # 2025년 49주차 크롤링
    # year, week를 지정하거나 None으로 현재 주차 사용
    crawl_weekly_papers(year=2025, week=49)

    # 현재 주차 크롤링
    # crawl_weekly_papers()
