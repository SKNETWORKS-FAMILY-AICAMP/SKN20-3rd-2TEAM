"""
팀 공식 크롤러: HuggingFace DailyPapers Weekly (Resume 가능 안정판)
- 전공자 구조(스키마/폴더/파일명) 100% 호환
- 우리가 만든 안정성(User-Agent + delay + 429 backoff) 통합
- TF-IDF 기반 keyword 추출 포함
- 중간 중단 후 재실행하면 기존 JSON 자동 스킵 (resume 기능)
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# 0) 로깅 설정
# ============================================================
def setup_logging(week_str: str) -> None:
    log_dir = "././01_data/logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/crawling_{week_str}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info(f"[START] 크롤링 시작: {week_str}")


# ============================================================
# 1) 안정적인 HTTP GET with retry & 429 방어
# ============================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

def get_with_retry(
    url: str,
    max_retries: int = 5,
    base_delay: float = 10.0,
    backoff: float = 2.0
):
    """안정적인 GET 요청: delay + user-agent + 429 backoff"""
    delay = base_delay

    for attempt in range(1, max_retries + 1):
        logging.info(f"[INFO] request {attempt}/{max_retries} to {url} (sleep {delay}s)")
        time.sleep(delay)

        resp = requests.get(url, headers=HEADERS)

        if resp.status_code == 429:
            logging.warning("[WARN] 429 Too Many Requests: delay ↑")
            delay *= backoff
            continue

        try:
            resp.raise_for_status()
            return resp
        except Exception as e:
            logging.error(f"[ERROR] HTTP {resp.status_code}: {e}")
            if attempt == max_retries:
                return None

    return None


# ============================================================
# 2) week URL 생성
# ============================================================
def get_week_url(year: int, week: int) -> tuple[str, str]:
    week_str = f"{year}-W{week:02d}"
    url = f"https://huggingface.co/papers/week/{week_str}"
    return week_str, url


# ============================================================
# 3) Weekly 페이지 → 논문 URL 수집
# ============================================================
def fetch_paper_urls(weekly_url: str) -> List[Dict[str, str]]:
    logging.info(f"[STATUS] 논문 목록 추출: {weekly_url}")

    resp = get_with_retry(weekly_url)
    if resp is None:
        logging.error("[FATAL] weekly 페이지 로딩 실패")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    papers = []
    for link in soup.select("a.line-clamp-3"):
        href = link.get("href")
        title = link.get_text(strip=True)

        if href and title:
            papers.append({
                "title": title,
                "url": f"https://huggingface.co{href}"
            })

    logging.info(f"[COUNT] 총 {len(papers)}개 논문 발견")
    return papers


# ============================================================
# 4) 상세 페이지 (abstract, github, upvote)
# ============================================================
def fetch_paper_details(url: str, title: str) -> Optional[Dict]:
    resp = get_with_retry(url)
    if resp is None:
        logging.warning(f"[SKIP] 상세페이지 불가: {title}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # abstract
    abs_tag = soup.select_one("p.text-gray-600")
    abstract = abs_tag.get_text(" ", strip=True) if abs_tag else ""

    if not abstract or len(abstract) < 30:
        logging.warning(f"[WARNING] abstract 없음/짧음: {title}")
        return None

    # github
    gh_tag = soup.find("a", href=lambda x: x and "github.com" in x)
    github_url = gh_tag["href"] if gh_tag else ""

    # upvote — 실제 HF 구조 맞춤형 selector
    upvote_div = soup.find(
        "div",
        class_=lambda c: c and "shadow-alternate" in c and "cursor-pointer" in c
    )
    upvote = 0
    if upvote_div:
        import re
        m = re.search(r"(\d+)", upvote_div.get_text(" ", strip=True))
        if m:
            upvote = int(m.group(1))

    return {
        "title": title,
        "abstract": abstract,
        "github_url": github_url,
        "huggingface_url": url,
        "upvote": upvote
    }


# ============================================================
# 5) 키워드 추출 (TF-IDF, 전공자 방식)
# ============================================================
def extract_keywords(text: str, top_n: int = 3) -> List[str]:
    if len(text.split()) < 10:
        return ["keyword1", "keyword2", "keyword3"][:top_n]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=100,
            ngram_range=(1, 2)
        )
        tfidf = vectorizer.fit_transform([text]).toarray()[0]
        features = vectorizer.get_feature_names_out()

        idxs = np.argsort(tfidf)[-top_n:][::-1]
        return [features[i] for i in idxs]

    except Exception:
        return ["keyword1", "keyword2", "keyword3"][:top_n]


# ============================================================
# 6) Resume 기능 — JSON 이미 있으면 스킵
# ============================================================
def already_saved(doc_id: str, week_str: str) -> bool:
    year = week_str[:4]
    save_dir = f"././01_data/documents/{year}/{week_str}"
    path = os.path.join(save_dir, f"{doc_id}.json")
    return os.path.exists(path)


# ============================================================
# 7) JSON 저장 (전공자 스키마 그대로)
# ============================================================
def save_document_json(data: Dict, week_str: str, idx: int):
    year = week_str[:4]
    week = week_str.split("-W")[1]

    doc_id = f"doc{year[2:]}{week}{idx+1:03d}"
    filename = f"{doc_id}.json"

    # resume: 이미 존재하면 스킵
    if already_saved(doc_id, week_str):
        logging.info(f"[SKIP] 이미 존재함: {filename}")
        return doc_id, False

    save_dir = f"././01_data/documents/{year}/{week_str}"
    os.makedirs(save_dir, exist_ok=True)

    document = {
        "context": data["abstract"],
        "metadata": {
            "paper_name": data["title"],
            "github_url": data["github_url"],
            "huggingface_url": data["huggingface_url"],
            "upvote": data["upvote"],
            "tags": data["tags"]
        }
    }

    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    logging.info(f"[SAVE] {filename}")
    return doc_id, True


# ============================================================
# 8) 전체 크롤링 실행
# ============================================================
def crawl_weekly_papers(year: int, week: int):
    week_str, weekly_url = get_week_url(year, week)
    setup_logging(week_str)

    start = time.time()

    paper_links = fetch_paper_urls(weekly_url)
    if not paper_links:
        logging.warning("[WARNING] 논문 없음")
        return

    saved = 0
    skipped = 0
    failed = 0

    records = []

    for idx, p in enumerate(paper_links):
        logging.info(f"[PROCESS] {idx+1}/{len(paper_links)}: {p['title']}")

        details = fetch_paper_details(p["url"], p["title"])
        if details is None:
            failed += 1
            continue

        tags = extract_keywords(details["abstract"])
        details["tags"] = tags

        doc_id, is_new = save_document_json(details, week_str, idx)

        if is_new:
            saved += 1
        else:
            skipped += 1

        records.append({
            "doc_id": doc_id,
            "context": details["abstract"],
            "metadata": {
                "paper_name": details["title"],
                "github_url": details["github_url"],
                "huggingface_url": details["huggingface_url"],
                "upvote": details["upvote"],
                "tags": tags
            }
        })

        time.sleep(1)

    # CSV 저장
    year = week_str[:4]
    csv_path = f"././01_data/documents/{year}/{week_str}/docs_info.csv"
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start

    logging.info("====================================================")
    logging.info(f"[DONE] 총 {len(paper_links)}개")
    logging.info(f"[STATS] 성공 {saved}, 스킵 {skipped}, 실패 {failed}")
    logging.info(f"[TIME] {elapsed:.1f}s")
    logging.info("====================================================")


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    crawl_weekly_papers(2025, 48)
