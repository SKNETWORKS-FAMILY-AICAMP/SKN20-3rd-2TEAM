# HuggingFace DailyPapers 크롤링 시스템

## 📋 개요

본 스크립트는 HuggingFace의 DailyPapers 웹사이트에서 AI/ML 관련 논문 정보를 자동으로 수집하는 크롤러입니다. 주간 단위로 게시된 논문들의 메타데이터와 초록을 수집하여 구조화된 JSON 파일로 저장합니다.

### 주요 기능
- HuggingFace Weekly Papers 페이지에서 논문 목록 자동 추출
- 각 논문의 상세 정보 크롤링 (제목, 저자, 초록, GitHub 링크, 추천 수)
- KeyBERT / TF-IDF를 활용한 자동 키워드 추출 비교
- 주차별/연도별 구조화된 JSON 파일 저장
- 재시도 로직 및 Rate Limiting 처리

---

## 🗂️ 데이터 구조

### 저장 디렉토리 구조
```
01_data/
└── documents/
    └── {year}/
        └── {year}-W{week}/
            ├── doc{YY}{ww}{001}.json
            ├── doc{YY}{ww}{002}.json
            └── ...
```

**예시**: `01_data/documents/2025/2025-W45/doc2545001.json`

### JSON 파일 포맷

**현재 버전 (KeyBERT)**
```json
{
  "context": "논문의 Abstract 전문...",
  "metadata": {
    "title": "논문 제목",
    "authors": ["저자1", "저자2", "저자3"],
    "publication_year": 2025,
    "github_url": "https://github.com/...",
    "huggingface_url": "https://huggingface.co/papers/...",
    "upvote": 123,
    "tags": ["keyword1", "keyword2", "keyword3"]
  }
}
```

**초기 버전 (TF-IDF)**
```json
{
  "context": "논문의 Abstract 전문...",
  "metadata": {
    "paper_name": "논문 제목",
    "github_url": "https://github.com/...",
    "huggingface_url": "https://huggingface.co/papers/...",
    "upvote": 123,
    "tag1": "keyword1",
    "tag2": "keyword2",
    "tag3": "keyword3"
  }
}
```

**주요 차이점**
- 현재 버전: `tags` 배열 사용, `authors` 및 `publication_year` 필드 추가
- 초기 버전: 개별 `tag1`, `tag2`, `tag3` 필드 사용, `paper_name` 필드명

---

## 🔧 주요 함수 설명

### 1. `extract_keywords(text: str, top_n: int = 3) -> List[str]`

**목적**: KeyBERT를 사용하여 논문 초록에서 핵심 키워드를 자동 추출

**매개변수**
- `text`: 논문 초록 텍스트
- `top_n`: 추출할 키워드 개수 (기본값: 3)

**반환값**: 추출된 키워드 리스트 (예: `["transformer", "attention mechanism", "nlp"]`)

**특징**
- KeyBERT의 MaxSum 알고리즘 사용
- 1~2단어 구문(n-gram) 추출
- 커스텀 불용어 필터링 적용
- 추출 실패 시 기본값 반환 (`keyword1`, `keyword2`, ...)

---

## 🏷️ 키워드 추출 방법론

본 프로젝트에서는 논문의 핵심 개념을 파악하기 위해 두 가지 키워드 추출 방식을 적용해 보았습니다.

### KeyBERT 기반 키워드 추출

**알고리즘**: KeyBERT의 BERT 임베딩 + MaxSum 다양성 알고리즘

**추출 과정**
1. BERT 모델로 문서와 후보 키워드의 의미적 유사도 계산
2. MaxSum 알고리즘으로 의미적으로 다양한 키워드 선택
3. 커스텀 불용어 필터링 적용

**장점**
- 문맥을 고려한 의미 기반 키워드 추출
- 사전 학습된 BERT 모델 활용으로 정확도 향상
- 간결한 코드로 고품질 키워드 확보

**예시 결과**
```python
Abstract: "We propose a novel transformer architecture using self-attention..."
Keywords: ["transformer architecture", "self attention", "neural network"]
```

---

### TF-IDF 기반 키워드 추출

**알고리즘**: Lemmatization → TF-IDF 벡터화 → 중복 필터링

**추출 과정**

1. 토큰화: 소문자 변환후 정규식으로 3글자 이상 영문 단어 추출
2. Lemmatization (형태소 정규화): WordNetLemmatizer 사용
   - 예: "training" → "train", "models" → "model"
3. 불용어 제거 (NLTK): 영어 stopwords와 논문 특화 불용어를 제거
4. TF-IDF 벡터화: n-gram 범위: (1, 2)
5. 점수 기반 정렬: TF-IDF 점수 상위 N개 선택
6. 중복 필터링: 긴 키워드에 포함된 짧은 키워드 제거 후 키워드를 3개로 맞춤
   - 예: ["attention", "self attention"] → ["self attention"]

**장점**
- 통계 기반 접근으로 해석 가능성 높음
- Lemmatization으로 단어 변형 문제 해결

**예시 결과**
```python
Abstract: "We trained multiple neural networks using transformers..."
Keywords: ["transformer", "neural network", "train"]
# "training" → "train" (lemmatized)
# "networks" → "network" (lemmatized)
```

**주요 차이점**
- **전처리**: 형태소 정규화로 단어 기본형 추출
- **필터링**: 2단계 중복 제거 (불용어 + substring 필터)

---

### 2. `get_with_retry(url: str, max_retries: int = 3)`

**목적**: 네트워크 오류 및 Rate Limiting을 처리하는 안정적인 HTTP 요청 함수

**매개변수**
- `url`: 요청할 URL
- `max_retries`: 최대 재시도 횟수 (기본값: 3)

**반환값**: `requests.Response` 객체 또는 `None` (실패 시)

**특징**
- HTTP 429 (Too Many Requests) 에러 대응
- 재시도 간 2초 대기 (기본), 429 에러 시 5초 대기
- 타임아웃 10초 설정
- User-Agent 헤더 자동 설정

---

### 3. `fetch_weekly_papers(year: int, week: int) -> List[Dict[str, str]]`

**목적**: HuggingFace Weekly Papers 페이지에서 해당 주차 논문 목록 추출

**매개변수**
- `year`: 연도 (예: 2025)
- `week`: 주차 (1~52)

**반환값**: 논문 정보 딕셔너리 리스트
```python
[
    {"title": "논문 제목", "url": "https://huggingface.co/papers/..."},
    ...
]
```

**크롤링 대상**
- URL 패턴: `https://huggingface.co/papers/week/{year}-W{week:02d}`
- CSS Selector: `a.line-clamp-3` (논문 제목 링크)

---

### 4. `fetch_paper_details(paper_url: str) -> Dict[str, any]`

**목적**: 개별 논문 상세 페이지에서 메타데이터 추출

**매개변수**
- `paper_url`: 논문 HuggingFace URL

**반환값**: 논문 상세 정보 딕셔너리
```python
{
    "context": "Abstract 전문",
    "authors": ["저자1", "저자2"],
    "github_url": "GitHub URL",
    "upvote": 123
}
```

**추출 데이터**
1. **Abstract**: `section div` 내 모든 `<p>` 태그 결합
2. **Authors**: CSS Selector로 저자 링크 추출
3. **GitHub URL**: `href*="github.com"` 속성 검색
4. **Upvote**: `div.font-semibold.text-orange-500` 내 숫자 파싱

---

### 5. `save_paper_json(paper_data: Dict, year: int, week: int, index: int) -> str`

**목적**: 크롤링한 논문 데이터를 구조화된 JSON 파일로 저장<br>
**JSON을 선택한 이유**: tag를 사용 할 것이기 때문에 구조화 된 데이터가 필요

**매개변수**
- `paper_data`: 논문 전체 데이터 (제목, 초록, 저자, URL 등)
- `year`: 연도
- `week`: 주차
- `index`: 논문 인덱스 (0부터 시작)

**반환값**: 저장된 문서 ID (예: `doc2545001`)

**파일명 규칙**
- 형식: `doc{YY}{ww}{NNN}.json`
- 예시: `doc2545001.json` = 2025년 45주차 1번째 논문

---

### 6. `crawl_weekly_papers(year: int, week: int)`

**목적**: 특정 주차의 전체 크롤링 프로세스 실행 (메인 함수)

**매개변수**
- `year`: 연도
- `week`: 주차

**실행 흐름**
```
1. Weekly 페이지에서 논문 목록 추출 (fetch_weekly_papers)
   ↓
2. 각 논문에 대해:
   a. 상세 정보 크롤링 (fetch_paper_details)
   b. KeyBERT로 키워드 추출 (extract_keywords)
   c. JSON 파일 저장 (save_paper_json)
   ↓
3. 성공/실패 통계 출력
```

**Rate Limiting 전략**
- 각 논문 처리 후 2초 대기
- 40개 논문마다 160초 휴식 (429 에러 방지)

---

## ⚙️ 설정 및 상수

### HTTP 헤더
```python
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"
}
```
- 브라우저 요청 위장으로 차단 방지

### 커스텀 불용어
```python
custom_stopwords = {
    "the", "a", "an", "and", "or", "but", ...,  # 일반 불용어
    "paper", "propose", "present", "show", ...  # 논문 특화 불용어
}
```
- 일반 영어 불용어 + 논문에서 자주 등장하는 일반적 용어 제거
- KeyBERT 키워드 추출 시 노이즈 감소

---

## 📦 의존성

### 필수 라이브러리 (KeyBERT 버전)
```python
import os, json, re, time, requests
from typing import List, Dict
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk
from keybert import KeyBERT
```

### 필수 라이브러리 (TF-IDF 버전)
```python
import os, json, re, time, requests
from typing import List, Dict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
```

### NLTK 데이터
- `corpora/stopwords`: 불용어 사전
- `corpora/wordnet`: 형태소 분석용 (Lemmatization)
- `corpora/omw-1.4`: Open Multilingual Wordnet (TF-IDF 버전)
- 초기 실행 시 자동 다운로드

---

## 🚀 실행 방법

### 기본 실행
```python
if __name__ == "__main__":
    # 2025년 47~49주차 크롤링
    for week in range(47, 50):
        try:
            crawl_weekly_papers(year=2025, week=week)
        except Exception as e:
            print(f"[FATAL] W{week:02d} 크롤링 실패: {e}")
```

### 커스터마이징
```python
# 단일 주차 크롤링
crawl_weekly_papers(year=2025, week=45)

# 여러 주차 크롤링
for week in range(45, 50):
    crawl_weekly_papers(year=2025, week=week)
```

---

## ⚠️ 주의사항

1. **Rate Limiting**
   - 40개 논문마다 160초 휴식 (하드코딩됨)
   - 너무 빠른 요청 시 429 에러 발생 가능

2. **네트워크 안정성**
   - 재시도 로직 있지만, 장기 실행 시 모니터링 필요
   - 실패한 논문은 로그에 기록되나 재시도하지 않음

3. **데이터 품질**
   - Abstract가 없는 논문은 자동 스킵
   - 키워드 추출 실패 시 기본값 사용 (`keyword1`, `keyword2`, ...)

---

## 🔍 문제 해결

### 자주 발생하는 오류

**1. 429 Too Many Requests**
- 원인: Rate Limit 초과
- 해결: `time.sleep()` 시간 증가 또는 휴식 빈도 조정

**2. 타임아웃 오류**
- 원인: 네트워크 불안정
- 해결: `timeout` 값 증가 (현재 10초)

**3. 키워드 추출 실패**
- 원인: Abstract가 너무 짧거나 특수문자만 포함
- 해결: 자동으로 기본값 사용 (정상 동작)

**4. 저장 경로 오류**
- 원인: 디렉토리 권한 문제
- 해결: `os.makedirs(exist_ok=True)` 확인 또는 수동 생성

---

## 📌 참고 사항

### 파일 위치
- **KeyBERT 버전**: `02_src/01_data_collection/crawling.py`
- **TF-IDF 버전**: `02_src/01_data_collection/past/crawling_past.py`

### 크롤링 정보
- 크롤링 대상: https://huggingface.co/papers/week/{year}-W{week}
- 로봇 배제 표준(robots.txt) 준수
- 학술 목적 데이터 수집용

### 기술 스택
- **키워드 추출**: KeyBERT / TF-IDF + Lemmatization
- **웹 스크래핑**: BeautifulSoup4 + requests
- **데이터 저장**: JSON 파일 시스템
- **자연어 처리**: NLTK, scikit-learn
