# 🤗 HuggingFace WeeklyPapers 기반 ML/DL/LLM 스택 RAG 챗봇 
## 👥 팀원
| <img src="" width="150"> <br> 김지은 |  <img src="" width="150"> <br> 박다정 |  <img src="" width="150"> <br> 오학성 |  <img src="" width="150"> <br> 정소영 |  <img src="" width="200"> <br> 황수현 |
|:------:|:------:|:------:|:------:|:------:|

## 1. 프로젝트 소개 및 목표

### 1.1 프로젝트 소개

이 프로젝트는 **HuggingFace Weekly Papers** 데이터를 자동으로 수집·분석하여,  
최신 **ML/DL/LLM 논문 트렌드**를 질의응답 형태로 제공하는 **RAG 기반 논문 검색 챗봇**입니다.
  - **핵심 기능:**
  - HuggingFace Weekly Papers를 정기적으로 **크롤링**
  - 논문 Abstract를 **청킹 + 임베딩 + 벡터DB 저장**
  - **K-Means 클러스터링 + 클러스터 메타데이터**로 트렌드/토픽 구조화
  - **LangGraph** 기반 RAG 라우팅으로 검색 전략 자동 선택
  - **HTML/JS UI + FastAPI 백엔드**를 통한 웹 챗봇 제공

### 1.2 목표

- RAG 파이프라인 전체(크롤링 → 전처리 → 벡터DB → RAG → UI)를 **직접 구현하고 이해하는 것**
- **클러스터링 + 하이브리드 검색(BM25 + Vector + RRF)** 으로 검색 품질 향상
- **RAG(Retrieval-Augmented Generation)** 구조와 LLM 기반 트렌드 분석 기능을 갖춘 AI 논문 검색 플랫폼을 구현하는 것을 목표

## 2. 프로젝트 디렉토리 구조(수정해야하므ㅡㅡㅡㅡㅡㅡㅡㅡ)

```bash
SKN20-3rd-2TEAM/
├── 01_data/                        # 데이터 및 벡터 저장소
│   ├── chunks/                     # 청킹된 문서(.pkl 등)
│   ├── documents_T/                # TF-IDF 기반으로 추출된 논문 JSON
│   ├── documents_K/                # KeyBERT 기반으로 추출된 논문 JSON
│   └── vector_db/                  # Chroma VectorDB 파일
│
├── 02_src/                         # 모든 소스 코드
│   ├── 01_data_collection/         # 데이터 수집(크롤링) 단계
│   │   └── crawling.py             # HuggingFace DailyPapers 크롤러
│   │
│   ├── 02_utils/                   # 공통 유틸 및 전처리/인덱싱 도구
│   │   ├── chunking.py             # JSON → LangChain Document → 청킹 → .pkl 저장
│   │   ├── evaluate_embeddings.py  # 임베딩 모델별 성능/품질 비교 실험 스크립트
│   │   └── vectordb.py             # 청크 → 임베딩 → Chroma VectorDB 생성/로드
│   │
│   ├── 03_rag/                     # RAG 코어 로직
│   │   └── simpleRAGsystem_2.py    # Retriever + LLM 조합 RAG 시스템 (ask_with_sources 등)
│   │
│   ├── 04_ui/                      # 프론트엔드 / 서비스 레이어
│   │   ├── app.py                  # Streamlit 메인 앱 엔트리 포인트
│   │   └── components.py           # UI 컴포넌트, 세션 상태, VectorDB/RAG 로딩, 키워드 UI 등
│   │
│   └── 05_explain/                 # 실험/구현 설명 문서
│       ├── crawling.md             # 크롤링 설계, 파이프라인, 예외 처리 설명
│       ├── evaluate_embeddings_results.md  # 임베딩 비교 실험 결과 및 분석
│       └── utils.md                # chunking/vectordb 등 유틸 모듈 설명 및 사용법
│
├── MDimages/                       # README 및 문서에 사용하는 이미지/다이어그램
│   # 예: 아키텍처 다이어그램, UI 스크린샷, 결과 그래프 등
│
├── .env                            # 환경 변수 설정 파일 (API 키, CHUNK_SIZE 등)
├── README.md                       # 프로젝트 설명 문서
└── requirements.txt                # Python 라이브러리 설치 목록
```

## 3. 시스템 아키텍처(다시ㅡㅡㅡㅡㅡㅡㅡㅡ)
<img src="MDimages/SystemArchitecture.png" width="85%"> 


## 📊 데이터 파이프라인(폴더 이름 수정 해야함 ㅡㅡㅡㅡㅡㅡㅡㅡㅡ)

1. **크롤링** (`crawling.py`)
   - HuggingFace Weekly Papers에서 논문 메타데이터 수집
   - Abstract, Authors, Upvote, GitHub/HF URL 등을 JSON으로 저장

2. **청킹** (`chunking.py`)
   - 수집된 논문을 검색 최적화 크기로 분할
   - RecursiveCharacterTextSplitter 활용

3. **클러스터링 + 임베딩** (`vectordb.py`)
   - K-Means 기반 논문 클러스터링 및 키워드 추출
   - OpenAI/HuggingFace 임베딩 모델로 벡터화
   - Chroma VectorDB 생성 및 저장

4. **RAG 시스템** (`langgraph_test.py`)
   - LangGraph 기반 Multi-Agent RAG 구축
   - Hybrid Retrieval (Vector + BM25) 적용

5. **백엔드 + HTML UI** (`app_v3_main.py` + `app_v3.html`)
   - FastAPI 백엔드로 RAG 시스템 서빙
   - HTML/JS 프론트엔드로 대화형 인터페이스 제공

## 4. 기술 스택

### 🧩 Backend

| 구분 | 기술 | 설명 |
|------|-------|-------|
| **언어** | 🐍 Python 3.10 | 전체 파이프라인 구현 핵심 언어 |
| **웹 프레임워크** | ⚡ FastAPI | 고성능 비동기 기반 API 서버 |
| **LLM 엔진** | 🤖 OpenAI GPT-4o-mini | 최종 답변 생성 모델 |
| **RAG Framework** | 🔗 LangChain / 🔀 LangGraph | 하이브리드 검색 + 조건부 라우팅 RAG |
| **Vector DB** | 🗂️ ChromaDB | 문서 임베딩 저장·검색 |
| **검색 엔진** | 🔍 BM25 + 🔎 Vector Search + 📈 RRF | 점수 융합 기반 상위 문서 검색 |
| **임베딩 모델** | 🧬 OpenAI / 🤗 HuggingFace Embeddings | 범용·논문 특화·다국어 임베딩 |
| **클러스터링** | 📊 Scikit-learn KMeans | 논문 주제 클러스터링 |
| **전처리** | ✂️ NLTK · TF-IDF · KeyBERT · Lemmatization | 불용어 제거, 표제화, 키워드 추출 |

---

### 🎨 Frontend

| 구분 | 기술 | 설명 |
|------|-------|-------|
| **UI 구조** | 🌐 HTML5 | 단일 페이지 챗봇 인터페이스 |
| **스타일링** | 🎨 CSS (Custom) | 다크/라이트 테마 & 반응형 디자인 |

---

### 📊 Data & Storage

| 구분 | 기술 | 핵심 개념 |
|------|-------|------------|
| **데이터 소스** | 📚 HuggingFace Weekly Papers | 매주 공개되는 최신 AI 논문 데이터를 수집하는 출처 |
| **원본 저장** | 📝 JSON | 크롤링한 논문의 제목, 초록, 메타데이터 등등 그대로 담아 기본 저장 |
| **청킹 결과** | 📁 PKL | 긴 텍스트를 검색하기 좋은 작은 조각들로 나눠 보관하는 중간 데이터 |
| **클러스터 정보** | 🧩 JSON | 비슷한 내용의 논문들을 주제별로 묶고, 대표 키워드 등 특징을 정리한 정보 |
| **벡터 저장소** | 🗄️ ChromaDB | 청킹된 문서를 임베딩 벡터 형태로 저장해 빠르게 검색할 수 있게 만드는 공간 |

## 5. 모듈별 상세 설명

### 📥 5.1 `crawling.py` — HuggingFace 논문 크롤링

**역할**  
- Weekly Papers 페이지에서 최신 논문 데이터 자동 수집  
- 논문 본문(`context`) + 메타데이터(`metadata`) 구조로 JSON 저장  
- TF-IDF / KeyBERT 기반 키워드 추가(옵션)  
- 이후 청킹·임베딩·클러스터링의 원본 데이터가 됨

**수집 JSON 구조**

| 필드 | 설명 | 예시 |
|------|------|------|
| **context** | 논문 Abstract + 커뮤니티 텍스트 | "SAM 3 introduces a novel concept..." |
| **metadata.title** | 논문 제목 | "SAM 3: Segment Anything with Concepts" |
| **metadata.authors** | 저자 목록 | ["Nicolas Carion", "Laura Gustafson", ...] |
| **metadata.publication_year** | 발행 연도 | 2025 |
| **metadata.github_url** | GitHub 링크 | "https://github.com/facebookresearch/sam3" |
| **metadata.huggingface_url** | HuggingFace 논문 링크 | "https://huggingface.co/papers/2511.16719" |
| **metadata.upvote** | 추천 수 | 108 |
---

### ✂️ 5.2 `chunking.py` — 텍스트 청킹

**역할**  
- 논문 JSON → 검색 최적화 청크로 분할  
- 각 청크에 `doc_id`, `chunk_index`, `total_chunks` 메타데이터 추가  
- `.pkl` 파일로 저장하여 VectorDB 구축에 사용

**환경 변수 (.env)**

```env
CHUNK_SIZE=200
CHUNK_OVERLAP=30
```

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| **CHUNK_SIZE** | 청크당 최대 문자 수 | 200 |
| **CHUNK_OVERLAP** | 청크 간 중복 문자 수 | 30 |
---

### 🧩 5.3 `clustering.py` — 자동 논문 클러스터링
**역할:** 논문(문서) 전체를 K-Means로 주제 그룹으로 묶고, 클러스터 메타데이터 생성

**주요 기능**
- 문서 임베딩 생성(OpenAI Embeddings + 캐시 관리)
- 최적 클러스터 수 자동 탐색(Elbow 기반)
- K-Means 클러스터링 수행
- 클러스터별 대표 정보 생성  
  - 대표 키워드(TF-IDF + Lemmatization)  
  - 중심 문서(top papers)  
  - 크기(size), 밀도(density), 평균 upvote  
- 결과 저장  
  - `cluster_assignments.json` (doc_id → cluster_id)  
  - `cluster_metadata.json` (클러스터별 특징)

**특징**
- vectordb.py에서 자동 호출되어 VectorDB 생성 과정에 포함됨
---

### 🗄️ 5.4 `vectordb.py` — 임베딩 + VectorDB 생성/로드

**위치:** `02_src/02_utils/vectordb.py`  
**역할:** 청킹된 문서를 임베딩하고 Chroma VectorDB로 저장하거나 로드

**내부 흐름 (`vectordb_save`)**
1. 청크 로드 (`load_chunks_from_pkl`)
2. 클러스터링 수행 (`cluster_documents`)
3. 각 청크에 `cluster_id` 메타데이터 부착
4. 임베딩 모델 선택(OpenAI 또는 HuggingFace)
5. Chroma VectorDB 생성 및 디스크 저장
---

### 🤖 5.5 `langgraph_test.py` — LangGraph 기반 RAG 검색 엔진

**위치:** `02_src/03_rag/langgraph_test.py`  
**역할:** LangGraph로 구성된 RAG 워크플로를 생성하고, 질문에 대한 최종 답변을 생성

**핵심 구성 요소**
- VectorDB + BM25 + RRF 하이브리드 검색
- 클러스터 기반 라우팅 (cluster_similarity_check_node)
- Tavily 기반 웹 검색 fallback
- 최종 답변 생성(한글 요약 구조화)

**주요 노드**
| 노드명 | 역할 |
|--------|-------|
| **translate_node** | 한글 입력 시 영어로 변환하여 검색 성능 향상 |
| **retrieve_node** | Vector + BM25 + RRF로 상위 문서 검색 |
| **evaluate_document_relevance_node** | 문서 관련성(high/medium/low) 분류 |
| **cluster_similarity_check_node** | medium일 경우 클러스터 내 문서 재검색 |
| **web_search_node** | 내부 문서 부족 시 웹 검색 수행 |
| **generate_final_answer_node** | 최종 한글 답변 생성(요약·인사이트·출처 포함) |
| **reject_node** | 비관련 질문 시 정중한 안내 |
---

### 💬 5.6 `app_v3_main.py` & `app_v3.html` — 웹 서비스 UI/백엔드

**UI 파일:** `02_src/04_ui/app_v3.html`  
**API 서버:** `02_src/04_ui/app_v3_main.py`

**백엔드 역할 (FastAPI)**
- 서버 실행 시 LangGraph RAG 시스템 초기화
- 제공 API  
  - `POST /api/chat` → 질문 → RAG 엔진 답변 반환  
  - `GET /api/stats` → 논문/청크/클러스터 통계  
  - `GET /api/trending-keywords` → 클러스터 기반 인기 키워드 제공  

**프론트 역할 (HTML/JS)**
- Fetch API로 FastAPI와 통신
- Chat UI 렌더링(메시지 버블, 출처 카드, 검색 타입 뱃지 등)
- 트렌드 키워드 버튼 클릭 시 자동 질문 전송
- 반응형 다크/라이트 테마 포함


## 6. 응답 전략 및 출력 형식

### 4.1. 시스템 프롬프트 (System Prompt)

```text
"당신은 HuggingFace DailyPapers Weekly 데이터를 기반으로 ML/DL/LLM 논문 트렌드를 검색하고 추천하는 전문 플랫폼입니다.

주요 규칙:
1. 모든 답변은 사용자가 질문한 내용과 검색된 논문 문서 (Context)만을 기반으로 합니다.
2. 검색된 문서에 정보가 없는 경우, '죄송합니다. 현재 데이터(HuggingFace Weekly Papers)에서 해당 논문/키워드 정보를 찾을 수 없습니다.'라고 명확하게 답변합니다.
3. 논문 검색 결과 제공 시, 반드시 아래의 **출력 형식**을 준수합니다.
4. 답변 시, 논문에 대한 설명은 간결하고 명확하게 제공하며, 핵심 키워드를 함께 언급합니다."
```
* *(구현 예정)*

### 4.2. 논문 검색 결과 출력 형식

```markdown
### 📚 트렌드 논문 검색 결과 (by HuggingFace WeeklyPapers)

* **논문 이름(Title):** [논문 이름]
    * **요약 안내:** [Abstract 기반의 간결한 핵심 요약]
    * **링크:** [[GitHub 주소 또는 HuggingFace 논문 링크](링크 URL)]

* **논문 이름(Title):** [두 번째 논문 이름]
    * **요약 안내:** [Abstract 기반의 간결한 핵심 요약]
    * **링크:** [[GitHub 주소 또는 HuggingFace 논문 링크](링크 URL)]
```
* *(구현 예정)*
-----

## . 향후 개발 계획 (TODO List)

  * **[P2]** HuggingFace 데이터 수집 로직을 주간 **Batch 스케줄링**으로 전환.
  * **[P2]** Streamlit UI 상단에 **상위 N개 트렌드 키워드**를 시각적으로 노출하고, **클릭 시 관련 논문 자동 조회** 기능 구현.
  * **[P3]** 사용자 질문에 따른 **Tool Calling (e.g., 특정 논문 상세 분석)** 기능 도입 검토.
  * *(구현 예정)*

-----

## 💬 한 줄 회고

> #### 김지은
> 
---

> #### 박다정
> 

---

> #### 오학성
> 

---

> #### 정소영
> 

---

> #### 황수현
> 

---
