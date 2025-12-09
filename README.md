# 🤗 HuggingFace WeeklyPapers 기반 ML/DL/LLM 스택 RAG 챗봇 
## 👥 팀원
| <img src="" width="150"> <br> 김지은 |  <img src="" width="150"> <br> 박다정 |  <img src="" width="150"> <br> 오학성 |  <img src="" width="150"> <br> 정소영 |  <img src="" width="200"> <br> 황수현 |
|:------:|:------:|:------:|:------:|:------:|

## 1. 프로젝트 개요 및 목표

### 1.1 개요

  * **프로젝트명:** HuggingFace WeeklyPapers 기반 최신 ML/DL/LLM 논문 트렌드 RAG 챗봇
  * **목적:** **RAG (Retrieval-Augmented Generation)** 패턴을 숙달하고, HuggingFace WeeklyPapers 데이터를 자동 수집하여,
  RAG(Retrieval-Augmented Generation) 구조와 LLM 기반 트렌드 분석 기능을 갖춘 AI 논문 검색 플랫폼을 구현하는 것을 목표로 한다.
  * **핵심 기능:**
      * 최신 ML/DL/LLM 논문 자동 수집 및 저장
      * Abstract 기반 자동 키워드 추출
      * VectorDB 기반 논문 검색
      * LangGraph 기반 RAG Workflow
      * Streamlit 대화형 챗봇 UI
      * 키워드 기반 논문 검색 및 요약 제공

### 1.2 전체 프로젝트 구조
### 프로젝트 디렉토리 구조

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
----- 
### 시스템 아키텍처
<img src="MDimages/SystemArchitecture.png" width="85%"> 

----- 
### 데이터 파이프라인

1. **크롤링 및 JSON 저장** (`crawling.py`)
   - HuggingFace WeeklyPapers에서 논문 메타데이터 수집
   - 수집된 논문 정보를 JSON 형식으로 저장

2. **청킹** (`chunking.py`)
   - 논문 텍스트를 검색에 적합한 크기로 분할

3. **임베딩 및 VectorDB 저장** (`vectordb.py`)
   - 청크를 벡터로 변환하여 Chroma VectorDB에 저장

4. **RAG 시스템** (`simpleRAGsystem_2.py`)
   - LangChain 기반 Retriever와 RAG 파이프라인 구성

5. **UI** (`app.py`)
   - Streamlit을 통한 대화형 인터페이스 제공

-----

## 2. 기술 스택

### 🔧 Backend

| 카테고리 | 기술 | 설명 |
|---------|------|------|
| **언어** | Python 3.10.11 | 핵심 개발 언어 |
| **LLM 프레임워크** | LangChain / LangGraph | RAG 파이프라인 구축 |
| **Vector DB** | ChromaDB | 임베딩 벡터 저장 및 검색 |
| **크롤링** | BeautifulSoup4 | 웹 스크래핑 |
| **불용어 제거 후 키워드 추출** | NLTK + KeyBERT / TF-IDF | 논문 불용어 제거 후 키워드 자동 추출 |
| **LLM API** | OpenAI ChatCompletion | 자연어 생성 및 응답 |

### 🎨 Frontend

| 기술 | 설명 |
|------|------|
| **Streamlit** | 대화형 웹 인터페이스 |
| **Custom CSS** | HuggingFace 스타일 UI 디자인 |

### 📊 Data

- **데이터 소스**: HuggingFace WeeklyPapers
- **수집 방식**: Weekly Abstract 자동 크롤링
- **저장 형식**: JSON (데이터 정제 후)
-----

## 3. 주요 기능

### 3.1 📥 데이터 수집 (크롤링)

**파일**: `crawling.py`

HuggingFace WeeklyPapers에서 주간 논문 데이터를 자동으로 수집합니다.

#### 수집 데이터

| 필드 | 설명 | 예시 |
|------|------|------|
| **Title** | 논문 제목 | Souper-Model: How Simple Arithmetic... |
| **Context** | 논문 초록 및 커뮤니티 댓글 | Large Language Models (LLMs) have... |
| **Authors** | 저자 목록 | Shalini Maiti, Amar Budhiraja, ... |
| **Publication Year** | 발행 연도 | 2025 |
| **GitHub URL** | 관련 코드 저장소 | https://github.com/facebookresearch/... |
| **HuggingFace URL** | 논문 페이지 링크 | https://huggingface.co/papers/2511.13254 |
| **Upvote** | 인기도 지표 | 134 |
| **Tags** | KeyBERT 추출 키워드 (Top 3) | averaging approaches, soup category, ... |

#### 주요 기능
- HuggingFace WeeklyPapers URL 자동 크롤링
- 논문 메타데이터 및 초록, 커뮤니티 반응 수집
- KeyBERT 기반 논문 키워드(Top 3) 자동 추출
- 체계적인 디렉토리 구조로 JSON 저장

#### 저장 형식
```
01_data/documents/{year}/{year}-W{week}/docYYWWNNN.json
```

**예시**:
```
01_data/documents/2025/2025-W11/doc2511001.json
01_data/documents/2025/2025-W11/doc2511002.json
```

#### JSON 구조
```json
{
  "context": "논문 초록 및 커뮤니티 댓글 전체 텍스트",
  "metadata": {
    "title": "논문 제목",
    "authors": ["저자1", "저자2", "..."],
    "publication_year": 2025,
    "github_url": "GitHub 저장소 URL",
    "huggingface_url": "HuggingFace 논문 페이지 URL",
    "upvote": 134,
    "tags": ["키워드1", "키워드2", "키워드3"]
  }
}
```
---

### 3.2 ✂️ 문서 청킹

**파일**: `chunking.py`

수집된 JSON 문서를 검색에 최적화된 크기로 분할합니다.

#### 처리 과정
1. JSON 문서를 LangChain Document로 로딩
2. `RecursiveCharacterTextSplitter` 기반 청킹
3. PKL 형식으로 저장 → VectorDB 인덱싱에 사용

#### 환경 변수 설정

`.env` 파일에서 청킹 파라미터를 설정합니다:
```env
CHUNK_SIZE=200
CHUNK_OVERLAP=20
```

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `CHUNK_SIZE` | 청크당 최대 문자 수 | 100 |
| `CHUNK_OVERLAP` | 청크 간 중복 문자 수 | 10 |

---

### 3.3 🗄️ VectorDB 저장 및 로딩

**파일**: `vectordb.py`

청킹된 문서를 벡터화하여 ChromaDB에 저장하고 로드합니다.

#### 지원 임베딩 모델

**HuggingFace 모델** (로컬 실행 가능)

| 모델명 | 특징 |
|--------|------|
| **all-MiniLM-L6-v** | 경량 고속 범용 모델 |
| **all-mpnet-base-v2** | 고품질 범용 임베딩 |
| **msmarco-MiniLM-L-6-v3** | 검색 최적화 |
| **allenai-specter** | 과학 논문 특화 |
| **bge-m3** | 다국어 지원 고성능 |
| **paraphrase-multilingual-mpnet-base-v2** | 다국어 의역 검색 |

**OpenAI 모델** (OPENAI API 키 필요)

| 모델명 | 특징 |
|--------|------|
| **text-embedding-3-small** | 고성능 임베딩 |

#### 주요 기능

- **벡터 DB 저장**: PKL 청크 → 임베딩 → ChromaDB 저장
- **벡터 DB 로드**: 저장된 컬렉션을 Retriever로 로드
- GPU 자동 감지 및 활용

#### 환경 변수
```env
MODEL_NAME=OpenAI
CHUNK_SIZE=100
CHUNK_OVERLAP=10
```

#### 사용 예시
```python
# VectorDB 생성
vectordb_save(model_name="OpenAI", chunk_size=100, chunk_overlap=10)

# VectorDB 로드
vectorstore = load_vectordb(model_name="OpenAI", chunk_size=100, chunk_overlap=10)
```
---

### 3.4 🤖 RAG 시스템

**파일**: `simpleRAGsystem_2.py`

*(구현 예정)*

---

### 3.5 💬 Streamlit UI

**파일**: `app.py`

*(구현 예정)*

-----

## 4. 응답 전략 및 출력 형식

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
