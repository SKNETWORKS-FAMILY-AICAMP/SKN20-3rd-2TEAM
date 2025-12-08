# 🚀 CLAUDE.md: RAG 기반 ML/DL/LLM 스택 챗봇 프로토타입 개발 계획 (V2: HuggingFace DailyPapers)

## 👥 팀원
| <img src="" width="150"> <br> 김지은 |  <img src="" width="150"> <br> 박다정 |  <img src="" width="150"> <br> 오학성 |  <img src="" width="150"> <br> 정소영 |  <img src="" width="200"> <br> 황수현 |
|:------:|:------:|:------:|:------:|:------:|

## 1\. 프로젝트 개요 및 목표

### 1.1. 개요

  * **프로젝트명:** HuggingFace DailyPapers 기반 최신 ML/DL/LLM 논문 트렌드 검색 플랫폼 프로토타입
  * **목적:** **RAG (Retrieval-Augmented Generation)** 패턴을 숙달하고, **HuggingFace DailyPapers Weekly 데이터**를 활용하여 LLM 기반의 실용적인 최신 기술 트렌드 검색 플랫폼을 구축한다.
  * **핵심 기능:**
      * Weekly 기반 트렌드 키워드 자동 추출 및 디스플레이.
      * 사용자 질문 또는 키워드 클릭에 따른 **VectorDB 기반 논문 검색 및 정보 제공**.

### 1.2. LLM 적용 방식

| 패턴 | 주요 라이브러리 | 설명 |
| :--- | :--- | :--- |
| **RAG** | `langchain`, `langchain_chroma` | HuggingFace DailyPapers 데이터를 외부 지식으로 활용하여 검색 정확도를 향상. |
| **Agent/Flow** | `langgraph` | 챗봇의 검색, 필터링, 응답 생성 등의 논리적 흐름을 정의. |

-----

## 2\. 기술 스택 및 환경 설정

### 2.1. 개발 환경

  * **Python Version:** $3.10.11$
  * **LLM Provider:** OpenAI (API Key 필수)
  * **UI/Front-end:** Streamlit

### 2.2. 핵심 라이브러리

| 분류 | 라이브러리/기술 | 비고 |
| :--- | :--- | :--- |
| **Orchestration** | `langchain`, `langgraph` | RAG 파이프라인 및 Agent Flow 정의 |
| **Data Acquisition** | `requests`, `BeautifulSoup` | **HuggingFace DailyPapers Weekly 페이지 파싱** |
| **Vector DB** | `langchain_chroma` | ChromaDB를 사용하여 벡터 임베딩 저장 및 검색 |
| **LLM Interface** | `langchain_openai` | LLM 및 Embeddings 모델 호출 |
| **UI** | `streamlit` | 챗봇 인터페이스 구축 |

-----

## 3\. 구현 상세 계획 (프로토타입 범위)

### 3.1. 데이터 수집 및 인덱싱 (Indexing)

1.  **데이터 소스:** HuggingFace DailyPapers Weekly 데이터.
2.  **수집 목표:** **논문 이름(Title), 요약(Abstract), GitHub 주소 또는 HuggingFace 논문 링크** 추출.
3.  **키워드 추출:** 수집된 논문 요약(Abstract) 기반으로 **가장 많이 언급된 상위 N개 키워드** 자동 추출 로직 구현.
4.  **Vector DB 저장:**
      * `OpenAIEmbeddings`를 사용하여 텍스트를 벡터화.
      * `Chroma` VectorStore에 저장 (논문 Title, Abstract 포함).

### 3.2. 챗봇/검색 흐름 정의 (LangGraph State & Flow)

  * **주요 기능 1: Weekly 기반 트렌드 키워드 디스플레이**
      * 추출된 상위 N개 키워드를 UI (Streamlit) 상단에 리스트/태그 형태로 표시.
  * **주요 기능 2: 트렌드 키워드 기반 문서 조회 (RAG Flow)**
      * **Trigger:** 트렌드 키워드 클릭 또는 사용자 질문 입력.
      * **Nodes:**
          * `fetch_retrieved_docs`: 사용자의 질문/키워드를 기반으로 Vector DB에서 관련 논문(`Document`) 검색.
          * `document_quality_grader`: 검색된 문서 목록의 **품질 필터링 (Grader)** 로직 구현 (선택적).
          * `generate_response`: 필터링된 논문 목록을 사용자에게 제시하는 응답 생성.

-----

## 4\. 응답 전략 및 출력 형식

### 4.1. 시스템 프롬프트 (System Prompt)

```text
"당신은 HuggingFace DailyPapers Weekly 데이터를 기반으로 ML/DL/LLM 논문 트렌드를 검색하고 추천하는 전문 플랫폼입니다.

주요 규칙:
1. 모든 답변은 사용자가 질문한 내용과 검색된 논문 문서 (Context)만을 기반으로 합니다.
2. 검색된 문서에 정보가 없는 경우, '죄송합니다. 현재 데이터(HuggingFace Weekly Papers)에서 해당 논문/키워드 정보를 찾을 수 없습니다.'라고 명확하게 답변합니다.
3. 논문 검색 결과 제공 시, 반드시 아래의 **출력 형식**을 준수합니다.
4. 답변 시, 논문에 대한 설명은 간결하고 명확하게 제공하며, 핵심 키워드를 함께 언급합니다."
```

### 4.2. 논문 검색 결과 출력 형식

```markdown
### 📚 트렌드 논문 검색 결과 (by HuggingFace DailyPapers)

* **논문 이름(Title):** [논문 이름]
    * **요약 안내:** [Abstract 기반의 간결한 핵심 요약]
    * **링크:** [[GitHub 주소 또는 HuggingFace 논문 링크](링크 URL)]

* **논문 이름(Title):** [두 번째 논문 이름]
    * **요약 안내:** [Abstract 기반의 간결한 핵심 요약]
    * **링크:** [[GitHub 주소 또는 HuggingFace 논문 링크](링크 URL)]
```

-----

## 5\. 향후 개발 계획 (TODO List)

  * **[P2]** HuggingFace 데이터 수집 로직을 주간 **Batch 스케줄링**으로 전환.
  * **[P2]** Streamlit UI 상단에 **상위 N개 트렌드 키워드**를 시각적으로 노출하고, **클릭 시 관련 논문 자동 조회** 기능 구현.
  * **[P3]** 사용자 질문에 따른 **Tool Calling (e.g., 특정 논문 상세 분석)** 기능 도입 검토.

-----

**이 파일 내용을 바탕으로 다음 단계인 데이터 수집 로직(섹션 3.1) 구현에 대해 궁금한 점이 있으신가요?**
