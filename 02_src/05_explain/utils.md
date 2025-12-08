# Utils 모듈 문서

## 개요

본 모듈은 문서 처리, 임베딩 모델 평가, 벡터 데이터베이스 관리를 위한 유틸리티 함수들을 제공합니다.

---

## 1. chunking.py

### 목적
JSON 문서 파일을 로딩하고 작은 청크로 분할하여 저장/로드하는 기능 제공

### 주요 경로 설정
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "01_data" / "documents"
CHUNKS_DIR = PROJECT_ROOT / "01_data" / "chunks"
```

### 주요 함수

#### 1.1 load_json_files(use_weeks: int = 6)
최근 N주차의 JSON 문서를 로딩합니다.

**매개변수:**
- `use_weeks`: 로딩할 최근 주차 수 (기본값: 6)

**반환값:**
- `List[Document]`: LangChain Document 리스트

**동작 방식:**
1. `01_data/documents/` 폴더 스캔
2. 최신 주차부터 use_weeks 만큼 JSON 파일 로딩
3. JSON의 context를 page_content로, metadata는 그대로 사용
4. upvote 필드 정규화 (문자열 → 숫자)

---

#### 1.2 chunk_documents(documents, chunk_size=100, chunk_overlap=10)
문서 리스트를 작은 청크로 분할합니다.

**매개변수:**
- `documents`: 원본 문서 리스트
- `chunk_size`: 청크 크기 (글자 수, 기본값: 100)
- `chunk_overlap`: 청크 간 중복 부분 (글자 수, 기본값: 10)

**반환값:**
- `List[Document]`: 청크로 분할된 문서 리스트

**동작 방식:**
1. RecursiveCharacterTextSplitter로 문서 분할
2. 각 청크에 `chunk_index`, `total_chunks` 정보 추가
3. 원본 문서의 메타데이터 유지

**분리 기준:**
```python
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
```

---

#### 1.3 save_chunks_to_pkl(chunks, chunk_size, chunk_overlap)
청크 리스트를 `.pkl` 파일로 저장합니다.

**매개변수:**
- `chunks`: 저장할 청크 리스트
- `chunk_size`: 청크 크기 (파일명에 사용)
- `chunk_overlap`: 청크 오버랩 (파일명에 사용)

**반환값:**
- `str`: 저장된 파일 경로

**파일명 형식:**
```
chunks_{chunk_size}_{chunk_overlap}.pkl
```

---

#### 1.4 chunk_and_save(use_weeks=6, chunk_size=100, chunk_overlap=10)
문서 로딩 → 청킹 → 저장을 한번에 실행하는 통합 함수입니다.

**사용 예시:**
```python
# 기본 설정
chunk_and_save()

# 커스텀 파라미터
chunk_and_save(use_weeks=4, chunk_size=200, chunk_overlap=20)
```

---

#### 1.5 load_chunks_from_pkl(chunk_size=100, chunk_overlap=10)
저장된 `.pkl` 파일에서 청크 리스트를 로딩합니다.

**사용 예시:**
```python
chunks = load_chunks_from_pkl(chunk_size=100, chunk_overlap=10)
print(f"로딩된 청크 개수: {len(chunks)}")
```

---

## 2. evaluate_embeddings.py

### 목적
8개의 임베딩 모델을 비교 평가하여 최적의 모델을 선택합니다.

### 평가 대상 모델

| 모델명 | 차원 | 특징 |
|--------|------|------|
| MiniLM-L6 | 384 | 빠른 속도 |
| MPNet | 768 | 높은 품질 |
| MsMarco | 384 | 검색 최적화 |
| SPECTER | 768 | 과학 논문 특화 |
| OpenAI-small | 1536 | OpenAI API |
| BGE-M3 | 1024 | 중국어/영어 지원 |
| Jina-v2 | - | Jina AI |
| Paraphrase-Multi | 768 | 다국어 지원 |

### 평가 메트릭

#### 2.1 Recall@k
상위 k개 검색 결과에서 관련 문서가 얼마나 포함되었는지 측정

```python
calculate_recall_at_k(relevant_indices, retrieved_indices, k)
```

#### 2.2 MRR (Mean Reciprocal Rank)
관련 문서가 검색 결과에서 얼마나 상위에 위치하는지 측정

```python
calculate_mrr(relevant_indices, retrieved_indices)
```

### 테스트 쿼리

총 10개의 테스트 쿼리를 사용하여 평가:
- Video understanding
- Multimodal benchmarks
- Code generation
- Diffusion models
- Reinforcement learning
- Mathematical reasoning
- Robotic manipulation
- Image editing
- Attention mechanisms
- Spatial reasoning

### 주요 함수

#### 2.1 init_models()
8개 임베딩 모델을 초기화합니다.

**반환값:**
- `Dict[str, any]`: 모델명 → 임베딩 모델 객체

**특징:**
- CUDA 사용 가능 시 GPU 활용
- normalize_embeddings=True 설정

---

#### 2.2 find_relevant_docs(chunks, test_query)
테스트 쿼리에 대한 관련 문서 인덱스를 찾습니다.

**동작 방식:**
- metadata의 tags와 쿼리의 relevant_keywords 매칭
- 부분 문자열 포함 여부로 판단

---

#### 2.3 evaluate_model(model_name, embedding_model, chunks, test_queries)
단일 모델을 평가합니다.

**반환값:**
- `EvaluationResult`: 평가 결과 (R@5, R@10, MRR, 평균 시간)

**평가 과정:**
1. 모든 문서 임베딩 생성 (배치 크기: 100)
2. 각 테스트 쿼리에 대해 검색 수행
3. Recall@5, Recall@10, MRR 계산
4. 평균 성능 지표 산출

---

#### 2.4 main()
전체 평가 프로세스를 실행합니다.

**실행 단계:**
1. `01_data/chunks/chunks_100_10.pkl` 로드
2. 샘플링 (2000개, 평가 시간 단축용)
3. 8개 모델 평가
4. Recall@10 기준 정렬 및 결과 출력

**출력 형식:**
```
Model                    R@5       R@10        MRR  Avg Time(s)
-----------------------------------------------------------------
[모델별 성능 지표 테이블]
```

---

## 3. vectordb.py

### 목적
문서 청크를 임베딩하여 Chroma 벡터 데이터베이스에 저장하고 로드하는 기능 제공

### 주요 경로 설정
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "01_data" / "chunks"
VECTORDB_DIR = PROJECT_ROOT / "01_data" / "vecter_db"
```

### 지원 임베딩 모델

```python
embedding_models = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "MsMarco": "sentence-transformers/msmarco-MiniLM-L-6-v3",
    "SPECTER": "sentence-transformers/allenai-specter",
    "OpenAI": "text-embedding-3-small",
    "BGE-M3": "BAAI/bge-m3",
    "Paraphrase-Multi": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
```

### 주요 함수

#### 3.1 vectordb_save(model_name, chunk_size=100, chunk_overlap=10)
문서 청크를 임베딩하여 Chroma 벡터 데이터베이스에 저장합니다.

**매개변수:**
- `model_name`: 사용할 임베딩 모델 (embedding_models의 키)
- `chunk_size`: 청크 크기 (기본값: 100)
- `chunk_overlap`: 청크 간 겹침 (기본값: 10)

**동작 방식:**
1. `load_chunks_from_pkl()`로 청크 로드
2. 메타데이터 전처리 (List → 쉼표 구분 문자열)
   - `authors`: ["A", "B"] → "A, B"
   - `tags`: ["tag1", "tag2"] → "tag1, tag2"
3. 임베딩 모델 초기화 (GPU 사용 가능 시 CUDA)
4. Chroma 벡터스토어 생성 및 저장

**컬렉션명 형식:**
```
chroma_{model_name}_{chunk_size}_{chunk_overlap}
```

**예외 처리:**
- OpenAI 모델: OpenAIEmbeddings 사용
- HuggingFace 모델: HuggingFaceEmbeddings 사용
- 모델 로딩 실패 시 오류 메시지 출력

---

#### 3.2 load_vectordb(model_name, chunk_size=100, chunk_overlap=10)
저장된 Chroma 벡터 데이터베이스를 로드합니다.

**반환값:**
- `Chroma`: 로드된 Chroma 벡터스토어 객체

**동작 방식:**
1. VECTORDB_DIR 존재 확인
2. Chroma 벡터스토어 로드
3. 처음 5개 문서와 메타데이터 출력 (샘플 확인용)

**사용 예시:**
```python
# OpenAI 임베딩 사용
vectordb_save("OpenAI", 100, 10)
vectorstore = load_vectordb("OpenAI", 100, 10)

# HuggingFace 모델 사용
vectordb_save("MiniLM-L6", 100, 10)
vectorstore = load_vectordb("MiniLM-L6", 100, 10)
```

---

## 전체 워크플로우

### 1단계: 문서 청킹
```python
from chunking import chunk_and_save

# JSON 문서 로딩 → 청킹 → pickle 저장
chunk_and_save(use_weeks=6, chunk_size=100, chunk_overlap=10)
```

### 2단계: 임베딩 모델 평가 (선택사항)
```python
from evaluate_embeddings import main

# 8개 모델 비교 평가
main()
```

### 3단계: 벡터 DB 생성
```python
from vectordb import vectordb_save, load_vectordb

# 최적 모델로 벡터 DB 저장
vectordb_save("OpenAI", 100, 10)

# 벡터 DB 로드
vectorstore = load_vectordb("OpenAI", 100, 10)
```

---

## 의존성

### 필수 라이브러리
- `langchain-core`: Document 클래스
- `langchain-text-splitters`: RecursiveCharacterTextSplitter
- `langchain-openai`: OpenAIEmbeddings
- `langchain-huggingface`: HuggingFaceEmbeddings
- `langchain-chroma`: Chroma 벡터스토어
- `torch`: GPU 가속
- `scikit-learn`: cosine_similarity
- `numpy`: 수치 연산
- `transformers`: AutoModel (Jina 모델용)

### 환경 변수
```env
OPENAI_API_KEY=your_openai_api_key
```

---

## 파일 구조

```
01_data/
├── documents/           # 원본 JSON 문서
│   └── {year}/
│       └── {week}/
│           └── *.json
├── chunks/              # 청크 pickle 파일
│   └── chunks_{size}_{overlap}.pkl
└── vecter_db/          # Chroma 벡터 DB
    └── chroma_{model}_{size}_{overlap}/
```

---

## 주요 특징

### 청킹 (chunking.py)
- 최근 N주차 데이터만 선택적 로딩
- RecursiveCharacterTextSplitter 사용
- 청크 인덱스 및 총 청크 수 메타데이터 자동 추가
- pickle 직렬화로 빠른 로드

### 평가 (evaluate_embeddings.py)
- 8개 임베딩 모델 동시 비교
- Recall@k, MRR 메트릭 제공
- GPU 자동 감지 및 활용
- 배치 처리로 메모리 효율성 확보

### 벡터 DB (vectordb.py)
- 다양한 임베딩 모델 지원
- 메타데이터 자동 전처리
- Chroma 영구 저장
- 모델별 독립 컬렉션 관리
