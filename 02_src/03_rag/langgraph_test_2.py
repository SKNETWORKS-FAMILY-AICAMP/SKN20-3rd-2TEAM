"""
LangGraph 기반 RAG 시스템

이 모듈은 LangGraph를 사용하여 다음과 같은 라우팅 로직을 구현합니다:
- retrieve → evaluate_document_relevance → {HIGH, MEDIUM, LOW}
- HIGH → generate_final_answer
- MEDIUM → cluster_similarity_check → {HIGH, LOW}
  - HIGH → generate_final_answer
  - LOW → web_search → generate_final_answer
- LOW → reject

VectorStore는 시작 시 로드되어 재사용됩니다.
"""

# ===== SECTION 1: IMPORTS =====
import os
import sys
import json
import re
import warnings
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain Community (BM25)
from langchain_community.retrievers import BM25Retriever

# LangGraph
from langgraph.graph import StateGraph, START, END

# ===== SECTION 2: ENVIRONMENT & PATHS =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "02_src" / "02_utils"))

warnings.filterwarnings("ignore")
load_dotenv()

# 환경 변수 로드
MODEL_NAME = os.getenv("MODEL_NAME", "MiniLM-L6")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 45))

# ===== SECTION 3: GRAPHSTATE =====
class GraphState(TypedDict):
    """
    LangGraph 상태 관리를 위한 TypedDict

    Attributes:
        question: 검색에 사용될 질문 (영어 번역된 질문 또는 원본)
        original_question: 사용자가 입력한 원본 질문
        translated_question: 영어로 번역된 질문 (한글인 경우)
        is_korean: 원본 질문이 한글인지 여부
        documents: 검색된 문서 리스트
        doc_scores: 각 문서의 유사도 점수 (L2 distance)
        cluster_id: 검색된 문서의 클러스터 ID
        cluster_similarity_score: 클러스터 내 평균 유사도 점수
        search_type: 검색 유형 ("internal", "cluster", "web", "rejected")
        relevance_level: 문서 관련성 수준 ("high", "medium", "low")
        answer: LLM이 생성한 최종 답변
        sources: 참조 문서 정보 리스트
        _vectorstore: VectorStore 객체 (내부 전용)
        _llm: LLM 객체 (내부 전용)
        _cluster_metadata_path: 클러스터 메타데이터 경로 (내부 전용)
    """
    question: str   # 검색에 사용될 질문 (영어 번역된 질문 또는 원본)
    original_question: str  # 사용자가 입력한 원본 질문
    translated_question: Optional[str]
    is_korean: bool
    documents: List[Document]
    doc_scores: List[float]
    cluster_id: Optional[int]
    cluster_similarity_score: Optional[float]
    search_type: str
    relevance_level: str
    answer: str
    sources: List[Dict[str, Any]]
    # Internal (injected at runtime)
    _vectorstore: Any
    _llm: Any
    _bm25_retriever: Any
    _cluster_metadata_path: str


# ===== SECTION 4: HELPER FUNCTIONS =====

def get_doc_hash_key(doc: Document) -> str:
    """
    문서의 고유 해시 키 생성 (RRF 중복 제거용)

    문서 내용(최대 1000자)과 출처를 결합하여 SHA256 해시를 생성합니다.
    이를 통해 BM25와 벡터 검색 결과에서 동일 문서를 식별할 수 있습니다.

    Args:
        doc: Document 객체

    Returns:
        str: SHA256 해시 문자열
    """
    content = doc.page_content[:1000]
    source = doc.metadata.get('source', '')
    data_to_hash = f"{content}|{source}"
    return hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()


def is_korean_text(text: str) -> bool:
    """
    텍스트에 한글이 포함되어 있는지 확인

    Args:
        text: 확인할 텍스트

    Returns:
        bool: 한글 포함 여부
    """
    # 한글 유니코드 범위: AC00-D7A3 (완성형), 1100-11FF (자모)
    korean_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
    return bool(korean_pattern.search(text))

def is_ai_ml_related_by_llm(question: str, llm) -> bool:
    """
    LLM을 사용해서 질문이 AI/ML 연구 논문 관련인지 판별한다.
    - True  : AI/ML 연구, 모델, 학습, 최적화, 응용 서비스 등과 관련
    - False : 해리포터 줄거리, 날씨, 일상, 영화/소설 감상 등 일반 질문

    LLM은 반드시 'YES' 또는 'NO'만 출력해야 한다.
    """
    if not question or llm is None:
        return False

    prompt = ChatPromptTemplate.from_messages([
(
    "system",
    """
You are a classifier that determines whether a user's question *could reasonably relate* to AI/ML/DL research topics.

Interpret broadly:  
If the question can *plausibly* be answered by AI/ML papers — even indirectly — return "YES".

Examples of YES:
- Topics about generation, animation, audio/visual models, identity preservation
- Conceptual questions that could relate to ML models
- Practical application questions (e.g., "How do I keep identity consistent when animating with audio?")
- General topics that ML papers often study (stability, alignment, optimization, synthesis)

Return "NO" **only** if the question has *no plausible connection* to ML research:
- Daily life, weather, food, personal advice
- Fictional stories, movies
- Purely philosophical or subjective opinions
- Generic coding help unrelated to ML models

Output must be exactly one word: YES or NO.
"""
),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip().upper()
        return result.startswith("Y")
    except Exception as e:
        print(f"[WARN] LLM topic classification failed: {e}")
        # 판별 실패하면 보수적으로 False 취급 (비 관련으로 간주)
        return False


# ===== SECTION 5: NODE FUNCTIONS =====

def translate_node(state: GraphState) -> dict:
    """
    노드 0: 한글 질문 번역

    질문이 한글인 경우 영어로 번역하여 검색 성능을 향상시킵니다.
    영어 논문 검색에 최적화되도록 학술적 표현으로 번역합니다.

    Args:
        state: 현재 GraphState

    Returns:
        dict: question, original_question, translated_question, is_korean을 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: translate] 질문 언어 확인 및 번역")
    print("="*60)

    original_question = state["original_question"]
    llm = state.get("_llm")

    # 한글 포함 여부 확인
    has_korean = is_korean_text(original_question)

    if not has_korean:
        print(f"[translate] 영어 질문 감지 - 번역 스킵")
        print(f"[translate] 원본: {original_question}")
        return {
            "question": original_question,
            "original_question": original_question,
            "translated_question": None,
            "is_korean": False
        }

    # 한글 질문 번역
    print(f"[translate] 한글 질문 감지 - 영어로 번역 중...")
    print(f"[translate] 원본: {original_question}")

    try:
        # 번역 프롬프트
        translate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional translator specializing in AI/ML/DL research topics.
Translate the Korean question into English with academic terminology.
Keep technical terms accurate and use standard AI/ML terminology.
Output ONLY the translated English text, nothing else."""),
            ("human", "{korean_text}")
        ])

        chain = translate_prompt | llm | StrOutputParser()
        translated = chain.invoke({"korean_text": original_question}).strip()

        print(f"[translate] 번역 완료: {translated}")

        return {
            "question": translated,  # 검색에 사용될 질문
            "original_question": original_question,
            "translated_question": translated,
            "is_korean": True
        }

    except Exception as e:
        print(f"[ERROR] 번역 중 오류 발생: {e}")
        print(f"[translate] 번역 실패 - 원본 질문 사용")
        return {
            "question": original_question,
            "original_question": original_question,
            "translated_question": None,
            "is_korean": True
        }


def retrieve_node(state: GraphState) -> dict:
    """
    노드 1: 하이브리드 검색 (BM25 + Vector Search with RRF)

    BM25가 없으면 벡터 검색만 사용하고,
    둘 다 있으면 RRF로 통합한다.
    """
    print("\n" + "="*60)
    print("[NODE: retrieve] 하이브리드 검색 시작 (BM25 + Vector + RRF)")
    print("="*60)

    question = state["question"]
    vectorstore = state.get("_vectorstore")
    bm25_retriever = state.get("_bm25_retriever")

    # ✅ 최소 조건: VectorStore는 반드시 있어야 함
    if vectorstore is None:
        print("[ERROR] VectorStore가 로드되지 않음")
        return {
            "documents": [],
            "doc_scores": [],
            "cluster_id": None,
            "search_type": "vector"  # 의미상만 표시
        }

    use_bm25 = bm25_retriever is not None

    try:
        # 1. 벡터 유사도 검색
        vector_docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)
        vector_docs = [doc for doc, score in vector_docs_with_scores]
        print(f"[retrieve] 벡터 검색: {len(vector_docs)}개 문서")

        # BM25가 없으면 그냥 벡터 검색 결과만 써버리기
        if not use_bm25:
            print("[retrieve] BM25Retriever 없음 → 벡터 검색만 사용")
            documents = vector_docs
            scores = [score for doc, score in vector_docs_with_scores]

            cluster_id = None
            if documents:
                cluster_id = documents[0].metadata.get("cluster_id", -1)

            return {
                "documents": documents,
                "doc_scores": scores,
                "cluster_id": cluster_id,
                "search_type": "vector"
            }

        # 2. BM25 키워드 검색 (BM25가 존재하는 경우만)
        bm25_docs = bm25_retriever.invoke(question)
        print(f"[retrieve] BM25 검색: {len(bm25_docs)}개 문서")

        # 3. RRF (Reciprocal Rank Fusion) 점수 계산
        RRF_K = 60  # 표준 RANK_BIAS 값
        fusion_scores = {}
        doc_map = {}

        # 3-1. 벡터 검색 결과 처리
        for rank, (doc, _score) in enumerate(vector_docs_with_scores):
            doc_key = get_doc_hash_key(doc)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            score = 1 / (RRF_K + rank + 1)  # rank는 0부터 시작하므로 +1
            fusion_scores[doc_key] = fusion_scores.get(doc_key, 0.0) + score

        # 3-2. BM25 검색 결과 처리
        for rank, doc in enumerate(bm25_docs):
            doc_key = get_doc_hash_key(doc)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            score = 1 / (RRF_K + rank + 1)
            fusion_scores[doc_key] = fusion_scores.get(doc_key, 0.0) + score

        # 4. 점수로 정렬 및 상위 5개 추출
        sorted_items = sorted(
            fusion_scores.items(), key=lambda x: x[1], reverse=True
        )

        documents = []
        scores = []
        for doc_key, score in sorted_items[:5]:
            documents.append(doc_map[doc_key])
            scores.append(score)

        if not documents:
            print("[retrieve] RRF 통합 결과 문서 없음")
            return {
                "documents": [],
                "doc_scores": [],
                "cluster_id": None,
                "search_type": "hybrid"
            }

        # 첫 번째 문서의 cluster_id 추출
        cluster_id = documents[0].metadata.get("cluster_id", -1)

        print(f"[retrieve] RRF 통합 완료: {len(documents)}개 문서")
        print(f"[retrieve] 최상위 RRF 점수: {scores[0]:.4f}")
        print(f"[retrieve] Cluster ID: {cluster_id}")

        return {
            "documents": documents,
            "doc_scores": scores,
            "cluster_id": cluster_id,
            "search_type": "hybrid"
        }

    except Exception as e:
        print(f"[ERROR] 검색 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "documents": [],
            "doc_scores": [],
            "cluster_id": None,
            "search_type": "hybrid"
        }



def evaluate_document_relevance_node(state: GraphState) -> dict:
    """
    노드 2: 문서 관련성 평가

    검색된 문서의 RRF 스코어를 기반으로 관련성을 3단계로 분류합니다.
    RRF 스코어는 높을수록 관련성이 높습니다.
    
    - HIGH (score >= 0.020): 매우 관련성이 높음
    - MEDIUM (0.015 <= score < 0.020): 보통 관련성
    - LOW (score < 0.015): 관련성 낮음

    Args:
        state: 현재 GraphState

    Returns:
        dict: relevance_level을 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: evaluate] 문서 관련성 평가")
    print("="*60)

    # 0) LLM 기반 토픽 가드: AI/ML 관련이 아니면 바로 LOW로 처리
    original_q = state.get("original_question", "")
    llm = state.get("_llm")

    try:
        if not is_ai_ml_related_by_llm(original_q, llm):
            print(f"[evaluate] LLM 판별: 비 AI/ML 질문 → LOW로 처리: {original_q}")
            return {"relevance_level": "low"}
    except Exception as e:
        print(f"[WARN] 토픽 판별 중 오류 발생, 스코어 기반 평가로 진행: {e}")

    # --- 여기서부터는 기존 RRF 스코어 기반 로직 그대로 ---
    documents = state["documents"]
    scores = state["doc_scores"]

    if not documents or not scores:
        print("[evaluate] 문서 없음 → LOW")
        return {"relevance_level": "low"}

    # ✅ 수정: RRF는 높을수록 좋으므로 max 사용
    best_score = max(scores)

    # ✅ 수정: RRF 스코어 특성에 맞는 임계값
    # RRF with K=60: 1등 = 1/61 ≈ 0.0164, 2등 = 1/62 ≈ 0.0161
    # 두 검색 방법에서 모두 1등이면: 0.0164 * 2 = 0.0328
    if best_score >= 0.020:  # 상위권에서 중복 발견
        level = "high"
    elif best_score >= 0.015:  # 한쪽에서만 상위권
        level = "medium"
    else:  # 낮은 순위
        level = "low"

    print(f"[evaluate] 최상위 RRF 스코어: {best_score:.6f} → {level.upper()}")
    
    # 디버깅을 위한 상세 정보
    if len(scores) >= 3:
        print(f"[evaluate] Top-3 스코어: {scores[0]:.6f}, {scores[1]:.6f}, {scores[2]:.6f}")

    return {"relevance_level": level}


def cluster_similarity_check_node(state: GraphState) -> dict:
    """
    노드 3: 클러스터 유사도 체크

    검색된 문서와 같은 클러스터에 속한 추가 문서들을 검색하여
    클러스터 내 평균 유사도를 계산합니다. 클러스터 밀도(density)도
    함께 고려하여 HIGH/LOW를 결정합니다.

    조건:
    - HIGH: avg_score <= 0.9 AND density >= 1.0
    - LOW: 그 외

    Args:
        state: 현재 GraphState

    Returns:
        dict: cluster_similarity_score, documents를 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: cluster_check] 클러스터 유사도 체크")
    print("="*60)

    cluster_id = state["cluster_id"]
    question = state["question"]
    vectorstore = state.get("_vectorstore")
    cluster_metadata_path = state.get("_cluster_metadata_path")

    if cluster_id is None or cluster_id == -1:
        print("[cluster_check] cluster_id 없음 → LOW")
        return {
            "cluster_similarity_score": 0.0,
            "search_type": "cluster"
        }

    try:
        # 클러스터 메타데이터 로드
        with open(cluster_metadata_path, 'r', encoding='utf-8') as f:
            cluster_meta = json.load(f)

        cluster_info = cluster_meta["clusters"].get(str(cluster_id))
        if not cluster_info:
            print(f"[cluster_check] cluster_id={cluster_id} 정보 없음 → LOW")
            return {
                "cluster_similarity_score": 0.0,
                "search_type": "cluster"
            }

        cluster_density = cluster_info.get("density", 0.0)
        print(f"[cluster_check] Cluster {cluster_id} 밀도: {cluster_density:.3f}")

        # 같은 클러스터 내 추가 문서 검색 (post-filtering 방식)
        # 더 많은 문서를 검색한 후 cluster_id로 필터링
        all_docs = vectorstore.similarity_search_with_score(question, k=20)

        # cluster_id로 필터링
        filtered_docs = [
            (doc, score) for doc, score in all_docs
            if doc.metadata.get("cluster_id") == cluster_id
        ][:5]

        if not filtered_docs:
            print("[cluster_check] 클러스터 내 추가 문서 없음 → LOW")
            return {
                "cluster_similarity_score": 0.0,
                "search_type": "cluster"
            }

        # 평균 점수 계산
        avg_score = sum(score for doc, score in filtered_docs) / len(filtered_docs)

        print(f"[cluster_check] 클러스터 내 문서 {len(filtered_docs)}개 검색")
        print(f"[cluster_check] 평균 점수: {avg_score:.4f}")

        # 기존 문서에 추가 문서 병합 (중복 제거)
        existing_docs = state["documents"]
        additional_docs = [doc for doc, score in filtered_docs[:3]]

        # 중복 제거 (page_content 기준)
        existing_contents = {doc.page_content for doc in existing_docs}
        unique_additional = [
            doc for doc in additional_docs
            if doc.page_content not in existing_contents
        ]

        merged_docs = existing_docs + unique_additional

        print(f"[cluster_check] 추가 문서 {len(unique_additional)}개 병합")

        return {
            "cluster_similarity_score": avg_score,
            "documents": merged_docs,
            "search_type": "cluster"
        }

    except Exception as e:
        print(f"[ERROR] 클러스터 체크 중 오류: {e}")
        return {
            "cluster_similarity_score": 0.0,
            "search_type": "cluster"
        }


def web_search_node(state: GraphState) -> dict:
    """
    노드 4: 웹 검색

    클러스터 유사도가 낮을 때 Tavily API를 사용하여 웹 검색을 수행합니다.
    Tavily API가 실패하면 관련 챗봇이 아니라는 메시지를 반환합니다.
    검색 결과에서 URL을 추출하여 메타데이터에 포함시킵니다.

    Args:
        state: 현재 GraphState

    Returns:
        dict: documents, search_type을 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: web_search] 웹 검색 시작")
    print("="*60)

    question = state["question"]

    # Tavily API 시도
    try:
        from langchain_community.retrievers import TavilySearchAPIRetriever
        print("[web_search] Tavily API 사용 중...")

        retriever = TavilySearchAPIRetriever(k=5)
        web_docs_raw = retriever.invoke(question)

        print(f"[web_search] Tavily로 {len(web_docs_raw)}개 문서 검색 완료")

        # ★ 개선: 웹 검색 결과의 메타데이터 정리
        processed_web_docs = []
        for i, doc in enumerate(web_docs_raw):
            # Tavily가 반환하는 메타데이터: title, source, score 등
            original_meta = doc.metadata
            
            # 제목과 출처 추출
            title = original_meta.get('title', '웹 검색 결과')
            source_url = original_meta.get('source', '')
            score = original_meta.get('score', 0.5)
            
            # 새로운 Document 생성 (메타데이터 형식 통일)
            web_doc = Document(
                page_content=doc.page_content,
                metadata={
                    'title': title,  # 웹 페이지 제목
                    'source': source_url,  # URL
                    'source_type': 'web',  # 웹 검색임을 명시
                    'score': score,
                    'index': i
                }
            )
            processed_web_docs.append(web_doc)
            
            print(f"  [{i+1}] {title[:60]}...")

        return {
            "documents": processed_web_docs,
            "search_type": "web",
            "doc_scores": [doc.metadata['score'] for doc in processed_web_docs]
        }

    except Exception as e:
        print(f"[web_search] Tavily 실패: {e}")
        print("[web_search] AI/ML 논문 관련 챗봇이 아닙니다")

        # Tavily 실패 시 관련 챗봇이 아니라는 메시지를 담은 문서 반환
        return {
            "documents": [Document(
                page_content="이 질문은 AI/ML 연구 논문 챗봇의 범위를 벗어납니다.",
                metadata={"source": "system", "source_type": "error"}
            )],
            "search_type": "web_failed",
            "doc_scores": [2.0]  # 높은 점수 = 관련성 낮음
        }


def generate_final_answer_node(state: GraphState) -> dict:
    """
    노드 5: 최종 답변 생성

    하이브리드 검색 결과(논문 + 웹)를 컨텍스트로 사용하여
    AI Tech Trend Navigator 스타일로 답변을 생성한다.
    """

    print("\n" + "=" * 60)
    print("[NODE: generate] 최종 답변 생성 (AI Tech Trend Navigator 프롬프트)")
    print("=" * 60)

    original_question = state["original_question"]      # 사용자가 입력한 질문
    documents = state["documents"]
    search_type = state.get("search_type", "internal")
    is_korean = state.get("is_korean", False)
    llm = state.get("_llm")

    # 번역 정보 출력 (디버그용)
    if is_korean:
        print(f"[generate] 원본 질문(한글): {original_question}")
        print(f"[generate] 검색 질문(영어): {state.get('question')}")

    # 1) CONTEXT 블록 만들기
    if not documents:
        # 너가 짜놓은 프롬프트 규약에 맞춤
        context_str = "NO_RELEVANT_PAPERS"
    else:
        context_blocks = []
        for i, doc in enumerate(documents[:5], 1):  # 최대 5개 문서만 사용
            meta = doc.metadata or {}

            title = meta.get("title", meta.get("paper_name", "No information"))
            authors = meta.get("authors", "No information")
            hf_url = meta.get("huggingface_url", meta.get("source", "No information"))
            gh_url = meta.get("github_url", "No information")
            upvote = meta.get("upvote", "No information")
            year = meta.get("publication_year", "No information")
            total_chunks = meta.get("total_chunks", "No information")
            doc_id = meta.get("doc_id", "No information")
            chunk_index = meta.get("chunk_index", "No information")

            block = f"""
[DOCUMENT {i}]
page_content:
{doc.page_content[:1000]}

metadata:
  title: {title}
  huggingface_url: {hf_url}
  github_url: {gh_url}
  upvote: {upvote}
  authors: {authors}
  publication_year: {year}
  total_chunks: {total_chunks}
  doc_id: {doc_id}
  chunk_index: {chunk_index}
"""
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

    # 2) 프롬프트 정의 (AI Tech Trend Navigator 버전)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are "AI Tech Trend Navigator", an expert assistant for AI/ML research papers.

[Role]
- You help users understand and leverage recent AI/ML papers collected from HuggingFace DailyPapers.
- Your main goals are:
  - Summarize and compare relevant papers clearly.
  - Explain core ideas in simple terms.
  - Highlight practical use-cases and implications for real-world services or products.

[Inputs]
The system provides:
- user_question: the user’s question.
- context: a set of retrieved documents, formatted as a single text block.
- Sometimes the context may be exactly the string "NO_RELEVANT_PAPERS".
- Each paper in the context may contain:
  - page_content (text)
  - metadata:
    - title
    - github_url (optional)
    - huggingface_url
    - upvote (integer)
    - authors
    - publication_year
    - total_chunks
    - doc_id
    - chunk_index

You must rely only on:
- the given context, and
- general, high-level AI/ML knowledge.
Do NOT invent specific paper titles, authors, datasets, metrics, or numerical results that are not supported by the context.

[Context Handling]
- If the context is "NO_RELEVANT_PAPERS", it means:
  - The retrieval system could not find any clearly relevant papers.
  - In this case, you may answer purely from your own general AI/ML knowledge.
  - Do NOT fabricate specific paper titles, authors, datasets, or numerical results.
  - You may skip the "Related papers" section or keep it very generic.
- If the context contains one or more papers:
  - Prefer to base your answer on those papers.
  - Use only the papers that are reasonably related to the user’s question.

[Main Tasks]
1. Understand the user’s intent
   - Roughly classify the question as one of:
     (a) concept/background explanation
     (b) single-paper summary
     (c) comparison or trend analysis
     (d) practical application and use-case ideas
   - If the intent is ambiguous, make a reasonable assumption and continue.
2. Use only the relevant papers
   - Focus on the most relevant 1-3 papers.
   - Ignore clearly unrelated ones.
3. Summarize each selected paper
   - What problem it tries to solve.
   - What approach/model/idea it uses.
   - What is new or strong.
   - Any obvious limitations or trade-offs.
4. Produce a synthesized answer
   - Don’t just list papers. Directly answer the user’s question.
   - When relevant, relate ideas to RAG, long-context, multimodal, etc.
   - Suggest how to apply the ideas in real projects/services.
5. Be honest about uncertainty
   - If context is weak, say so.
   - Suggest what extra info or papers would help.

[Style]
- Prefer clear, concise sentences over heavy academic wording.
- Briefly explain technical terms when needed.
- Never fabricate paper titles, authors, datasets, or numerical results.
- Regardless of the input language, ALWAYS respond in Korean.
"""
        ),
        (
            "human",
            """
[QUESTION]
{question}

[CHAT HISTORY]
{chat_history}

[Context]
The following CONTEXT block may contain 0 or more papers.
If it is "NO_RELEVANT_PAPERS", please answer from your general AI/ML knowledge.

[CONTEXT]
======== START ========
{context}
======== END =========

Please structure your answer as follows (flexible, but try to follow this):

1) One-line summary  
2) Key insights (3-6 bullets)  
3) Detailed explanation  
4) Sources summary  
    Organize the papers used above based on metadata:
        [title: ...]
            - authors: ...
            - huggingface_url: ...
            - github_url: ...
            - 👍 23

        [title: ...]
            - authors: ...
            - huggingface_url: ...
            - github_url: ...
            - 👍 10

⚠ For information not present in the metadata, write “No information available.”
⚠ Do not hallucinate papers or details not shown in context.
⚠ Regardless of the input language, ALWAYS respond in Korean.
⚠ If you want to use bold, italics, and other text formatting elements, use ‘HTML’. Markdown formatting is not supported.
    """)
    ])

    # 3) 체인 실행
    try:
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question": original_question,
            "chat_history": "",      # 아직 히스토리 없으니 빈 문자열
            "context": context_str
        })
        print("[generate] 답변 생성 완료 (AI Tech Trend Navigator 스타일)")
    except Exception as e:
        print(f"[ERROR] 답변 생성 중 오류: {e}")
        answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    # 4) sources 및 참고 URL 구성
    sources: List[Dict[str, Any]] = []

    return {
        "answer": answer,
        "sources": sources
    }


def reject_node(state: GraphState) -> dict:
    """
    노드 6: 거부 응답

    관련성이 낮은 질문에 대해 정중하게 거부하고
    더 나은 질문 방법을 제안합니다.

    Args:
        state: 현재 GraphState

    Returns:
        dict: answer, sources, search_type을 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: reject] 질문 거부")
    print("="*60)

    question = state["original_question"]

    answer = f"""죄송합니다. '{question}'와 관련된 적절한 문서를 찾지 못했습니다.

다음과 같이 시도해보세요:
1. 더 구체적인 키워드 사용 (예: "transformer", "attention mechanism")
2. 영어 학술 용어 사용
3. 질문을 다시 표현해보기
4. AI/ML/DL 관련 주제로 질문하기

이 시스템은 HuggingFace에 게시된 AI/ML 연구 논문을 기반으로 답변합니다."""

    print(f"[reject] 거부 메시지 반환")

    return {
        "answer": answer,
        "sources": [],
        "search_type": "rejected"
    }


# ===== SECTION 5: CONDITIONAL EDGE FUNCTIONS =====

def route_after_evaluate(state: GraphState) -> Literal["generate", "cluster_check", "reject"]:
    """
    조건부 라우팅: evaluate 노드 이후

    관련성 수준에 따라 다음 노드를 결정합니다:
    - high → generate (바로 답변 생성)
    - medium → cluster_check (클러스터 유사도 확인)
    - low → reject (질문 거부)

    Args:
        state: 현재 GraphState

    Returns:
        str: 다음 노드 이름
    """
    level = state.get("relevance_level", "low")

    print(f"\n[ROUTING] evaluate → {level.upper()}", end=" → ")

    if level == "high":
        print("generate")
        return "generate"
    elif level == "medium":
        print("cluster_check")
        return "cluster_check"
    else:  # low
        print("reject")
        return "reject"


def route_after_cluster_check(state: GraphState) -> Literal["generate", "web_search"]:
    """
    조건부 라우팅: cluster_check 노드 이후

    클러스터 유사도와 밀도를 기반으로 다음 노드를 결정합니다:
    - HIGH (avg_score <= 0.9 AND density >= 1.0) → generate
    - LOW → web_search

    Args:
        state: 현재 GraphState

    Returns:
        str: 다음 노드 이름
    """
    cluster_score = state.get("cluster_similarity_score", 0.0)
    cluster_id = state.get("cluster_id", -1)
    cluster_metadata_path = state.get("_cluster_metadata_path")

    print(f"\n[ROUTING] cluster_check → ", end="")

    # 클러스터 메타데이터에서 밀도 확인
    try:
        with open(cluster_metadata_path, 'r', encoding='utf-8') as f:
            cluster_meta = json.load(f)

        cluster_info = cluster_meta["clusters"].get(str(cluster_id), {})
        density = cluster_info.get("density", 0.0)

        # HIGH 조건: 평균 점수 <= 0.9 AND 밀도 >= 1.0
        if cluster_score <= 0.9 and density >= 1.0:
            print(f"HIGH (score={cluster_score:.3f}, density={density:.3f}) → generate")
            return "generate"
        else:
            print(f"LOW (score={cluster_score:.3f}, density={density:.3f}) → web_search")
            return "web_search"

    except Exception as e:
        print(f"ERROR ({e}) → web_search")
        return "web_search"


# ===== SECTION 6: GRAPH BUILDER =====

def build_langgraph_rag():
    """
    LangGraph StateGraph 구축 및 컴파일

    모든 노드와 조건부 엣지를 포함한 완전한 그래프를 생성합니다.

    Returns:
        CompiledGraph: 컴파일된 LangGraph 객체
    """
    print("\n" + "="*60)
    print("[GRAPH BUILD] LangGraph 구축 시작")
    print("="*60)

    # StateGraph 생성
    graph = StateGraph(GraphState)

    # 노드 추가
    graph.add_node("translate", translate_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_document_relevance_node)
    graph.add_node("cluster_check", cluster_similarity_check_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_final_answer_node)
    graph.add_node("reject", reject_node)

    print("[GRAPH] 7개 노드 추가 완료 (translate 포함)")

    # 기본 엣지
    graph.add_edge(START, "translate")  # START → translate (한글 번역)
    graph.add_edge("translate", "retrieve")  # translate → retrieve
    graph.add_edge("retrieve", "evaluate")

    # 조건부 엣지
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "generate": "generate",
            "cluster_check": "cluster_check",
            "reject": "reject"
        }
    )

    graph.add_conditional_edges(
        "cluster_check",
        route_after_cluster_check,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )

    # 종료 엣지
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("reject", END)

    print("[GRAPH] 모든 엣지 추가 완료")

    # 컴파일
    compiled_graph = graph.compile()
    print("[GRAPH] 컴파일 완료")

    return compiled_graph


# ===== SECTION 7: INTERACTIVE MODE =====

def run_interactive_mode(vectorstore, llm, bm25_retriever, cluster_metadata_path, langgraph_app):
    """
    대화형 모드 실행

    사용자가 CMD에서 직접 질문을 입력하고 답변을 받을 수 있습니다.
    'quit', 'exit', 'q'를 입력하면 종료됩니다.

    Args:
        vectorstore: VectorStore 객체
        llm: LLM 객체
        bm25_retriever: BM25Retriever 객체
        cluster_metadata_path: 클러스터 메타데이터 경로
        langgraph_app: 컴파일된 LangGraph 앱
    """
    print("\n" + "="*60)
    print("대화형 모드 시작")
    print("="*60)
    print("질문을 입력하세요 (종료: quit, exit, q)")
    print("="*60 + "\n")

    while True:
        try:
            # 사용자 입력 받기
            question = input("\n[질문] >> ").strip()

            # 종료 명령 체크
            if question.lower() in ['quit', 'exit', 'q', '종료']:
                print("\n프로그램을 종료합니다.")
                break

            # 빈 입력 체크
            if not question:
                print("[경고] 질문을 입력해주세요.")
                continue

            print(f"\n{'#'*60}")
            print(f"# 질문: {question}")
            print(f"{'#'*60}")

            # 초기 상태 구성
            initial_state = {
                "question": "",  # 번역 후 업데이트됨
                "original_question": question,  # 사용자 입력 원본
                "translated_question": None,
                "is_korean": False,
                "documents": [],
                "doc_scores": [],
                "cluster_id": None,
                "cluster_similarity_score": None,
                "search_type": "",
                "relevance_level": "",
                "answer": "",
                "sources": [],
                "_vectorstore": vectorstore,
                "_llm": llm,
                "_bm25_retriever": bm25_retriever,
                "_cluster_metadata_path": cluster_metadata_path
            }

            # LangGraph 실행
            result = langgraph_app.invoke(initial_state)

            # 결과 출력
            print("\n" + "="*60)
            print("[최종 결과]")
            print("="*60)
            print(f"원본 질문: {result.get('original_question')}")
            if result.get('is_korean') and result.get('translated_question'):
                print(f"번역된 질문: {result.get('translated_question')}")
            print(f"검색 유형: {result.get('search_type')}")
            print(f"관련성 수준: {result.get('relevance_level')}")
            print(f"클러스터 ID: {result.get('cluster_id')}")
            print(f"\n[답변]\n{result['answer']}")
            print(f"\n[출처 개수] {len(result.get('sources', []))}")

            # 출처 상세 정보 (처음 3개만)
            sources = result.get('sources', [])
            if sources:
                print("\n[주요 출처]")
                for i, source in enumerate(sources[:3], 1):
                    if source.get('type') == 'paper':
                        print(f"  {i}. {source.get('title')} (upvote: {source.get('upvote', 0)})")
                        # HuggingFace URL 표시
                        if source.get('source'):
                            print(f"     HuggingFace: {source.get('source')}")
                        # GitHub URL 표시
                        if source.get('github_url'):
                            print(f"     GitHub: {source.get('github_url')}")
                    else:
                        print(f"  {i}. {source.get('title')} (웹)")
                        # 웹 URL 표시
                        if source.get('source'):
                            print(f"     URL: {source.get('source')}")

            print("\n" + "-"*60)

        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] 오류 발생: {e}")
            import traceback
            traceback.print_exc()


# ===== SECTION 8: EXTERNAL API FUNCTIONS =====

# 전역 변수로 리소스 관리 (외부에서 사용)
_vectorstore = None
_llm = None
_bm25_retriever = None
_cluster_metadata_path = None
_langgraph_app = None


def initialize_langgraph_system(
    model_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0
) -> dict:
    """
    LangGraph 시스템 초기화 (외부 서버에서 호출용)

    이 함수는 FastAPI/Flask 서버 시작 시 한 번 호출하여
    VectorStore, LLM, LangGraph를 초기화합니다.

    Args:
        model_name: 임베딩 모델 이름 (기본값: 환경변수에서 로드)
        chunk_size: 청크 크기 (기본값: 환경변수에서 로드)
        chunk_overlap: 청크 오버랩 (기본값: 환경변수에서 로드)
        llm_model: LLM 모델 이름
        llm_temperature: LLM temperature 설정

    Returns:
        dict: 초기화 상태 정보

    Example:
        >>> from langgraph_test import initialize_langgraph_system
        >>> result = initialize_langgraph_system()
        >>> print(result['status'])
        'success'
    """
    global _vectorstore, _llm, _bm25_retriever, _cluster_metadata_path, _langgraph_app

    try:
        print("\n[INIT] LangGraph 시스템 초기화 중...")

        # 환경변수에서 기본값 로드
        if model_name is None:
            model_name = MODEL_NAME
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = CHUNK_OVERLAP

        # VectorStore 로드
        from vectordb import load_vectordb
        print(f"[LOADING] VectorStore 로딩 중... (model: {model_name})")
        _vectorstore = load_vectordb(model_name, chunk_size, chunk_overlap)

        # BM25 Retriever 초기화
        print("[LOADING] BM25 Retriever 초기화 중...")
        collection_data = _vectorstore._collection.get(include=['documents', 'metadatas'])
        all_documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(collection_data['documents'], collection_data['metadatas'])
        ]
        if not all_documents:
            raise ValueError('Chroma DB에 문서가 없습니다. BM25 인덱스 생성이 불가합니다.')

        _bm25_retriever = BM25Retriever.from_documents(all_documents)
        _bm25_retriever.k = 3  # BM25 검색 결과 개수 설정
        print(f"[SUCCESS] BM25 인덱스 생성 완료: {len(all_documents)}개 문서")

        # LLM 초기화
        print(f"[LOADING] LLM 초기화 중... (model: {llm_model})")
        _llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

        # Cluster metadata path
        _cluster_metadata_path = str(PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json")

        # LangGraph 컴파일
        print("[LOADING] LangGraph 컴파일 중...")
        _langgraph_app = build_langgraph_rag()

        print("[SUCCESS] LangGraph 시스템 초기화 완료!\n")

        return {
            'status': 'success',
            'message': 'LangGraph system initialized successfully',
            'config': {
                'model_name': model_name,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'llm_model': llm_model,
                'llm_temperature': llm_temperature
            }
        }

    except Exception as e:
        print(f"[ERROR] 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


def ask_question(question: str, verbose: bool = False) -> dict:
    """
    LangGraph를 사용하여 질문에 답변 (외부 서버에서 호출용)

    이 함수는 FastAPI/Flask 엔드포인트에서 호출하여
    사용자 질문에 대한 답변을 생성합니다.

    Args:
        question: 사용자 질문
        verbose: 상세 로그 출력 여부

    Returns:
        dict: 답변 및 출처 정보
        {
            'success': bool,
            'question': str,
            'answer': str,  # URL 포함된 답변
            'sources': List[dict],
            'metadata': {
                'search_type': str,
                'relevance_level': str,
                'cluster_id': int,
                'is_korean': bool,
                'translated_question': str
            }
        }

    Example:
        >>> from langgraph_test import initialize_langgraph_system, ask_question
        >>> initialize_langgraph_system()
        >>> result = ask_question("What is transformer architecture?")
        >>> print(result['answer'])
    """
    global _vectorstore, _llm, _bm25_retriever, _cluster_metadata_path, _langgraph_app

    # 초기화 확인
    if _langgraph_app is None:
        return {
            'success': False,
            'error': 'LangGraph system not initialized. Call initialize_langgraph_system() first.'
        }

    try:
        if verbose:
            print(f"\n[QUESTION] {question}")

        # 초기 상태 구성
        initial_state = {
            "question": "",  # 번역 후 업데이트됨
            "original_question": question,
            "translated_question": None,
            "is_korean": False,
            "documents": [],
            "doc_scores": [],
            "cluster_id": None,
            "cluster_similarity_score": None,
            "search_type": "",
            "relevance_level": "",
            "answer": "",
            "sources": [],
            "_vectorstore": _vectorstore,
            "_llm": _llm,
            "_bm25_retriever": _bm25_retriever,
            "_cluster_metadata_path": _cluster_metadata_path
        }

        # LangGraph 실행
        result = _langgraph_app.invoke(initial_state)

        if verbose:
            print(f"[ANSWER] 답변 생성 완료")
            print(f"[SOURCES] {len(result.get('sources', []))}개 출처")

        # 응답 구성
        return {
            'success': True,
            'question': question,
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'metadata': {
                'search_type': result.get('search_type', ''),
                'relevance_level': result.get('relevance_level', ''),
                'cluster_id': result.get('cluster_id'),
                'is_korean': result.get('is_korean', False),
                'translated_question': result.get('translated_question')
            }
        }

    except Exception as e:
        print(f"[ERROR] 질문 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def get_system_status() -> dict:
    """
    시스템 상태 확인 (외부 서버에서 호출용)

    Returns:
        dict: 시스템 초기화 상태
    """
    return {
        'initialized': _langgraph_app is not None,
        'vectorstore_loaded': _vectorstore is not None,
        'llm_loaded': _llm is not None,
        'bm25_retriever_loaded': _bm25_retriever is not None,
        'cluster_metadata_loaded': _cluster_metadata_path is not None
    }


# ===== SECTION 9: MAIN EXECUTION =====

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LangGraph RAG System - 대화형 모드")
    print("="*60)

    # 리소스 초기화
    print("\n[INIT] 리소스 초기화 시작...")

    try:
        from vectordb import load_vectordb

        # VectorStore 로드
        print("[LOADING] VectorStore 로딩 중...")
        vectorstore = load_vectordb(MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)

        # BM25 Retriever 초기화
        print("[LOADING] BM25 Retriever 초기화 중...")
        collection_data = vectorstore._collection.get(include=['documents', 'metadatas'])
        all_documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(collection_data['documents'], collection_data['metadatas'])
        ]
        if not all_documents:
            raise ValueError('Chroma DB에 문서가 없습니다. BM25 인덱스 생성이 불가합니다.')

        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 3  # BM25 검색 결과 개수 설정
        print(f"[SUCCESS] BM25 인덱스 생성 완료: {len(all_documents)}개 문서")

        # LLM 초기화
        print("[LOADING] LLM 초기화 중...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Cluster metadata path
        cluster_metadata_path = str(PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json")

        # LangGraph 컴파일
        print("[LOADING] LangGraph 컴파일 중...")
        langgraph_app = build_langgraph_rag()

        print("\n[SUCCESS] 모든 리소스 초기화 완료!\n")

        # 대화형 모드 실행
        run_interactive_mode(vectorstore, llm, bm25_retriever, cluster_metadata_path, langgraph_app)

    except Exception as e:
        print(f"\n[ERROR] 초기화 또는 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
