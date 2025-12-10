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
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    question: str
    original_question: str
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
    _cluster_metadata_path: str


# ===== SECTION 4: HELPER FUNCTIONS =====

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


# ===== SECTION 5: NODE FUNCTIONS =====

def translate(state: GraphState) -> dict:
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


def retrieve(state: GraphState) -> dict:
    """
    노드 1: 벡터 유사도 검색

    Chroma VectorStore를 사용하여 질문과 유사한 문서를 검색합니다.
    상위 5개 문서와 그들의 유사도 점수를 반환하며,
    첫 번째 문서의 cluster_id를 추출합니다.

    Args:
        state: 현재 GraphState

    Returns:
        dict: documents, doc_scores, cluster_id, search_type를 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: retrieve] 벡터 검색 시작")
    print("="*60)

    question = state["question"]
    vectorstore = state.get("_vectorstore")

    if not vectorstore:
        print("[ERROR] VectorStore가 로드되지 않음")
        return {
            "documents": [],
            "doc_scores": [],
            "cluster_id": None,
            "search_type": "internal"
        }

    try:
        # 벡터 유사도 검색 (거리 점수 포함)
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)

        documents = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]

        # 첫 번째 문서의 cluster_id 추출
        cluster_id = None
        if documents:
            cluster_id = documents[0].metadata.get("cluster_id", -1)

        print(f"[retrieve] 검색 완료: {len(documents)}개 문서")
        if scores:
            print(f"[retrieve] 최상위 문서 점수: {scores[0]:.4f}")
        if cluster_id is not None:
            print(f"[retrieve] Cluster ID: {cluster_id}")

        return {
            "documents": documents,
            "doc_scores": scores,
            "cluster_id": cluster_id,
            "search_type": "internal"
        }

    except Exception as e:
        print(f"[ERROR] 검색 중 오류: {e}")
        return {
            "documents": [],
            "doc_scores": [],
            "cluster_id": None,
            "search_type": "internal"
        }


def evaluate_document_relevance(state: GraphState) -> dict:
    """
    노드 2: 문서 관련성 평가

    검색된 문서의 유사도 점수를 기반으로 관련성을 3단계로 분류합니다.
    - HIGH (score <= 0.8): 매우 관련성이 높음
    - MEDIUM (0.8 < score <= 1.2): 보통 관련성
    - LOW (score > 1.2): 관련성 낮음

    Args:
        state: 현재 GraphState

    Returns:
        dict: relevance_level을 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: evaluate] 문서 관련성 평가")
    print("="*60)

    documents = state["documents"]
    scores = state["doc_scores"]

    if not documents or not scores:
        print("[evaluate] 문서 없음 → LOW")
        return {"relevance_level": "low"}

    # 최상위 문서의 점수로 판단 (낮을수록 좋음)
    best_score = min(scores)

    # 점수 기준에 따른 분류
    if best_score <= 0.8:
        level = "high"
    elif best_score <= 1.2:
        level = "medium"
    else:
        level = "low"

    print(f"[evaluate] 최상위 점수: {best_score:.4f} → {level.upper()}")

    return {"relevance_level": level}


def cluster_similarity_check(state: GraphState) -> dict:
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


def web_search(state: GraphState) -> dict:
    """
    노드 4: 웹 검색

    클러스터 유사도가 낮을 때 Tavily API를 사용하여 웹 검색을 수행합니다.
    Tavily API가 실패하면 관련 챗봇이 아니라는 메시지를 반환합니다.

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


def generate_final_answer(state: GraphState) -> dict:
    """
    노드 5: 최종 답변 생성

    검색된 문서들을 컨텍스트로 사용하여 LLM이 한글로 답변을 생성합니다.
    답변 형식:
    1. 핵심 요약 (1-2문장)
    2. 주요 인사이트 (불릿 포인트)
    3. 상세 설명

    Args:
        state: 현재 GraphState

    Returns:
        dict: answer, sources를 포함한 상태 업데이트
    """
    print("\n" + "="*60)
    print("[NODE: generate] 최종 답변 생성")
    print("="*60)

    original_question = state["original_question"]  # 원본 질문 (사용자가 입력한 질문)
    search_question = state["question"]  # 검색에 사용된 질문 (번역된 질문 또는 원본)
    documents = state["documents"]
    search_type = state.get("search_type", "internal")
    is_korean = state.get("is_korean", False)
    llm = state.get("_llm")

    # 번역 정보 출력
    if is_korean:
        print(f"[generate] 원본 질문(한글): {original_question}")
        print(f"[generate] 검색 질문(영어): {search_question}")

    # 웹 검색 실패 케이스 체크
    if search_type == "web_failed":
        print("[generate] 웹 검색 실패 - 관련 챗봇이 아님")
        answer = """죄송합니다. 이 질문은 AI/ML 연구 논문 챗봇의 범위를 벗어납니다.

이 챗봇은 HuggingFace에 게시된 AI/ML/DL 연구 논문을 기반으로 답변합니다.
다음과 같은 주제에 대해 질문해주세요:
- 딥러닝 모델 아키텍처 (Transformer, CNN, RNN 등)
- 자연어 처리 (NLP)
- 컴퓨터 비전
- 강화학습
- 생성 AI (LLM, Diffusion Models 등)
- 기타 AI/ML 관련 연구

AI/ML과 관련된 질문으로 다시 시도해주세요."""

        return {
            "answer": answer,
            "sources": []
        }

    # 컨텍스트 생성
    if not documents:
        context = "관련 문서를 찾지 못했습니다."
    else:
        context_blocks = []
        for i, doc in enumerate(documents[:5], 1):  # 최대 5개 문서 사용
            meta = doc.metadata

            # 문서 출처에 따라 다른 포맷 사용
            if meta.get("source_type") == "web":
                block = f"""
[웹 문서 {i}]
제목: {meta.get('title', 'N/A')}
출처: {meta.get('source', 'N/A')}
내용: {doc.page_content[:500]}...
"""
            else:
                block = f"""
[논문 {i}]
제목: {meta.get('title', meta.get('paper_name', 'N/A'))}
저자: {meta.get('authors', 'N/A')}
출처: {meta.get('huggingface_url', meta.get('source', 'N/A'))}
내용: {doc.page_content[:500]}...
"""
            context_blocks.append(block)

        context = "\n".join(context_blocks)

    # Prompt 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 AI/ML 연구 논문 전문가입니다.
주어진 문서를 바탕으로 사용자 질문에 한글로 답변하세요.

답변 형식:
1. 핵심 요약 (1-2문장)
2. 주요 인사이트 (3-5개 불릿 포인트)
3. 상세 설명

문서가 없거나 관련성이 낮으면 솔직히 알려주세요.
웹 검색 결과인 경우 출처가 웹임을 명시하세요."""),
        ("human", """질문: {question}

검색 유형: {search_type}

참고 문서:
{context}

위 정보를 바탕으로 한글로 답변해주세요.""")
    ])

    # Chain 실행
    try:
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question": original_question,  # 원본 질문으로 답변 생성
            "context": context,
            "search_type": search_type
        })

        print("[generate] 답변 생성 완료")

    except Exception as e:
        print(f"[ERROR] 답변 생성 중 오류: {e}")
        answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    # Sources 구성
    sources = []
    for doc in documents[:5]:
        meta = doc.metadata

        if meta.get("source_type") == "web":
            sources.append({
                "title": meta.get("title", "Unknown"),
                "source": meta.get("source", ""),
                "type": "web"
            })
        elif meta.get("source_type") != "error":
            sources.append({
                "title": meta.get("title", meta.get("paper_name", "Unknown")),
                "authors": meta.get("authors", ""),
                "source": meta.get("huggingface_url", ""),
                "github_url": meta.get("github_url", ""),
                "upvote": meta.get("upvote", 0),
                "type": "paper"
            })

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

    # ★ 수정: 원본 질문(사용자가 입력한 그대로) 사용
    original_question = state["original_question"]
    answer = f"""죄송합니다. '{original_question}'와 관련된 적절한 문서를 찾지 못했습니다.

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
    graph.add_node("translate", translate)
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate", evaluate_document_relevance)
    graph.add_node("cluster_check", cluster_similarity_check)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate_final_answer)
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

def run_interactive_mode(vectorstore, llm, cluster_metadata_path, langgraph_app):
    """
    대화형 모드 실행

    사용자가 CMD에서 직접 질문을 입력하고 답변을 받을 수 있습니다.
    'quit', 'exit', 'q'를 입력하면 종료됩니다.

    Args:
        vectorstore: VectorStore 객체
        llm: LLM 객체
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
                    else:
                        print(f"  {i}. {source.get('title')} (웹)")

            print("\n" + "-"*60)

        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] 오류 발생: {e}")
            import traceback
            traceback.print_exc()


# ===== SECTION 8: MAIN EXECUTION =====

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
        run_interactive_mode(vectorstore, llm, cluster_metadata_path, langgraph_app)

    except Exception as e:
        print(f"\n[ERROR] 초기화 또는 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
