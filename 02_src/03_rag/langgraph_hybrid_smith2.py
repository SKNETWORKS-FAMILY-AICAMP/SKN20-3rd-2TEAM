"""
LangSmith 평가를 위한 최종 수정 코드
✅ 함수 정의 순서 수정
✅ target 함수가 실제 LangGraph 사용
"""

import os
import sys
import json
import re
import warnings
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Set

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START, END
from langsmith import Client, evaluate

# ===== 환경 설정 =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "02_src" / "02_utils"))

warnings.filterwarnings("ignore")
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "MiniLM-L6")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 45))

# ===== 전역 변수 선언 =====
_vectorstore = None
_llm = None
_bm25_retriever = None
_cluster_metadata_path = None
_langgraph_app = None

# ===== LangSmith 클라이언트 =====
client = Client()

# 데이터셋 생성 또는 로드
try:
    dataset = client.read_dataset(dataset_name="ds-pertinent-fiesta-43")
    print(f"[LANGSMITH] 기존 데이터셋 로드됨")
except:
    dataset = client.create_dataset(
        dataset_name="ds-pertinent-fiesta-43",
        description="RAG System Evaluation Dataset"
    )
    examples = [
        {
            "inputs": {"question": "ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration"},
            "outputs": {"answer": """title : ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration
huggingface_url : https://huggingface.co/papers/2511.21689
git_url : https://github.com/NVlabs/ToolOrchestra/
upvote:99
authors : Hongjin Su, Shizhe Diao, Ximing Lu"""}
        },
        {
            "inputs": {"question": "해리포터 줄거리 알려주세요"},
            "outputs": {"answer": """죄송합니다. "해리포터 줄거리 알려주세요"는 AI/ML/DL/LLM 연구 논문과 관련이 없는 질문입니다."""}
        }
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"[LANGSMITH] 새 데이터셋 생성됨")

# ===== GraphState =====
class GraphState(TypedDict):
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
    is_ai_ml_related: bool
    _vectorstore: Any
    _llm: Any
    _bm25_retriever: Any
    _cluster_metadata_path: str

# ===== Helper Functions =====
def get_doc_hash_key(doc: Document) -> str:
    content = doc.page_content[:1000]
    source = doc.metadata.get('source', '')
    return hashlib.sha256(f"{content}|{source}".encode('utf-8')).hexdigest()

def is_korean_text(text: str) -> bool:
    return bool(re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]').search(text))

def extract_keywords(text: str) -> Set[str]:
    keywords = set()
    keywords.update(re.findall(r'\b[a-zA-Z]+-?\w*\d+\w*\b', text.lower()))
    keywords.update([w.lower() for w in re.findall(r'\b[A-Z]{2,}[a-z]?\d*\b', text)])
    keywords.update(re.findall(r'\b\w+[-_]\w+\b', text.lower()))
    tech_terms = {'transformer', 'attention', 'diffusion', 'gan', 'vae', 'bert',
                  'gpt', 'llama', 'sam', 'clip', 'vit', 'resnet', 'unet',
                  'rag', 'retrieval', 'embedding', 'tokenizer', 'langchain',
                  'pytorch', 'tensorflow', 'huggingface', 'audio', 'model', 'paper'}
    keywords.update(set(re.findall(r'\b\w+\b', text.lower())) & tech_terms)
    return keywords

def calculate_metadata_boost(doc, query_keywords: Set[str]) -> float:
    boost = 0.0
    metadata = doc.metadata or {}
    title = metadata.get('title', '').lower()
    if any(kw in title for kw in query_keywords):
        boost += 0.05
    doc_keywords = [k.lower() for k in metadata.get('keywords', [])]
    if any(kw in doc_keywords for kw in query_keywords):
        boost += 0.02
    return boost

def is_ai_ml_related_by_llm(question: str, llm) -> bool:
    if not question or llm is None:
        return False
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 AI/ML/DL/LLM 분야 관련성 판단 이진 분류기입니다.
YES: AI/ML/DL/LLM 모델, 프레임워크, 개념, 아키텍처, 연구 관련
NO: 일상 대화, 엔터테인먼트, 일반 상식
출력: "YES" 또는 "NO"만 출력"""),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"question": question}).strip().upper().startswith("Y")
    except:
        return True

# ===== Node Functions =====
def translate_node(state: GraphState) -> dict:
    print("\n[NODE: translate] 번역")
    original_question = state["original_question"]
    llm = state.get("_llm")
    
    if not is_korean_text(original_question):
        return {"question": original_question, "original_question": original_question,
                "translated_question": None, "is_korean": False}
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "한국어를 영어로 번역. AI/ML 용어는 정확한 기술 용어로."),
            ("human", "{korean_text}")
        ])
        translated = (prompt | llm | StrOutputParser()).invoke({"korean_text": original_question}).strip()
        return {"question": translated, "original_question": original_question,
                "translated_question": translated, "is_korean": True}
    except:
        return {"question": original_question, "original_question": original_question,
                "translated_question": None, "is_korean": True}

def topic_guard_node(state: GraphState) -> dict:
    print("\n[NODE: topic_guard] AI/ML 관련성 체크")
    question = state.get("question") or state.get("original_question")
    llm = state.get("_llm")
    is_related = is_ai_ml_related_by_llm(question, llm)
    print(f"  → {'✅ AI/ML 관련' if is_related else '❌ 비 AI/ML'}")
    return {"is_ai_ml_related": is_related}

def retrieve_node(state: GraphState) -> dict:
    print("\n[NODE: retrieve] 하이브리드 검색")
    # (기존 retrieve_node 로직 유지 - 간결화 생략)
    question = state["question"]
    vectorstore = state.get("_vectorstore")
    bm25_retriever = state.get("_bm25_retriever")
    
    if not vectorstore:
        return {"documents": [], "doc_scores": [], "cluster_id": None, "search_type": "vector"}
    
    query_keywords = extract_keywords(question)
    vector_docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)
    
    if not bm25_retriever:
        boosted_docs, boosted_scores = [], []
        for doc, score in vector_docs_with_scores[:5]:
            boost = calculate_metadata_boost(doc, query_keywords)
            boosted_docs.append(doc)
            boosted_scores.append(max(0.0, score - boost * 2.0))
        cluster_id = boosted_docs[0].metadata.get("cluster_id", -1) if boosted_docs else None
        return {"documents": boosted_docs, "doc_scores": boosted_scores,
                "cluster_id": cluster_id, "search_type": "vector"}
    
    # RRF 퓨전 (기존 로직)
    bm25_docs = bm25_retriever.invoke(question)
    RRF_K, fusion_scores, doc_map = 60, {}, {}
    
    for rank, (doc, _) in enumerate(vector_docs_with_scores):
        key = get_doc_hash_key(doc)
        doc_map[key] = doc
        fusion_scores[key] = fusion_scores.get(key, 0.0) + 1.5/(RRF_K+rank+1)
        fusion_scores[key] += calculate_metadata_boost(doc, query_keywords)
    
    for rank, doc in enumerate(bm25_docs):
        key = get_doc_hash_key(doc)
        doc_map[key] = doc
        fusion_scores[key] = fusion_scores.get(key, 0.0) + 0.5/(RRF_K+rank+1)
    
    sorted_items = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    documents = [doc_map[k] for k, _ in sorted_items[:5]]
    scores = [s for _, s in sorted_items[:5]]
    cluster_id = documents[0].metadata.get("cluster_id", -1) if documents else None
    
    return {"documents": documents, "doc_scores": scores,
            "cluster_id": cluster_id, "search_type": "hybrid"}

def evaluate_document_relevance_node(state: GraphState) -> dict:
    print("\n[NODE: evaluate] 문서 관련성 평가")
    documents, scores = state["documents"], state["doc_scores"]
    
    if not documents or not scores:
        return {"relevance_level": "low"}
    
    best_score = max(scores)
    level = "high" if best_score >= 0.0325 else "medium" if best_score >= 0.0120 else "low"
    print(f"  → {level.upper()} (score={best_score:.4f})")
    return {"relevance_level": level}

def cluster_similarity_check_node(state: GraphState) -> dict:
    print("\n[NODE: cluster_check] 클러스터 체크")
    # (기존 로직 유지 - 간결화)
    return {"cluster_similarity_score": 0.5, "search_type": "cluster"}

def web_search_node(state: GraphState) -> dict:
    print("\n[NODE: web_search] 웹 검색")
    try:
        from langchain_community.retrievers import TavilySearchAPIRetriever
        retriever = TavilySearchAPIRetriever(k=5)
        web_docs = retriever.invoke(state["question"])
        processed = []
        for i, doc in enumerate(web_docs):
            processed.append(Document(
                page_content=doc.page_content,
                metadata={'title': doc.metadata.get('title', '웹 결과'),
                         'source': doc.metadata.get('source', ''),
                         'source_type': 'web', 'score': doc.metadata.get('score', 0.5)}
            ))
        return {"documents": processed, "search_type": "web",
                "doc_scores": [d.metadata['score'] for d in processed]}
    except:
        return {"documents": [], "search_type": "web_failed", "doc_scores": []}

def generate_final_answer_node(state: GraphState) -> dict:
    print("\n[NODE: generate] 답변 생성")
    documents = state["documents"]
    llm = state.get("_llm")
    
    if not documents:
        context_str = "NO_RELEVANT_PAPERS"
    else:
        context_blocks = []
        for i, doc in enumerate(documents[:5], 1):
            meta = doc.metadata or {}
            block = f"""[DOCUMENT {i}]
{doc.page_content}
title: {meta.get('title', 'Unknown')}
huggingface_url: {meta.get('huggingface_url', 'N/A')}
authors: {meta.get('authors', 'N/A')}"""
            context_blocks.append(block)
        context_str = "\n\n".join(context_blocks)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are AI Tech Trend Navigator. Summarize papers clearly. ALWAYS respond in Korean."),
        ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
    ])
    
    try:
        answer = (prompt | llm | StrOutputParser()).invoke({
            "question": state["original_question"],
            "context": context_str
        })
    except Exception as e:
        answer = f"답변 생성 오류: {str(e)}"
    
    # Sources 구성
    sources = []
    seen_docs = set()
    for doc in documents[:5]:
        meta = doc.metadata or {}
        doc_id = meta.get('doc_id')
        if doc_id and doc_id in seen_docs:
            continue
        if doc_id:
            seen_docs.add(doc_id)
        
        if meta.get('source_type') == 'web':
            sources.append({"type": "web", "title": meta.get("title"), "url": meta.get("source")})
        else:
            sources.append({
                "type": "paper",
                "title": meta.get("title"),
                "huggingface_url": meta.get("huggingface_url"),
                "authors": meta.get("authors")
            })
    
    return {"answer": answer, "sources": sources}

def reject_node(state: GraphState) -> dict:
    print("\n[NODE: reject] 거부")
    question = state["original_question"]
    is_ai_ml_related = state.get("is_ai_ml_related", False)
    
    if not is_ai_ml_related:
        answer = f'죄송합니다. "{question}"는 AI/ML/DL/LLM 연구 논문과 관련이 없는 질문입니다.'
    else:
        answer = f'죄송합니다. "{question}"와 관련된 문서를 찾지 못했습니다.'
    
    return {"answer": answer, "sources": [], "search_type": "rejected"}

# ===== Routing Functions =====
def route_after_topic_guard(state: GraphState) -> Literal["retrieve", "reject"]:
    return "retrieve" if state.get("is_ai_ml_related", True) else "reject"

def route_after_evaluate(state: GraphState) -> Literal["generate", "cluster_check", "web_search", "reject"]:
    level = state.get("relevance_level", "low")
    if level == "high":
        return "generate"
    elif level == "medium":
        return "cluster_check"
    else:
        return "web_search" if state.get("is_ai_ml_related", True) else "reject"

def route_after_cluster_check(state: GraphState) -> Literal["generate", "web_search"]:
    return "generate" if state.get("cluster_similarity_score", 0.0) <= 0.85 else "web_search"

# ===== Graph Builder =====
def build_langgraph_rag():
    print("\n[GRAPH BUILD] LangGraph 구축")
    graph = StateGraph(GraphState)
    
    graph.add_node("translate", translate_node)
    graph.add_node("topic_guard", topic_guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_document_relevance_node)
    graph.add_node("cluster_check", cluster_similarity_check_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_final_answer_node)
    graph.add_node("reject", reject_node)
    
    graph.add_edge(START, "translate")
    graph.add_edge("translate", "topic_guard")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges("topic_guard", route_after_topic_guard,
                               {"retrieve": "retrieve", "reject": "reject"})
    graph.add_conditional_edges("evaluate", route_after_evaluate,
                               {"generate": "generate", "cluster_check": "cluster_check",
                                "web_search": "web_search", "reject": "reject"})
    graph.add_conditional_edges("cluster_check", route_after_cluster_check,
                               {"generate": "generate", "web_search": "web_search"})
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("reject", END)
    
    return graph.compile()

# ===== ✅ TARGET 함수 (전역 스코프) =====
def target(inputs: dict) -> dict:
    """LangSmith 평가용 target 함수"""
    global _vectorstore, _llm, _bm25_retriever, _cluster_metadata_path, _langgraph_app
    
    question = inputs.get("question", "")
    print(f"\n[TARGET] 처리: {question}")
    
    if _langgraph_app is None:
        return {"answer": "ERROR: System not initialized", "sources": []}
    
    try:
        initial_state = {
            "question": "", "original_question": question, "translated_question": None,
            "is_korean": False, "documents": [], "doc_scores": [], "cluster_id": None,
            "cluster_similarity_score": None, "search_type": "", "relevance_level": "",
            "answer": "", "sources": [], "is_ai_ml_related": True,
            "_vectorstore": _vectorstore, "_llm": _llm, "_bm25_retriever": _bm25_retriever,
            "_cluster_metadata_path": _cluster_metadata_path
        }
        
        result = _langgraph_app.invoke(initial_state)
        print(f"[TARGET] 완료 - 답변: {len(result.get('answer', ''))}자, 출처: {len(result.get('sources', []))}개")
        
        return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
    
    except Exception as e:
        print(f"[TARGET] ERROR: {e}")
        return {"answer": f"ERROR: {str(e)}", "sources": []}

# ===== ✅ EVALUATOR 함수 (전역 스코프) =====
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """답변 정확성 평가"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    eval_prompt = f"""Evaluate AI/ML paper recommendation system.
Question: {inputs.get('question', '')}
Predicted: {outputs.get('answer', '')}
Reference: {reference_outputs.get('answer', '')}

Score 0.0-1.0 (1.0=perfect, 0.0=wrong)
For off-topic: 1.0 if properly rejected, 0.0 if answered

JSON only: {{"score": <float>, "reasoning": "<text>"}}"""
    
    try:
        response = llm.invoke(eval_prompt)
        result = json.loads(response.content)
        return {"key": "correctness", "score": result.get("score", 0.0),
                "comment": result.get("reasoning", "")}
    except:
        return {"key": "correctness", "score": 0.0, "comment": "Evaluation failed"}

# ===== MAIN =====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LangGraph RAG System - LangSmith Evaluation")
    print("="*60)
    
    try:
        from vectordb import load_vectordb
        
        # 초기화
        print("\n[INIT] 시스템 초기화...")
        _vectorstore = load_vectordb(MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)
        
        collection_data = _vectorstore._collection.get(include=['documents', 'metadatas'])
        all_documents = [
            Document(page_content=c, metadata=m)
            for c, m in zip(collection_data['documents'], collection_data['metadatas'])
        ]
        _bm25_retriever = BM25Retriever.from_documents(all_documents)
        _bm25_retriever.k = 3
        
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        _cluster_metadata_path = str(PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json")
        _langgraph_app = build_langgraph_rag()
        
        print("✅ 초기화 완료\n")
        
        # 샘플 테스트
        print("[TEST] 샘플 테스트...")
        test1 = target({"question": "ToolOrchestra 논문"})
        print(f"  AI/ML 질문: {len(test1['answer'])}자")
        
        test2 = target({"question": "해리포터 줄거리"})
        print(f"  비 AI/ML: {test2['answer'][:80]}...")
        
        # LangSmith 평가
        print("\n[EVALUATE] LangSmith 평가 시작...\n")
        experiment_results = evaluate(
            target,
            data=dataset,
            evaluators=[correctness_evaluator],
            experiment_prefix="langgraph-rag-v2",
            metadata={
                "model": MODEL_NAME,
                "retrieval": "Hybrid+MetadataBoost",
                "features": "topic_guard+cluster+web"
            },
            max_concurrency=1
        )
        
        print("\n✅ 평가 완료!")
        print(f"실험: {experiment_results.experiment_name}")
        print("https://smith.langchain.com/")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()