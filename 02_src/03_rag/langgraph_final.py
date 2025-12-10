"""
LangGraph 기반 RAG 시스템 (최종 완성 버전 - Sources 포함)

전체 개선 사항:
1. ✅ TypedDict import 오류 수정
2. ✅ SAM3 문제 해결: 메타데이터 부스팅
3. ✅ langchain 문제 해결: 스마트 라우팅
4. ✅ "배아파, 뭐해?" 문제 해결: topic_guard_node
5. ✅ RRF/Cluster 임계값 최적화
6. ✅ Sources 구성: 허깅페이스 URL, 웹 검색 URL 표시
"""

# ===== SECTION 1: IMPORTS =====
import os
import sys
import json
import re
import warnings
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Set

# ✅ TypedDict import 수정
try:
    from typing import TypedDict
except ImportError:
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

MODEL_NAME = os.getenv("MODEL_NAME", "MiniLM-L6")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 45))

# ===== SECTION 3: GRAPHSTATE =====
class GraphState(TypedDict):
    """LangGraph 상태 관리"""
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

# ===== SECTION 4: HELPER FUNCTIONS =====
def get_doc_hash_key(doc: Document) -> str:
    content = doc.page_content[:1000]
    source = doc.metadata.get('source', '')
    data_to_hash = f"{content}|{source}"
    return hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

def is_korean_text(text: str) -> bool:
    korean_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
    return bool(korean_pattern.search(text))

def extract_keywords(text: str) -> Set[str]:
    """키워드 추출"""
    keywords = set()
    pattern1 = r'\b[a-zA-Z]+-?\w*\d+\w*\b'
    keywords.update(re.findall(pattern1, text.lower()))
    pattern2 = r'\b[A-Z]{2,}[a-z]?\d*\b'
    keywords.update([w.lower() for w in re.findall(pattern2, text)])
    pattern3 = r'\b\w+[-_]\w+\b'
    keywords.update(re.findall(pattern3, text.lower()))
    tech_terms = {
        'transformer', 'attention', 'diffusion', 'gan', 'vae', 'bert', 
        'gpt', 'llama', 'sam', 'clip', 'vit', 'resnet', 'unet',
        'rag', 'retrieval', 'embedding', 'tokenizer', 'langchain',
        'pytorch', 'tensorflow', 'huggingface'
    }
    words = set(re.findall(r'\b\w+\b', text.lower()))
    keywords.update(words & tech_terms)
    return keywords

def calculate_metadata_boost(doc, query_keywords: Set[str]) -> float:
    """메타데이터 부스팅"""
    boost = 0.0
    metadata = doc.metadata or {}
    title = metadata.get('title', '').lower()
    for keyword in query_keywords:
        if keyword in title:
            boost += 0.05
            break
    doc_keywords = metadata.get('keywords', [])
    if isinstance(doc_keywords, list):
        doc_keywords_lower = [k.lower() for k in doc_keywords]
        for keyword in query_keywords:
            if keyword in doc_keywords_lower:
                boost += 0.02
                break
    doc_id = metadata.get('doc_id', '').lower()
    for keyword in query_keywords:
        if keyword in doc_id:
            boost += 0.01
            break
    return boost

def is_ai_ml_related_by_llm(question: str, llm) -> bool:
    """LLM으로 AI/ML 관련성 판별"""
    if not question or llm is None:
        return False

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a classifier that determines whether a user's question relates to AI/ML/DL/LLM research, models, or tools.

Return "YES" if the question is about:
- AI/ML/DL/LLM models, architectures, algorithms
- AI/ML/DL/LLM concepts, tools, libraries
- AI/ML/DL/LLM applications, research papers

Return "NO" if unrelated to AI/ML/DL/LLM:
- Daily life, personal health
- Daily conversation
- Entertainment, sports, general knowledge

Output must be exactly: YES or NO."""),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip().upper()
        return result.startswith("Y")
    except Exception as e:
        print(f"[WARN] LLM topic classification failed: {e}")
        return True

# ===== SECTION 5: NODE FUNCTIONS =====
def translate_node(state: GraphState) -> dict:
    """노드 0: 한글 질문 번역"""
    print("\n" + "="*60)
    print("[NODE: translate] 질문 언어 확인 및 번역")
    print("="*60)

    original_question = state["original_question"]
    llm = state.get("_llm")
    has_korean = is_korean_text(original_question)

    if not has_korean:
        print(f"[translate] 영어 질문 - 번역 스킵")
        return {
            "question": original_question,
            "original_question": original_question,
            "translated_question": None,
            "is_korean": False
        }

    print(f"[translate] 한글 질문 - 영어로 번역 중...")
    try:
        translate_prompt = ChatPromptTemplate.from_messages([
            ("system", '''You are a professional translator related to AI/ML/DL/LLM. Translate Korean into processed English for your research. 
            For questions not related to AI/ML/DL/LLM, translate them into general language expressions without paraphrases or related terms.(Large Language Model).

            e.g.
            레그 -> Rag,rag
            래그 -> Rag,rag
            랭체인 -> Langchain
            랭그래프 -> Langgraph 
            Output ONLY English text.'''),
            ("human", "{korean_text}")
        ])
        chain = translate_prompt | llm | StrOutputParser()
        translated = chain.invoke({"korean_text": original_question}).strip()
        print(f"[translate] 번역 완료: {translated}")
        return {
            "question": translated,
            "original_question": original_question,
            "translated_question": translated,
            "is_korean": True
        }
    except Exception as e:
        print(f"[ERROR] 번역 실패: {e}")
        return {
            "question": original_question,
            "original_question": original_question,
            "translated_question": None,
            "is_korean": True
        }

def topic_guard_node(state: GraphState) -> dict:
    """✅ 노드 1: AI/ML/DL/LLM 관련성 사전 체크"""
    print("\n" + "="*60)
    print("[NODE: topic_guard] AI/ML/DL/LLM 관련성 사전 체크")
    print("="*60)
    
    original_q = state.get("original_question", "")
    question = state.get("question", "")
    llm = state.get("_llm")
    
    print(f"[topic_guard] 질문: {original_q}")
    
    try:
        is_related = is_ai_ml_related_by_llm(question if question else original_q, llm)
        if is_related:
            print(f"[topic_guard] ✅ AI/ML/DL/LLM 관련 → 검색 진행")
            return {"is_ai_ml_related": True}
        else:
            print(f"[topic_guard] ❌ 비 AI/ML/DL/LLM → 검색 스킵")
            return {"is_ai_ml_related": False}
    except Exception as e:
        print(f"[WARN] 토픽 가드 실패: {e}")
        return {"is_ai_ml_related": True}

def retrieve_node(state: GraphState) -> dict:
    """✅ 노드 2: 하이브리드 검색 + 메타데이터 부스팅"""
    print("\n" + "="*60)
    print("[NODE: retrieve] 하이브리드 검색")
    print("="*60)

    question = state["question"]
    vectorstore = state.get("_vectorstore")
    bm25_retriever = state.get("_bm25_retriever")

    if vectorstore is None:
        print("[ERROR] VectorStore 없음")
        return {"documents": [], "doc_scores": [], "cluster_id": None, "search_type": "vector"}

    query_keywords = extract_keywords(question)
    if query_keywords:
        print(f"[retrieve] 키워드: {query_keywords}")

    use_bm25 = bm25_retriever is not None

    try:
        vector_docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)
        vector_docs = [doc for doc, score in vector_docs_with_scores]
        print(f"[retrieve] 벡터 검색: {len(vector_docs)}개")

        if not use_bm25:
            print("[retrieve] BM25 없음 → 벡터만 사용")
            boosted_docs = []
            boosted_scores = []
            for doc, score in vector_docs_with_scores[:5]:
                boost = calculate_metadata_boost(doc, query_keywords)
                adjusted_score = max(0.0, score - boost * 2.0)
                boosted_docs.append(doc)
                boosted_scores.append(adjusted_score)
                if boost > 0:
                    print(f"[retrieve] 부스팅: {doc.metadata.get('title', 'N/A')[:50]} (+{boost:.3f})")

            cluster_id = boosted_docs[0].metadata.get("cluster_id", -1) if boosted_docs else None
            return {
                "documents": boosted_docs,
                "doc_scores": boosted_scores,
                "cluster_id": cluster_id,
                "search_type": "vector"
            }

        bm25_docs = bm25_retriever.invoke(question)
        print(f"[retrieve] BM25 검색: {len(bm25_docs)}개")

        RRF_K = 60
        fusion_scores = {}
        doc_map = {}
        metadata_boosts = {}

        for rank, (doc, _score) in enumerate(vector_docs_with_scores):
            doc_key = get_doc_hash_key(doc)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            score = 1 / (RRF_K + rank + 1)
            fusion_scores[doc_key] = fusion_scores.get(doc_key, 0.0) + score
            if doc_key not in metadata_boosts:
                metadata_boosts[doc_key] = calculate_metadata_boost(doc, query_keywords)

        for rank, doc in enumerate(bm25_docs):
            doc_key = get_doc_hash_key(doc)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            score = 1 / (RRF_K + rank + 1)
            fusion_scores[doc_key] = fusion_scores.get(doc_key, 0.0) + score
            if doc_key not in metadata_boosts:
                metadata_boosts[doc_key] = calculate_metadata_boost(doc, query_keywords)

        for doc_key in fusion_scores:
            boost = metadata_boosts.get(doc_key, 0.0)
            if boost > 0:
                original_score = fusion_scores[doc_key]
                fusion_scores[doc_key] += boost
                doc = doc_map[doc_key]
                print(f"[retrieve] 부스팅: {doc.metadata.get('title', 'N/A')[:50]}")
                print(f"           {original_score:.4f} → {fusion_scores[doc_key]:.4f} (+{boost:.4f})")

        sorted_items = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        documents = []
        scores = []
        for doc_key, score in sorted_items[:5]:
            documents.append(doc_map[doc_key])
            scores.append(score)

        if not documents:
            print("[retrieve] 문서 없음")
            return {"documents": [], "doc_scores": [], "cluster_id": None, "search_type": "hybrid"}

        cluster_id = documents[0].metadata.get("cluster_id", -1)
        print(f"[retrieve] 완료: {len(documents)}개, 최상위={scores[0]:.4f}, Cluster={cluster_id}")

        return {
            "documents": documents,
            "doc_scores": scores,
            "cluster_id": cluster_id,
            "search_type": "hybrid"
        }

    except Exception as e:
        print(f"[ERROR] 검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"documents": [], "doc_scores": [], "cluster_id": None, "search_type": "hybrid"}

def evaluate_document_relevance_node(state: GraphState) -> dict:
    """✅ 노드 3: 문서 관련성 평가"""
    print("\n" + "="*60)
    print("[NODE: evaluate] 문서 관련성 평가")
    print("="*60)

    original_q = state.get("original_question", "")
    question = state.get("question", "")
    documents = state["documents"]
    scores = state["doc_scores"]
    is_ai_ml_related = state.get("is_ai_ml_related", True)
    
    print(f"[evaluate] AI/ML/DL/LLM 관련성: {is_ai_ml_related}")

    query_keywords = extract_keywords(question)
    if not query_keywords:
        query_keywords = extract_keywords(original_q)

    if documents and query_keywords:
        for doc in documents[:3]:
            metadata = doc.metadata or {}
            title = metadata.get('title', '').lower()
            doc_keywords = [k.lower() for k in metadata.get('keywords', [])]
            for keyword in query_keywords:
                if keyword in title or keyword in ' '.join(doc_keywords):
                    print(f"[evaluate] ✅ 메타데이터 매칭: '{keyword}'")
                    return {"relevance_level": "high", "is_ai_ml_related": is_ai_ml_related}

    if not documents or not scores:
        print("[evaluate] 문서 없음 → LOW")
        return {"relevance_level": "low", "is_ai_ml_related": is_ai_ml_related}

    best_score = max(scores)
    if best_score >= 0.0325:
        level = "high"
    elif best_score >= 0.0120:
        level = "medium"
    else:
        level = "low"

    print(f"[evaluate] RRF={best_score:.6f} → {level.upper()}")
    return {"relevance_level": level, "is_ai_ml_related": is_ai_ml_related}

def cluster_similarity_check_node(state: GraphState) -> dict:
    """노드 4: 클러스터 유사도 체크"""
    print("\n" + "="*60)
    print("[NODE: cluster_check] 클러스터 체크")
    print("="*60)

    cluster_id = state["cluster_id"]
    question = state["question"]
    vectorstore = state.get("_vectorstore")
    cluster_metadata_path = state.get("_cluster_metadata_path")

    if cluster_id is None or cluster_id == -1:
        print("[cluster_check] cluster_id 없음")
        return {"cluster_similarity_score": 0.0, "search_type": "cluster"}

    try:
        with open(cluster_metadata_path, 'r', encoding='utf-8') as f:
            cluster_meta = json.load(f)

        cluster_info = cluster_meta["clusters"].get(str(cluster_id))
        if not cluster_info:
            print(f"[cluster_check] 정보 없음")
            return {"cluster_similarity_score": 0.0, "search_type": "cluster"}

        cluster_density = cluster_info.get("density", 0.0)
        print(f"[cluster_check] Cluster {cluster_id} 밀도: {cluster_density:.3f}")

        all_docs = vectorstore.similarity_search_with_score(question, k=20)
        filtered_docs = [
            (doc, score) for doc, score in all_docs
            if doc.metadata.get("cluster_id") == cluster_id
        ][:5]

        if not filtered_docs:
            print("[cluster_check] 추가 문서 없음")
            return {"cluster_similarity_score": 0.0, "search_type": "cluster"}

        avg_score = sum(score for doc, score in filtered_docs) / len(filtered_docs)
        print(f"[cluster_check] {len(filtered_docs)}개, 평균={avg_score:.4f}")

        existing_docs = state["documents"]
        additional_docs = [doc for doc, score in filtered_docs[:3]]
        existing_contents = {doc.page_content for doc in existing_docs}
        unique_additional = [doc for doc in additional_docs if doc.page_content not in existing_contents]
        merged_docs = existing_docs + unique_additional

        print(f"[cluster_check] 추가 {len(unique_additional)}개 병합")

        return {
            "cluster_similarity_score": avg_score,
            "documents": merged_docs,
            "search_type": "cluster"
        }

    except Exception as e:
        print(f"[ERROR] 클러스터 체크 오류: {e}")
        return {"cluster_similarity_score": 0.0, "search_type": "cluster"}

def web_search_node(state: GraphState) -> dict:
    """노드 5: 웹 검색"""
    print("\n" + "="*60)
    print("[NODE: web_search] 웹 검색 시작")
    print("="*60)

    question = state["question"]

    try:
        from langchain_community.retrievers import TavilySearchAPIRetriever
        print("[web_search] Tavily API 사용")

        retriever = TavilySearchAPIRetriever(k=5)
        web_docs_raw = retriever.invoke(question)
        print(f"[web_search] Tavily: {len(web_docs_raw)}개")

        processed_web_docs = []
        for i, doc in enumerate(web_docs_raw):
            original_meta = doc.metadata
            title = original_meta.get('title', '웹 검색 결과')
            source_url = original_meta.get('source', '')
            score = original_meta.get('score', 0.5)
            
            web_doc = Document(
                page_content=doc.page_content,
                metadata={
                    'title': title,
                    'source': source_url,
                    'source_type': 'web',
                    'score': score,
                    'index': i
                }
            )
            processed_web_docs.append(web_doc)
            print(f"  [{i+1}] {title[:60]}")

        return {
            "documents": processed_web_docs,
            "search_type": "web",
            "doc_scores": [doc.metadata['score'] for doc in processed_web_docs]
        }

    except Exception as e:
        print(f"[web_search] Tavily 실패: {e}")
        return {
            "documents": [Document(
                page_content="웹 검색 실패",
                metadata={"source": "system", "source_type": "error"}
            )],
            "search_type": "web_failed",
            "doc_scores": [2.0]
        }

def generate_final_answer_node(state: GraphState) -> dict:
    """✅ 노드 6: 최종 답변 생성 + Sources 구성"""
    print("\n" + "=" * 60)
    print("[NODE: generate] 최종 답변 생성")
    print("=" * 60)

    original_question = state["original_question"]
    documents = state["documents"]
    search_type = state.get("search_type", "internal")
    is_korean = state.get("is_korean", False)
    llm = state.get("_llm")

    if is_korean:
        print(f"[generate] 원본(한글): {original_question}")
        print(f"[generate] 검색(영어): {state.get('question')}")

    # 1) CONTEXT 블록
    if not documents:
        context_str = "NO_RELEVANT_PAPERS"
    else:
        context_blocks = []
        for i, doc in enumerate(documents[:5], 1):
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
{doc.page_content}

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

    # 2) 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are "AI Tech Trend Navigator", an expert for AI/ML/DL/LLM papers.
Summarize papers clearly. Explain in simple terms. Highlight practical use-cases.
Rely only on given context. Do not invent details.
ALWAYS respond in Korean."""),
        ("human", """
[QUESTION]
{question}

[CONTEXT]
{context}

Answer structure:
1) One-line summary
2) Key insights (max 3 bullets)
3) Detailed explanation

⚠ Do not hallucinate. ALWAYS respond in Korean. No bold/italics.
""")
    ])

    # 3) 체인 실행
    try:
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": original_question, "context": context_str})
        print("[generate] 답변 생성 완료")
    except Exception as e:
        print(f"[ERROR] 답변 생성 오류: {e}")
        answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    # ✅ 4) SOURCES 구성
        # 4) sources 구성 - 중복 제거 포함
    seen_docs = set()
    sources: List[Dict[str, Any]] = []
    
    for doc in documents[:5]:  # 답변에 사용된 문서들
        meta = doc.metadata or {}
        doc_id = meta.get('doc_id')
        
        # doc_id가 있는 경우 중복 체크
        if doc_id and doc_id in seen_docs:
            print(f"[generate] 중복 문서 스킵: doc_id={doc_id}")
            continue
        
        # doc_id 기록
        if doc_id:
            seen_docs.add(doc_id)
        
        # 웹 검색 결과인 경우
        if meta.get('source_type') == 'web':
            sources.append({
                "type": "web",
                "title": meta.get("title", "웹 검색 결과"),
                "url": meta.get("source", ""),
                "score": meta.get("score", 0.5)
            })
            print(f"[generate] 웹 출처 추가: {meta.get('title', 'Unknown')[:50]}")
        # 논문 문서인 경우
        else:
            title = meta.get("title", meta.get("paper_name", "Unknown"))
            hf_url = meta.get("huggingface_url", meta.get("source", ""))
            gh_url = meta.get("github_url", "")
            authors = meta.get("authors", "Unknown")
            year = meta.get("publication_year", "Unknown")
            upvote = meta.get("upvote", 0)
            doc_id = meta.get("doc_id", "")
            
            sources.append({
                "type": "paper",
                "title": title,
                "huggingface_url": hf_url,
                "github_url": gh_url,
                "authors": authors,
                "year": year,
                "upvote": upvote,
                "doc_id": doc_id
            })
            print(f"[generate] 논문 출처 추가: {meta.get('title', 'Unknown')[:50]}")
    
    print(f"[generate] 총 {len(sources)}개 고유 출처 추가됨")

    return {
        "answer": answer,
        "sources": sources
    }
    # sources: List[Dict[str, Any]] = []
    
    # if documents:
    #     print(f"[generate] Sources 구성 중... ({len(documents)}개 문서)")
        
    #     for i, doc in enumerate(documents[:5]):  # 최대 5개
    #         meta = doc.metadata or {}
    #         source_type = meta.get("source_type", "paper")
            
    #         if source_type == "web":
    #             # 웹 검색 결과
    #             sources.append({
    #                 "type": "web",
    #                 "title": meta.get("title", "웹 검색 결과"),
    #                 "url": meta.get("source", ""),
    #                 "score": meta.get("score", 0.5)
    #             })
    #             print(f"  [Source {i+1}] 웹: {meta.get('title', 'N/A')[:50]}")
    #         else:
    #             # 내부 논문
    #             title = meta.get("title", meta.get("paper_name", "Unknown"))
    #             hf_url = meta.get("huggingface_url", meta.get("source", ""))
    #             gh_url = meta.get("github_url", "")
    #             authors = meta.get("authors", "Unknown")
    #             year = meta.get("publication_year", "Unknown")
    #             upvote = meta.get("upvote", 0)
    #             doc_id = meta.get("doc_id", "")
                
    #             sources.append({
    #                 "type": "paper",
    #                 "title": title,
    #                 "huggingface_url": hf_url,
    #                 "github_url": gh_url,
    #                 "authors": authors,
    #                 "year": year,
    #                 "upvote": upvote,
    #                 "doc_id": doc_id
    #             })
    #             print(f"  [Source {i+1}] 논문: {title[:50]}")
    
    # print(f"[generate] 최종 Sources 개수: {len(sources)}")

    # ✅ 답변 텍스트에는 출처를 추가하지 않음 (UI 참조자료 섹션에만 표시)
    # return {
    #     "answer": answer,
    #     "sources": sources
    # }

def reject_node(state: GraphState) -> dict:
    """✅ 노드 7: 거부 응답"""
    print("\n" + "="*60)
    print("[NODE: reject] 질문 거부")
    print("="*60)

    question = state["original_question"]
    is_ai_ml_related = state.get("is_ai_ml_related", False)

    if not is_ai_ml_related:
        answer = f"""죄송합니다. "{question}"는 AI/ML 연구 논문과 관련이 없는 질문입니다.

이 시스템은 다음과 같은 질문에 답변할 수 있습니다:
• AI/ML 모델 및 아키텍처 (GPT, BERT, SAM, Diffusion 등)
• AI/ML 개념 및 기술 (Transformer, Attention, Fine-tuning 등)
• AI/ML 도구 및 라이브러리 (LangChain, Transformers, PyTorch 등)
• 최근 AI/ML 연구 동향 및 논문

AI/ML 관련 질문으로 다시 시도해주세요! 🤗"""
        print(f"[reject] 비 AI/ML 질문 거부")
    else:
        answer = f"""죄송합니다. '{question}'와 관련된 적절한 문서를 찾지 못했습니다.

다음과 같이 시도해보세요:
1. 더 구체적인 키워드 사용 (예: "transformer", "attention mechanism")
2. 영어 학술 용어 사용
3. 질문을 다시 표현해보기
4. 최근 발표된 논문 키워드로 검색

이 시스템은 HuggingFace에 게시된 최근 10주간의 AI/ML 연구 논문을 기반으로 답변합니다."""
        print(f"[reject] AI/ML 관련이지만 문서 없음")

    return {"answer": answer, "sources": [], "search_type": "rejected"}

# ===== SECTION 6: CONDITIONAL EDGE FUNCTIONS =====
def route_after_topic_guard(state: GraphState) -> Literal["retrieve", "reject"]:
    is_ai_ml_related = state.get("is_ai_ml_related", True)
    print(f"\n[ROUTING] topic_guard → ", end="")
    if is_ai_ml_related:
        print("retrieve")
        return "retrieve"
    else:
        print("reject")
        return "reject"

def route_after_evaluate(state: GraphState) -> Literal["generate", "cluster_check", "web_search", "reject"]:
    level = state.get("relevance_level", "low")
    documents = state.get("documents", [])
    is_ai_ml_related = state.get("is_ai_ml_related", True)

    print(f"\n[ROUTING] evaluate → {level.upper()} (AI/ML: {is_ai_ml_related})", end=" → ")

    if level == "high":
        print("generate")
        return "generate"
    elif level == "medium":
        print("cluster_check")
        return "cluster_check"
    else:
        if is_ai_ml_related:
            if documents and len(documents) > 0:
                print("cluster_check")
                return "cluster_check"
            else:
                print("web_search")
                return "web_search"
        else:
            print("reject")
            return "reject"

def route_after_cluster_check(state: GraphState) -> Literal["generate", "web_search"]:
    cluster_score = state.get("cluster_similarity_score", 0.0)
    cluster_id = state.get("cluster_id", -1)
    cluster_metadata_path = state.get("_cluster_metadata_path")

    print(f"\n[ROUTING] cluster_check → ", end="")

    try:
        with open(cluster_metadata_path, 'r', encoding='utf-8') as f:
            cluster_meta = json.load(f)
        cluster_info = cluster_meta["clusters"].get(str(cluster_id), {})
        density = cluster_info.get("density", 0.0)

        if cluster_score <= 0.85 and density >= 1.572:
            print(f"HIGH (score={cluster_score:.3f}, density={density:.3f}) → generate")
            return "generate"
        else:
            print(f"LOW (score={cluster_score:.3f}, density={density:.3f}) → web_search")
            return "web_search"
    except Exception as e:
        print(f"ERROR ({e}) → web_search")
        return "web_search"

# ===== SECTION 7: GRAPH BUILDER =====
def build_langgraph_rag():
    print("\n" + "="*60)
    print("[GRAPH BUILD] LangGraph 구축")
    print("="*60)

    graph = StateGraph(GraphState)

    graph.add_node("translate", translate_node)
    graph.add_node("topic_guard", topic_guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_document_relevance_node)
    graph.add_node("cluster_check", cluster_similarity_check_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_final_answer_node)
    graph.add_node("reject", reject_node)

    print("[GRAPH] 8개 노드 추가 완료")

    graph.add_edge(START, "translate")
    graph.add_edge("translate", "topic_guard")
    graph.add_edge("retrieve", "evaluate")

    graph.add_conditional_edges("topic_guard", route_after_topic_guard, {"retrieve": "retrieve", "reject": "reject"})
    graph.add_conditional_edges("evaluate", route_after_evaluate, {"generate": "generate", "cluster_check": "web_search", "web_search": "web_search", "reject": "reject"})
    graph.add_conditional_edges("cluster_check", route_after_cluster_check, {"generate": "generate", "web_search": "web_search"})

    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("reject", END)

    print("[GRAPH] 엣지 추가 완료")

    compiled_graph = graph.compile()
    print("[GRAPH] 컴파일 완료")

    return compiled_graph

# ===== SECTION 8: INTERACTIVE MODE =====
def run_interactive_mode(vectorstore, llm, bm25_retriever, cluster_metadata_path, langgraph_app):
    print("\n" + "="*60)
    print("대화형 모드 시작")
    print("="*60)

    while True:
        try:
            question = input("\n[질문] >> ").strip()
            if question.lower() in ['quit', 'exit', 'q', '종료']:
                print("\n종료합니다.")
                break
            if not question:
                continue

            initial_state = {
                "question": "", "original_question": question, "translated_question": None, "is_korean": False,
                "documents": [], "doc_scores": [], "cluster_id": None, "cluster_similarity_score": None,
                "search_type": "", "relevance_level": "", "answer": "", "sources": [], "is_ai_ml_related": True,
                "_vectorstore": vectorstore, "_llm": llm, "_bm25_retriever": bm25_retriever,
                "_cluster_metadata_path": cluster_metadata_path
            }

            result = langgraph_app.invoke(initial_state)

            print("\n" + "="*60)
            print("[최종 결과]")
            print("="*60)
            print(f"\n[답변]\n{result['answer']}")
            print(f"\n[출처] {len(result.get('sources', []))}개")
            for i, src in enumerate(result.get('sources', [])[:3], 1):
                if src.get('type') == 'web':
                    print(f"  {i}. {src.get('title')} (웹)")
                    print(f"     {src.get('url')}")
                else:
                    print(f"  {i}. {src.get('title')}")
                    print(f"     HF: {src.get('huggingface_url')}")
            print("\n" + "-"*60)

        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

# ===== SECTION 9: EXTERNAL API FUNCTIONS =====
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
    global _vectorstore, _llm, _bm25_retriever, _cluster_metadata_path, _langgraph_app

    try:
        print("\n[INIT] 초기화 중...")
        if model_name is None:
            model_name = MODEL_NAME
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = CHUNK_OVERLAP

        from vectordb import load_vectordb
        print(f"[LOADING] VectorStore: {model_name}")
        _vectorstore = load_vectordb(model_name, chunk_size, chunk_overlap)

        print("[LOADING] BM25 Retriever")
        collection_data = _vectorstore._collection.get(include=['documents', 'metadatas'])
        all_documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(collection_data['documents'], collection_data['metadatas'])
        ]
        if not all_documents:
            raise ValueError('문서 없음')

        _bm25_retriever = BM25Retriever.from_documents(all_documents)
        _bm25_retriever.k = 3
        print(f"[SUCCESS] BM25: {len(all_documents)}개")

        print(f"[LOADING] LLM: {llm_model}")
        _llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

        _cluster_metadata_path = str(PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json")

        print("[LOADING] LangGraph 컴파일")
        _langgraph_app = build_langgraph_rag()

        print("[SUCCESS] 초기화 완료!\n")

        return {'status': 'success', 'message': 'Initialized successfully'}

    except Exception as e:
        print(f"[ERROR] 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}

def ask_question(question: str, verbose: bool = False) -> dict:
    global _vectorstore, _llm, _bm25_retriever, _cluster_metadata_path, _langgraph_app

    if _langgraph_app is None:
        return {'success': False, 'error': 'Not initialized'}

    try:
        if verbose:
            print(f"\n[QUESTION] {question}")

        initial_state = {
            "question": "", "original_question": question, "translated_question": None, "is_korean": False,
            "documents": [], "doc_scores": [], "cluster_id": None, "cluster_similarity_score": None,
            "search_type": "", "relevance_level": "", "answer": "", "sources": [], "is_ai_ml_related": True,
            "_vectorstore": _vectorstore, "_llm": _llm, "_bm25_retriever": _bm25_retriever,
            "_cluster_metadata_path": _cluster_metadata_path
        }

        result = _langgraph_app.invoke(initial_state)

        if verbose:
            print(f"[ANSWER] 완료")
            print(f"[SOURCES] {len(result.get('sources', []))}개")

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
                'translated_question': result.get('translated_question'),
                'is_ai_ml_related': result.get('is_ai_ml_related', True)
            }
        }

    except Exception as e:
        print(f"[ERROR] 질문 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def get_system_status() -> dict:
    return {
        'initialized': _langgraph_app is not None,
        'vectorstore_loaded': _vectorstore is not None,
        'llm_loaded': _llm is not None,
        'bm25_retriever_loaded': _bm25_retriever is not None,
        'cluster_metadata_loaded': _cluster_metadata_path is not None
    }

# ===== SECTION 10: MAIN =====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LangGraph RAG System")
    print("="*60)

    try:
        from vectordb import load_vectordb

        print("[LOADING] VectorStore")
        vectorstore = load_vectordb(MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)

        print("[LOADING] BM25")
        collection_data = vectorstore._collection.get(include=['documents', 'metadatas'])
        all_documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(collection_data['documents'], collection_data['metadatas'])
        ]
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 3
        print(f"[SUCCESS] BM25: {len(all_documents)}개")

        print("[LOADING] LLM")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        cluster_metadata_path = str(PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json")

        print("[LOADING] LangGraph")
        langgraph_app = build_langgraph_rag()

        print("\n[SUCCESS] 모든 리소스 로드 완료!\n")

        run_interactive_mode(vectorstore, llm, bm25_retriever, cluster_metadata_path, langgraph_app)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()