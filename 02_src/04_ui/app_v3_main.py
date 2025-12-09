"""
FastAPI RAG Chatbot Server

이 서버는 HTML 프론트엔드와 LangGraph RAG 시스템(langgraph_test.py)을 연결합니다.
- 통계 정보 제공
- 트렌드 키워드 제공
- LangGraph 기반 채팅 응답 (내부 검색 + 웹 검색)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ★ 변경: LangChain LLM (langgraph_test에서 사용하는 것과 동일 모델)
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# ============================================================================
# 경로 설정
# ============================================================================

# main.py 위치: 02_src/04_ui/main.py
CURRENT_DIR = Path(__file__).parent  # 02_src/04_ui
SRC_DIR = CURRENT_DIR.parent  # 02_src
PROJECT_ROOT = SRC_DIR.parent  # project root

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "01_data"
CLUSTERS_DIR = DATA_DIR / "clusters"

# 모듈 경로
RAG_PATH = SRC_DIR / "03_rag"      # langgraph_test.py 위치
UTILS_PATH = SRC_DIR / "02_utils"  # vectordb.py 위치

# sys.path에 추가
import sys
sys.path.insert(0, str(RAG_PATH))
sys.path.insert(0, str(UTILS_PATH))

# ============================================================================
# FastAPI 앱 생성
# ============================================================================

app = FastAPI(
    title="HuggingFace Papers RAG API",
    description="LangGraph 기반 RAG 챗봇 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 전역 변수
# ============================================================================

# LangGraph 앱을 저장할 전역 변수
rag_application: Optional[object] = None

# ★ 변경: langgraph_test용 전역 리소스
vectorstore = None
llm = None
cluster_metadata_path: Optional[str] = None


# ============================================================================
# Pydantic 모델
# ============================================================================

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str


# ============================================================================
# 서버 시작/종료 이벤트
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 LangGraph RAG 시스템 로드"""
    global rag_application, vectorstore, llm, cluster_metadata_path
    
    print("\n" + "=" * 70)
    print("🚀 HuggingFace Papers RAG Server - Starting Up (langgraph_test 버전)")
    print("=" * 70)
    
    try:
        # 1. 경로 확인
        print(f"\n[INFO] 프로젝트 루트: {PROJECT_ROOT}")
        print(f"[INFO] RAG 경로: {RAG_PATH}")
        print(f"[INFO] langgraph_test.py 존재: {(RAG_PATH / 'langgraph_test.py').exists()}")

        # 2. langgraph_test 모듈 임포트
        print("\n[STEP 1/4] langgraph_test 모듈 임포트 중...")
        try:
            # ★ 변경: lg_grade 대신 langgraph_test 사용
            from langgraph_test import (
                build_langgraph_rag,
                MODEL_NAME,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )
            from vectordb import load_vectordb
            print("✅ langgraph_test 모듈 임포트 성공")
        except ImportError as e:
            print(f"❌ 임포트 실패: {e}")
            print(f"[DEBUG] sys.path: {sys.path[:8]}")
            raise

        # 3. VectorStore / LLM / Cluster metadata 초기화
        print("\n[STEP 2/4] VectorStore / LLM / Cluster 메타데이터 초기화 중...")

        # VectorStore 로드 (langgraph_test의 __main__과 동일 로직)
        print(f"[LOAD] VectorStore 로딩 중... (MODEL_NAME={MODEL_NAME}, "
              f"CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP})")
        vectorstore = load_vectordb(MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)
        print("[SUCCESS] VectorStore 로딩 완료")

        # LLM 초기화 (langgraph_test와 동일 모델)
        print("[LOAD] LLM 초기화 중...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("[SUCCESS] LLM 초기화 완료")

        # Cluster metadata 경로 (langgraph_test의 PROJECT_ROOT 기준과 동일)
        cluster_metadata_path = str(
            PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json"
        )
        print(f"[INFO] Cluster metadata path: {cluster_metadata_path}")

        # 4. LangGraph 앱 컴파일
        print("\n[STEP 3/4] LangGraph 그래프 컴파일 중...")
        rag_application = build_langgraph_rag()
        print(f"✅ LangGraph 앱 생성 완료 (타입: {type(rag_application).__name__})")

        # 5. 완료
        print("\n[STEP 4/4] 초기화 완료")
        print("\n" + "=" * 70)
        print("✅ RAG 서버 준비 완료! (langgraph_test)")
        print("📡 API 문서: http://localhost:8000/docs")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ [FAILED] 서버 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/")
async def root():
    """루트 경로 - 서버 상태 확인"""
    return {
        "status": "running",
        "message": "HuggingFace Papers RAG API Server (langgraph_test)",
        "rag_loaded": rag_application is not None,
        "endpoints": {
            "stats": "/api/stats",
            "trending_keywords": "/api/trending-keywords",
            "chat": "/api/chat",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy" if rag_application is not None else "initializing",
        "rag_loaded": rag_application is not None
    }


@app.get("/api/stats")
async def get_stats() -> Dict:
    """
    통계 정보 반환
    - 논문 개수
    - 키워드 개수
    - 사용된 주차 수
    """
    try:
        # cluster_assignments.json에서 문서 개수 확인
        assignments_path = CLUSTERS_DIR / "cluster_assignments.json"
        
        if not assignments_path.exists():
            # 파일이 없으면 기본값 반환
            return {
                "paper_count": 0,
                "keyword_count": 0,
                "weeks_used": 0
            }
        
        with open(assignments_path, "r", encoding="utf-8") as f:
            assignments_data = json.load(f)
        
        paper_count = assignments_data.get("_metadata", {}).get("n_documents", 0)
        weeks_used = len(assignments_data.get("_metadata", {}).get("weeks_used", []))
        
        # cluster_metadata.json에서 키워드 개수 확인
        metadata_path = CLUSTERS_DIR / "cluster_metadata.json"
        
        if not metadata_path.exists():
            keyword_count = 0
        else:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # 모든 클러스터의 고유 키워드 수집
            all_keywords = set()
            for cluster_id, info in metadata.get("clusters", {}).items():
                all_keywords.update(info.get("keywords", []))
            
            keyword_count = len(all_keywords)
        
        return {
            "paper_count": paper_count,
            "keyword_count": keyword_count,
            "weeks_used": weeks_used
        }
    
    except Exception as e:
        print(f"[ERROR] 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trending-keywords")
async def get_trending_keywords(top_n: int = 7) -> Dict:
    """
    트렌드 키워드 반환
    - 모든 클러스터의 키워드를 수집하여 빈도수 기준 상위 N개 반환
    """
    try:
        metadata_path = CLUSTERS_DIR / "cluster_metadata.json"
        
        if not metadata_path.exists():
            # 파일이 없으면 기본 키워드 반환
            return {
                "keywords": ["LLM", "Transformer", "RAG", "Vision", "Diffusion", "Agent", "Multimodal"]
            }
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # 모든 클러스터의 키워드 수집
        all_keywords = []
        for cluster_id, info in metadata.get("clusters", {}).items():
            all_keywords.extend(info.get("keywords", []))
        
        # 빈도수 계산 및 상위 N개 추출
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(top_n)]
        
        # 부족하면 기본 키워드로 채우기
        default_keywords = ["LLM", "Transformer", "RAG", "Vision", "Diffusion", "Agent", "Multimodal"]
        for kw in default_keywords:
            if kw not in top_keywords and len(top_keywords) < top_n:
                top_keywords.append(kw)
        
        return {
            "keywords": top_keywords[:top_n]
        }
    
    except Exception as e:
        print(f"[ERROR] 트렌드 키워드 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict:
    """
    LangGraph RAG 기반 채팅 응답 생성 (langgraph_test 사용)
    1. LangGraph 앱 실행 (번역 → 내부 검색 → 문서 평가 → 클러스터 체크 → 웹 검색/생성)
    2. 최종 답변 및 검색 타입, 참조 문서 반환
    """
    # RAG 시스템 확인
    if rag_application is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    
    try:
        query = request.message
        print(f"\n{'='*70}")
        print(f"📝 [QUERY] {query}")
        print(f"{'='*70}")
        
        # ★ 변경: langgraph_test의 GraphState 구조에 맞는 초기 상태 정의
        initial_state = {
            # 질문 관련
            "original_question": query,     # 사용자가 입력한 원본 질문
            "question": query,              # 초기에는 동일, translate 노드에서 갱신
            "translated_question": None,
            "is_korean": False,

            # 검색/클러스터 관련
            "documents": [],
            "doc_scores": [],
            "cluster_id": None,
            "cluster_similarity_score": None,
            "search_type": "internal",
            "relevance_level": "",

            # 출력 관련
            "answer": "",
            "sources": [],

            # 내부 리소스 주입
            "_vectorstore": vectorstore,
            "_llm": llm,
            "_cluster_metadata_path": cluster_metadata_path,
        }
        
        # LangGraph 앱 실행
        print("[LANGGRAPH] RAG 파이프라인 실행 중... (langgraph_test)")
        result = rag_application.invoke(initial_state)
        
        # 결과 추출
        answer = result.get("answer", "답변 생성 실패")
        search_type = result.get("search_type", "unknown")
        documents = result.get("documents", [])

        print(f"✅ [SUCCESS] 응답 생성 완료")
        print(f"   - 검색 타입: {search_type}")
        print(f"   - 문서 수: {len(documents)}")
        
        # 출처 정보 구성 (기존 방식 유지: Document 메타데이터 기반)
        sources = []
        for doc in documents[:5]:  # 최대 5개만
            metadata = getattr(doc, "metadata", {}) or {}
            
            # 웹 검색 결과인 경우
            if search_type == "web" or metadata.get("source_type") == "web":
                # ★ langgraph_test.py에서 이미 정리된 title 사용
                sources.append({
                    "doc_id": str(metadata.get("source", "web_unknown"))[:50],
                    "title": metadata.get("title", "웹 검색 결과"),  # Tavily의 실제 제목
                    "source_type": "web",
                    "url": metadata.get("source", "")
                })
            # 내부 문서인 경우
            else:
                sources.append({
                    "doc_id": metadata.get("doc_id", "unknown"),
                    "title": metadata.get("title", "Unknown"),
                    "authors": metadata.get("authors", "Unknown"),
                    "year": metadata.get("publication_year", "Unknown"),
                    "source_type": "internal"
                })
        
        return {
            "response": answer,
            "sources": sources,
            "search_type": search_type,
            "doc_count": len(documents)
        }
    
    except Exception as e:
        print(f"❌ [ERROR] 채팅 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류 발생: {str(e)}")


# ============================================================================
# 서버 실행
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🤗 HuggingFace Papers RAG Server (langgraph_test 연결 버전)")
    print("=" * 70)
    print("Starting server on http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )