"""
FastAPI RAG Chatbot Server (Fixed Path Version)

이 서버는 HTML 프론트엔드와 LangGraph RAG 시스템(langgraph_final.py)을 연결합니다.
- 통계 정보 제공
- 트렌드 키워드 제공
- LangGraph 기반 채팅 응답 (하이브리드 검색 + 웹 검색)
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# ============================================================================
# 경로 설정 (수정됨)
# ============================================================================

# 현재 파일의 절대 경로
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent  # 04_ui 디렉토리

# 프로젝트 구조에 맞게 경로 설정
# 현재 위치: 02_src/04_ui/app_v4_main.py
SRC_DIR = CURRENT_DIR.parent  # 02_src
PROJECT_ROOT = SRC_DIR.parent  # 프로젝트 루트

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "01_data"
CLUSTERS_DIR = DATA_DIR / "clusters"

# RAG 모듈 경로
RAG_DIR = SRC_DIR / "03_rag"
UTILS_DIR = SRC_DIR / "02_utils"

# sys.path에 추가 (맨 앞에 추가해서 우선순위 높임)
sys.path.insert(0, str(RAG_DIR))
sys.path.insert(0, str(UTILS_DIR))

# 경로 디버깅 출력
print("\n" + "=" * 70)
print("📁 경로 설정")
print("=" * 70)
print(f"현재 파일: {CURRENT_FILE}")
print(f"현재 디렉토리: {CURRENT_DIR}")
print(f"SRC 디렉토리: {SRC_DIR}")
print(f"프로젝트 루트: {PROJECT_ROOT}")
print(f"RAG 디렉토리: {RAG_DIR}")
print(f"UTILS 디렉토리: {UTILS_DIR}")
print(f"RAG 디렉토리 존재: {RAG_DIR.exists()}")
print(f"langgraph_final.py 존재: {(RAG_DIR / 'langgraph_final.py').exists()}")
print("=" * 70 + "\n")

# ============================================================================
# FastAPI 앱 생성
# ============================================================================

app = FastAPI(
    title="HuggingFace Papers RAG API",
    description="LangGraph 기반 RAG 챗봇 API (Hybrid Search + Web Search)",
    version="2.0.0"
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

# LangGraph 시스템 초기화 여부
system_initialized = False

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
    """서버 시작 시 LangGraph RAG 시스템 초기화"""
    global system_initialized
    
    print("\n" + "=" * 70)
    print("🚀 HuggingFace Papers RAG Server - Starting Up")
    print("=" * 70)
    
    try:
        # langgraph_final 모듈 임포트 시도
        print("\n[STEP 1/2] langgraph_final 모듈 임포트 중...")
        
        try:
            from langgraph_final import initialize_langgraph_system, get_system_status
            print("✅ langgraph_final 모듈 임포트 성공")
        except ModuleNotFoundError as e:
            print(f"❌ 모듈을 찾을 수 없습니다: {e}")
            print("\n[디버깅] sys.path 확인:")
            for i, path in enumerate(sys.path[:10]):
                print(f"  [{i}] {path}")
            
            print("\n[디버깅] RAG 디렉토리 내용:")
            if RAG_DIR.exists():
                for item in RAG_DIR.iterdir():
                    print(f"  - {item.name}")
            else:
                print(f"  RAG 디렉토리가 존재하지 않습니다: {RAG_DIR}")
            
            raise
        
        # LangGraph 시스템 초기화
        print("\n[STEP 2/2] LangGraph 시스템 초기화 중...")
        result = initialize_langgraph_system()
        
        if result['status'] == 'success':
            system_initialized = True
            print("\n" + "=" * 70)
            print("✅ RAG 서버 준비 완료!")
            print("📡 API 문서: http://localhost:8000/docs")
            print("🌐 웹 인터페이스: http://localhost:8000")
            print("=" * 70 + "\n")
        else:
            print(f"\n❌ 초기화 실패: {result.get('message')}")
            raise Exception(result.get('message'))
        
    except Exception as e:
        print(f"\n❌ [FAILED] 서버 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    """
    루트 경로 - HTML 인터페이스 제공
    """
    try:
        # app_v3.html 파일 찾기 (여러 경로 시도)
        possible_paths = [
            CURRENT_DIR / "app_v3.html",  # 같은 디렉토리
            PROJECT_ROOT / "app_v3.html",  # 프로젝트 루트
            CURRENT_DIR / "app_v4.html",   # v4 버전도 시도
        ]
        
        html_path = None
        for path in possible_paths:
            if path.exists():
                html_path = path
                print(f"[INFO] HTML 파일 발견: {path}")
                break
        
        if html_path is None:
            # HTML 파일이 없으면 기본 메시지 반환
            return """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>RAG Chatbot</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                        h1 { color: #ff9d00; }
                        .info { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
                    </style>
                </head>
                <body>
                    <h1>🤗 HuggingFace Papers RAG Chatbot</h1>
                    <div class="info">
                        <p><strong>서버가 정상 작동 중입니다!</strong></p>
                        <p>HTML 파일을 찾을 수 없습니다. app_v3.html 또는 app_v4.html을 다음 위치에 배치해주세요:</p>
                        <ul>
                            <li>""" + str(CURRENT_DIR) + """</li>
                            <li>""" + str(PROJECT_ROOT) + """</li>
                        </ul>
                    </div>
                    <div class="info">
                        <p><strong>대신 API를 직접 사용할 수 있습니다:</strong></p>
                        <ul>
                            <li><a href="/docs">API 문서 보기</a></li>
                            <li><a href="/api/health">서버 상태 확인</a></li>
                            <li><a href="/api/stats">통계 정보</a></li>
                        </ul>
                    </div>
                </body>
            </html>
            """
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        print(f"[ERROR] HTML 파일 로드 실패: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>오류</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    try:
        from langgraph_final import get_system_status
        status = get_system_status()
        
        return {
            "status": "healthy" if status['initialized'] else "initializing",
            "system_status": status
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
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
                metadata_data = json.load(f)
            keyword_count = metadata_data.get("_metadata", {}).get("n_clusters", 0)
        
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
    LangGraph RAG 기반 채팅 응답 생성
    
    1. langgraph_final.ask_question() 호출
    2. 최종 답변 및 검색 타입, 참조 문서 반환
    """
    if not system_initialized:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    
    try:
        from langgraph_final import ask_question
        
        query = request.message
        print(f"\n{'='*70}")
        print(f"📝 [QUERY] {query}")
        print(f"{'='*70}")
        
        # LangGraph 시스템에 질문
        result = ask_question(query, verbose=True)
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"질문 처리 실패: {result.get('error')}"
            )
        
        # 응답 데이터 구성
        answer = result.get('answer', '')
        sources_data = result.get('sources', [])
        metadata = result.get('metadata', {})
        
        search_type = metadata.get('search_type', 'unknown')
        
        print(f"✅ [SUCCESS] 응답 생성 완료")
        print(f"   - 검색 타입: {search_type}")
        print(f"   - 출처 수: {len(sources_data)}")
        
        # 출처 정보 포맷팅 (프론트엔드 형식에 맞게)
        formatted_sources = []
        for source in sources_data[:5]:  # 최대 5개
            source_type = source.get('type', 'unknown')
            
            if source_type == 'web':
                # 웹 검색 결과
                web_url = source.get('url', '')
                formatted_sources.append({
                    "doc_id": str(source.get('url', 'web_unknown'))[:50],
                    "title": source.get('title', '웹 검색 결과'),
                    "source_type": "web",
                    "url": web_url  # ✅ 수정: 'source' → 'url'
                })
                print(f"  [웹] {source.get('title', 'N/A')[:50]}")
                print(f"       URL: {web_url[:80]}")
            else:
                # 내부 논문
                hf_url = source.get('huggingface_url', '')
                gh_url = source.get('github_url', '')
                formatted_sources.append({
                    "doc_id": source.get('doc_id', 'unknown'),
                    "title": source.get('title', 'Unknown'),
                    "authors": source.get('authors', 'Unknown'),
                    "year": source.get('year', 'Unknown'),
                    "source_type": "internal",
                    "upvote": source.get('upvote', 0),
                    "hf_url": hf_url,  # ✅ 수정: 'source' → 'huggingface_url'
                    "github_url": gh_url
                })
                print(f"  [논문] {source.get('title', 'N/A')[:50]}")
                print(f"        HF: {hf_url[:80]}")
                if gh_url:
                    print(f"        GH: {gh_url[:80]}")
        
        return {
            "response": answer,
            "sources": formatted_sources,
            "search_type": search_type,
            "doc_count": len(sources_data),
            "metadata": {
                "is_korean": metadata.get('is_korean', False),
                "translated_question": metadata.get('translated_question'),
                "relevance_level": metadata.get('relevance_level', ''),
                "cluster_id": metadata.get('cluster_id')
            }
        }
    
    except HTTPException:
        raise
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
    print("🤗 HuggingFace Papers RAG Server (Fixed Path Version)")
    print("=" * 70)
    print("Starting server on http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Web Interface: http://localhost:8000")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )