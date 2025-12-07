"""
Streamlit UI 컴포넌트 모듈

이 모듈은 HuggingFace DailyPapers RAG 챗봇의 UI 컴포넌트를 제공합니다.
- 리소스 로딩 (VectorDB, KeywordManager, RAG System)
- 세션 상태 관리
- UI 렌더링 (헤더, 채팅, 사이드바)
"""

import streamlit as st
from pathlib import Path
from typing import List, Tuple
import sys
import json
from collections import Counter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "01_data"
SRC_DIR = PROJECT_ROOT / "02_src"

# vectordb 모듈 import
sys.path.insert(0, str(SRC_DIR / "02_utils"))
from vectordb import load_vectordb

# SimpleRAGSystem 임포트
sys.path.insert(0, str(SRC_DIR / "04_rag"))
from simpleRAGsystem_2 import SimpleRAGSystem

# HuggingFace 스타일 CSS
HUGGINGFACE_STYLE = """
<style>
    /* 메인 컨테이너 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* 헤더 스타일 */
    .header-container {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }

    /* 키워드 태그 스타일 */
    .stPills {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* 채팅 메시지 스타일 */
    .stChatMessage {
        background: white;
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* 사이드바 스타일 */
    .css-1d391kg {
        background: #f8f9fa;
    }

    /* 메트릭 카드 스타일 */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
    }

    /* HuggingFace 로고 색상 */
    .hf-color {
        color: #FF9D00;
    }

    /* 논문 카드 스타일 */
    .paper-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }

    .paper-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    /* Streamlit의 text_input 컴포넌트의 특정 클래스(data-testid)를 타겟팅하여 컨테이너에 맞게 스타일 조정 */
    .fixed-bottom-container div[data-testid="stTextInput"] {
        margin-bottom: 0;
    }

    /* 메인 컨텐츠 영역 - 하단 고정 요소를 위한 여백 */
    .main-content {
        padding-bottom: 280px;
        min-height: 100vh;
    }

    /* 트렌드 키워드 제목 스타일 */
    .trend-title {
        color: #FF9D00;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }

    /* 검색 제목 스타일 */
    .search-title {
        color: #FF9D00;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
"""


# ==================== 키워드 추출 ====================

def get_trending_keywords_from_json(weeks: int = 6, top_n: int = 7) -> List[Tuple[str, int]]:
    """
    최근 N주간의 JSON 데이터에서 트렌딩 키워드 추출

    Args:
        weeks: 분석할 최근 주 수 (기본값: 6)
        top_n: 반환할 상위 키워드 개수 (기본값: 7)

    Returns:
        List of tuples: [(키워드, 개수), ...]
    """
    try:
        docs_dir = PROJECT_ROOT / "01_data" / "documents" / "2025"

        if not docs_dir.exists():
            raise FileNotFoundError("문서 디렉토리가 존재하지 않습니다")

        all_keywords = []

        # 모든 주차 디렉토리를 이름순으로 정렬 (내림차순)
        week_dirs = sorted([d for d in docs_dir.iterdir() if d.is_dir()], reverse=True)

        # 최근 N주 데이터 처리
        for week_dir in week_dirs[:weeks]:
            for json_file in week_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        tags = data.get('metadata', {}).get('tags', [])
                        all_keywords.extend(tags)
                except Exception:
                    # 문제가 있는 파일은 건너뛰기
                    continue

        # 키워드를 찾지 못한 경우 예외 발생
        if not all_keywords:
            raise ValueError("키워드를 찾을 수 없습니다")

        # 키워드 개수 계산 후 상위 N개 반환
        keyword_counts = Counter(all_keywords)
        return keyword_counts.most_common(top_n)

    except Exception:
        # 모든 예외 발생 시 더미 데이터 반환
        return [
            ("LLM", 45), ("Transformer", 38), ("RAG", 32),
            ("Vision", 28), ("Diffusion", 25), ("Agent", 22),
            ("Multimodal", 20)
        ][:top_n]


# ==================== 리소스 로딩 ====================
def load_vectorstore():
    """VectorDB 로드 (세션 스테이트 사용)

    Returns:
        VectorStore 또는 None: ChromaDB 벡터 저장소
    """
    # 이미 로드된 경우 재사용
    if "vectorstore" in st.session_state:
        return st.session_state.vectorstore

    try:
        with st.spinner("🔄 VectorDB 로딩 중..."):
            # vectordb.py의 load_vectordb() 함수 호출
            vectorstore = load_vectordb(
                model_name="MiniLM-L6",
                chunk_size=100,
                chunk_overlap=10
            )

            # 세션 스테이트에 저장
            st.session_state.vectorstore = vectorstore
            st.toast("✅ VectorDB 로드 완료", icon="✅")
            return vectorstore

    except Exception as e:
        st.error(f"❌ VectorDB 로드 실패: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def load_rag_system(vectorstore):
    """RAG 시스템 초기화 (세션 스테이트 사용)

    Args:
        vectorstore: VectorStore 객체

    Returns:
        SimpleRAGSystem 또는 None: RAG 시스템 객체
    """
    # 이미 로드된 경우 재사용
    if "rag_system" in st.session_state:
        return st.session_state.rag_system

    try:
        if vectorstore is None:
            st.warning("⚠️ VectorDB가 로드되지 않아 RAG 시스템을 초기화할 수 없습니다.")
            return None

        with st.spinner("🔄 RAG 시스템 초기화 중..."):
            # LLM 초기화
            llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

            # RAG 시스템 초기화 (retriever_k=3으로 상위 3개 문서 검색)
            rag_system = SimpleRAGSystem(vectorstore, llm, retriever_k=3)

            # 세션 스테이트에 저장
            st.session_state.rag_system = rag_system
            st.toast("✅ RAG 시스템 초기화 완료", icon="✅")
            return rag_system

    except Exception as e:
        st.error(f"❌ RAG 시스템 초기화 실패: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# ==================== 세션 초기화 ====================

def init_session_state():
    """세션 상태 초기화

    Streamlit session_state에 필요한 변수들을 초기화합니다.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_keyword" not in st.session_state:
        st.session_state.selected_keyword = None  # 단일 키워드

    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "chat"  # "chat" or "keyword"

    if "last_searched_keyword" not in st.session_state:
        st.session_state.last_searched_keyword = None  # 중복 검색 방지

    if "keyword_selection_key" not in st.session_state:
        st.session_state.keyword_selection_key = 0  # pills 위젯 초기화용 카운터


# ==================== UI 렌더링 ====================

def render_header():
    """헤더: 제목"""
    # HuggingFace 스타일 CSS 적용
    st.markdown(HUGGINGFACE_STYLE, unsafe_allow_html=True)

    # 헤더 컨테이너
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color: #FF9D00; font-size: 3rem; margin-bottom: 0.5rem;'>
                🤗 HuggingFace DailyPapers
            </h1>
            <p style='color: #6c757d; font-size: 1.2rem; margin-top: 0;'>
                RAG 기반 최신 ML/DL/LLM 논문 검색 챗봇
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")


def render_chat_interface(rag_system):
    """채팅 인터페이스

    Args:
        rag_system: SimpleRAGSystem 객체 또는 None
    """
    # 1. 메인 컨텐츠 영역 (답변 표시)
    # st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Q&A 메시지 표시
    if len(st.session_state.messages) == 0:
        st.markdown("""
            <div style='text-align: center; color: #6c757d; padding: 3rem 1rem;'>
                <h3 style='color: #FF9D00;'>💬 대화를 시작해보세요!</h3>
                <p>하단의 트렌드 키워드를 선택하거나 검색창에 질문을 입력하세요.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤗" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])

    # st.markdown('</div>', unsafe_allow_html=True)

    # 2. 하단 고정 영역 (트렌드 키워드 + 검색창)
    # 트렌드 키워드
    st.markdown('<div class="trend-title">🔥 트렌드 키워드</div>', unsafe_allow_html=True)

    trending = get_trending_keywords_from_json(weeks=6, top_n=7)
    keyword_labels = [kw for kw, count in trending]  # 개수 제거, 키워드만 표시

    selected = st.pills(
        label="trend keyword",
        options=keyword_labels,
        selection_mode="single",
        label_visibility="collapsed",
        key=f"keyword_pills_{st.session_state.keyword_selection_key}"
    )

    user_input = st.chat_input(
        placeholder=" 🔍 논문에 대해 질문하거나 키워드를 입력하세요..."
    )

    # 3. 키워드 선택 시 검색 실행
    if selected:
        keyword = selected

        # 중복 검색 방지
        if keyword != st.session_state.get("last_searched_keyword", None):
            # 키워드 기반 검색 메시지
            query = f"📌 선택한 키워드: {keyword}"

            # 사용자 메시지 추가
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)
            add_message("user", query)

            # AI 응답 생성
            with st.chat_message("assistant", avatar="🤗"):
                # RAG 시스템이 없으면 예시 응답 사용
                if rag_system is None:
                    with st.spinner("🔎 논문을 검색하고 답변을 생성하는 중..."):
                        result_text = get_example_keyword_response(keyword)
                    st.markdown(result_text)
                else:
                    # 실제 RAG 시스템으로 키워드 기반 질문 생성 (스트리밍)
                    keyword_query = f"{keyword}에 대한 최신 연구 동향을 알려주세요."

                    with st.spinner("🔎 논문 검색 중..."):
                        result = rag_system.ask_with_sources(keyword_query, stream=True)

                    # 스트리밍 응답 표시
                    result_text = st.write_stream(result['answer_stream'])

                # 참조 논문 표시
                # if rag_system is not None:
                #     sources = result.get('sources', [])
                #     if sources:
                #         st.markdown("---")
                #         with st.expander(f"📚 참조된 논문 ({len(sources)}개)", expanded=True):
                #             for i, source in enumerate(sources, 1):
                #                 render_paper_card(source, i)
                #     else:
                #         st.info("💡 검색된 관련 논문이 없습니다.")

            # 어시스턴트 응답 저장
            add_message("assistant", result_text)

            # 키워드 선택 해제 및 상태 업데이트
            st.session_state.last_searched_keyword = keyword
            st.session_state.keyword_selection_key += 1
            st.rerun()

    # 사용자 입력 처리
    if user_input:
        # 키워드 선택 해제
        st.session_state.keyword_selection_key += 1
        st.session_state.last_searched_keyword = None

        # 사용자 메시지 추가
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        add_message("user", user_input)

        # AI 응답 생성
        with st.chat_message("assistant", avatar="🤗"):
            # RAG 시스템이 없으면 예시 응답 사용
            if rag_system is None:
                with st.spinner("🔎 논문을 검색하고 답변을 생성하는 중..."):
                    # simpleRAGsystem_2.py의 출력 형식을 시뮬레이션
                    result = get_example_rag_response(user_input)
                    response_text = result['answer']
                st.markdown(response_text)
            else:
                # 실제 RAG 시스템 호출 (스트리밍)
                with st.spinner("🔎 논문 검색 중..."):
                    result = rag_system.ask_with_sources(user_input, stream=True)

                # 스트리밍 응답 표시
                response_text = st.write_stream(result['answer_stream'])

            # 참조 논문 표시
            # sources = result.get('sources', [])
            # if sources:
            #     st.markdown("---")
            #     with st.expander(f"📚 참조된 논문 ({len(sources)}개)", expanded=True):
            #         for i, source in enumerate(sources, 1):
            #             render_paper_card(source, i)
            # else:
            #     st.info("💡 검색된 관련 논문이 없습니다.")

        # 어시스턴트 응답 저장 (답변만 저장, 출처는 제외)
        add_message("assistant", response_text)


def render_sidebar(rag_system=None):
    """사이드바: 설정 & 통계

    Args:
        rag_system: SimpleRAGSystem 객체 (대화 히스토리 초기화용)
    """
    with st.sidebar:
        # 로고 영역
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0; margin-bottom: 1rem;'>
                <h2 style='color: #FF9D00; margin: 0;'>🤗</h2>
                <p style='color: #6c757d; font-size: 0.9rem; margin: 0;'>HuggingFace Papers</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 설정 섹션
        st.markdown("### ⚙️ 설정")

        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.selected_keyword = None
            st.session_state.last_searched_keyword = None
            # 키워드 선택 해제
            st.session_state.keyword_selection_key += 1
            # RAG 시스템의 chat_history도 초기화
            if rag_system is not None:
                rag_system.clear_history()
            st.toast("✅ 대화가 초기화되었습니다", icon="✅")
            st.rerun()

        # 캐시 초기화 버튼
        if st.button("🔄 캐시 초기화", use_container_width=True):
            # 대화 초기화
            st.session_state.messages = []
            st.session_state.selected_keyword = None
            st.session_state.last_searched_keyword = None
            # 키워드 선택 해제
            st.session_state.keyword_selection_key += 1
            # 세션 스테이트에서 VectorDB와 RAG 시스템 제거
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore
            if "rag_system" in st.session_state:
                del st.session_state.rag_system
            st.toast("✅ 캐시와 대화가 모두 초기화되었습니다. 다음 요청 시 재로드됩니다.", icon="✅")
            st.rerun()

        st.markdown("---")

        # 통계 섹션
        st.markdown("### 📊 통계")

        # 논문 및 키워드 개수 표시
        st.metric("📄 논문 개수", "506")
        st.metric("🏷️ 키워드 개수", "1,449")
        st.info("💡 최근 6주간의 데이터")

        st.markdown("---")

        # 정보 섹션
        st.markdown("### ℹ️ 정보")
        st.markdown("""
            <div style='font-size: 0.85rem; color: #6c757d;'>
                <p><strong>데이터 소스:</strong><br/>HuggingFace DailyPapers</p>
                <p><strong>업데이트:</strong><br/>최근 5주 논문</p>
                <p><strong>기술 스택:</strong><br/>RAG + LangChain + OpenAI</p>
            </div>
        """, unsafe_allow_html=True)


# ==================== 헬퍼 함수 ====================

def add_message(role: str, content: str):
    """메시지 추가

    Args:
        role: "user" 또는 "assistant"
        content: 메시지 내용
    """
    st.session_state.messages.append({
        "role": role,
        "content": content
    })


def get_example_keyword_response(keyword: str) -> str:
    """키워드 선택 시 예시 응답 생성

    Args:
        keyword: 선택된 키워드

    Returns:
        str: 마크다운 형식의 응답
    """
    return f"""
### 🔍 '{keyword}' 관련 논문 검색 결과

선택하신 키워드와 관련된 최신 논문들입니다.

#### 📊 검색 통계
- 전체 논문 수: **3개**
- 평균 추천수: **203.3**
- 주요 연구 분야: Machine Learning, Deep Learning

---

**💡 참고**: 이는 RAG 시스템이 구현되었을 때의 예시 출력입니다.
실제 시스템이 연결되면 VectorDB에서 '{keyword}' 키워드로 태그된 논문들을 검색하여 표시합니다.

**실제 구현 시 동작:**
1. KeywordManager가 '{keyword}' 태그를 가진 논문들을 검색
2. 관련도가 높은 상위 10개 논문을 선택
3. 각 논문의 메타데이터(제목, 링크, 추천수, 태그)를 카드 형식으로 표시
"""


def get_example_rag_response(question: str) -> dict:
    """RAG 시스템 예시 응답 생성

    실제 RAG 시스템이 연결되면 이 함수는 제거되고
    rag_system.ask_with_sources()로 대체됩니다.

    Args:
        question: 사용자 질문

    Returns:
        dict: {"answer": str, "sources": list} 형식
    """
    # 예시 응답 - simpleRAGsystem_2.py의 출력 형식과 동일
    example_answer = f"""
### 질문에 대한 답변

**질문:** {question}

#### 📌 핵심 요약
RAG(Retrieval-Augmented Generation)는 외부 지식 베이스를 활용하여 LLM의 답변 품질을 향상시키는 기법입니다.

#### 💡 주요 인사이트
- **검색 기반 생성**: 관련 문서를 먼저 검색한 후 LLM이 답변 생성
- **환각(Hallucination) 감소**: 실제 문서를 참조하여 사실 기반 답변 제공
- **최신 정보 활용**: 학습 데이터 외의 최신 정보도 활용 가능
- **컨텍스트 확장**: 긴 문맥을 효율적으로 처리

#### 📚 관련 논문 (상위 3개)

1. **Retrieval-Augmented Generation for Large Language Models: A Survey**
   - RAG 시스템의 전반적인 구조와 최신 트렌드를 다룬 서베이 논문
   - 다양한 RAG 변형 기법들을 비교 분석

2. **Self-RAG: Learning to Retrieve, Generate, and Critique**
   - 자기 성찰 기반 RAG 시스템
   - 검색 결과의 품질을 스스로 평가하고 개선

3. **CRAG: Corrective Retrieval Augmented Generation**
   - 검색 결과의 오류를 자동으로 수정하는 기법
   - 신뢰도 기반 필터링 적용

#### 📖 상세 설명

RAG는 크게 3단계로 구성됩니다:

1. **검색(Retrieval)**: 벡터 데이터베이스에서 관련 문서 검색
2. **증강(Augmentation)**: 검색된 문서를 프롬프트에 추가
3. **생성(Generation)**: LLM이 증강된 컨텍스트 기반으로 답변 생성

최근 연구 트렌드:
- Hybrid Retrieval (Dense + Sparse)
- Self-Reflection RAG
- Adaptive Retrieval (필요시에만 검색)
- Multi-hop RAG (다단계 검색)

---

**💡 참고**: 이는 RAG 시스템이 구현되었을 때의 예시 출력입니다.
실제 시스템이 연결되면 데이터베이스의 논문 정보를 기반으로 답변이 생성됩니다.
"""

    # 예시 출처 데이터 - simpleRAGsystem_2.py의 sources 형식과 동일
    example_sources = [
        {
            "paper_name": "Retrieval-Augmented Generation for Large Language Models: A Survey",
            "huggingface_url": "https://huggingface.co/papers/2312.10997",
            "github_url": "https://github.com/example/rag-survey",
            "upvote": 245,
            "tags": ["RAG", "LLM", "Survey", "Retrieval"]
        },
        {
            "paper_name": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            "huggingface_url": "https://huggingface.co/papers/2310.11511",
            "github_url": "https://github.com/example/self-rag",
            "upvote": 198,
            "tags": ["RAG", "Self-Reflection", "LLM"]
        },
        {
            "paper_name": "CRAG: Corrective Retrieval Augmented Generation",
            "huggingface_url": "https://huggingface.co/papers/2401.15884",
            "github_url": None,
            "upvote": 167,
            "tags": ["RAG", "Corrective", "Retrieval"]
        }
    ]

    return {
        "answer": example_answer,
        "sources": example_sources
    }


def format_papers_as_markdown(papers: List) -> str:
    """논문 목록을 마크다운으로 포맷팅

    Args:
        papers: Document 객체 리스트

    Returns:
        str: 마크다운 형식의 논문 목록
    """
    if not papers:
        return "관련 논문을 찾지 못했습니다."

    output = "### 📚 검색 결과\n\n"

    for i, doc in enumerate(papers, 1):
        metadata = doc.metadata

        title = metadata.get('paper_name', 'N/A')
        upvote = metadata.get('upvote', 0)
        tags = metadata.get('tags', [])
        hf_url = metadata.get('huggingface_url', '#')
        github_url = metadata.get('github_url', None)

        output += f"**{i}. {title}**\n"
        output += f"- 👍 추천수: {upvote}\n"
        output += f"- 🏷️ 태그: {', '.join(tags[:3])}\n"
        output += f"- 🔗 [HuggingFace 논문]({hf_url})"

        if github_url:
            output += f" | [GitHub]({github_url})"

        output += "\n\n"

    return output


def render_paper_card(source: dict, index: int):
    """논문 카드 렌더링 (HuggingFace 스타일)

    simpleRAGsystem_2.py의 sources 형식에 맞춰 렌더링:
    {
        "paper_name": str,
        "huggingface_url": str,
        "github_url": str or None,
        "upvote": int,
        "tags": list
    }

    Args:
        source: 논문 정보 딕셔너리
        index: 논문 순서 번호
    """
    # 논문 제목과 추천수
    paper_name = source.get('paper_name', '(제목 없음)')
    upvote = source.get('upvote', 0)

    # 카드 컨테이너
    st.markdown(f"""
        <div style='background: white; border-radius: 0.75rem; padding: 1.5rem; margin: 1rem 0;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); border-left: 4px solid #FF9D00;'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;'>
                <div style='flex: 1;'>
                    <h4 style='color: #1f2937; margin: 0; font-size: 1.1rem;'>{index}. {paper_name}</h4>
                </div>
                <div style='margin-left: 1rem;'>
                    <span style='background: #FF9D00; color: white; padding: 0.35rem 0.75rem;
                                 border-radius: 1rem; font-size: 0.9rem; font-weight: 600;'>
                        👍 {upvote}
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 태그 표시
    tags = source.get('tags', [])
    if tags:
        tag_badges = []
        for tag in tags:
            tag_badges.append(
                f"<span style='background: #f3f4f6; color: #374151; padding: 0.25rem 0.75rem; "
                f"border-radius: 9999px; font-size: 0.85rem; margin-right: 0.5rem; "
                f"display: inline-block; margin-bottom: 0.25rem;'>"
                f"🏷️ {tag}</span>"
            )
        st.markdown("".join(tag_badges), unsafe_allow_html=True)

    st.markdown("")  # 공백

    # 링크
    col1, col2 = st.columns(2)
    with col1:
        hf_url = source.get('huggingface_url', '#')
        if hf_url and hf_url != '#':
            st.markdown(f"[🤗 HuggingFace 논문]({hf_url})")

    with col2:
        github_url = source.get('github_url', None)
        if github_url:
            st.markdown(f"[💻 GitHub 저장소]({github_url})")

    st.markdown("---")
