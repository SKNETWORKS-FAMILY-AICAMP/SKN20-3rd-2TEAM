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

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "01_data"
SRC_DIR = PROJECT_ROOT / "02_src"

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
</style>
"""

# ==================== 리소스 로딩 ====================

@st.cache_resource
def load_vectorstore():
    """VectorDB 로드 (캐싱)

    Returns:
        VectorStore 또는 None: ChromaDB 벡터 저장소
    """
    try:
        import pickle
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma

        # chunks_all.pkl 파일 로드
        chunks_path = DATA_DIR / "chunks" / "chunks_all.pkl"

        if not chunks_path.exists():
            st.error(f"❌ chunks_all.pkl 파일을 찾을 수 없습니다: {chunks_path}")
            return None

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        # ChromaDB 생성 (in-memory)
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        vectorstore = Chroma.from_documents(
            documents=chunks,
            collection_name='huggingface_papers',
            embedding=embeddings
        )

        st.success(f"✅ VectorDB 로드 완료: {len(chunks)}개 문서")
        return vectorstore

    except Exception as e:
        st.error(f"❌ VectorDB 로드 실패: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


@st.cache_resource
def load_keyword_manager():
    """KeywordManager 로드 (캐싱)

    Returns:
        KeywordManager 또는 None: 키워드 관리 객체
    """
    try:
        # TODO: KeywordManager 로드 로직 구현
        # 예시:
        # import sys
        # sys.path.insert(0, str(SRC_DIR / "02_utils"))
        # from documents import load_all_documents
        # from keyword_manager import KeywordManager
        #
        # documents = load_all_documents(year=2025, weeks=[49, 48, 47, 46, 45])
        # km = KeywordManager(documents)
        # return km

        st.info("⚠️ KeywordManager 로드 대기 중 - 키워드 시스템 미구현")
        return None

    except Exception as e:
        st.error(f"KeywordManager 로드 실패: {e}")
        return None


@st.cache_resource
def load_rag_system(_vectorstore):
    """RAG 시스템 초기화 (캐싱)

    Args:
        _vectorstore: VectorStore 객체 (언더스코어는 캐싱 제외를 위한 관례)

    Returns:
        SimpleRAGSystem 또는 None: RAG 시스템 객체
    """
    try:
        if _vectorstore is None:
            st.warning("⚠️ VectorDB가 로드되지 않아 RAG 시스템을 초기화할 수 없습니다.")
            return None

        # SimpleRAGSystem 임포트
        import sys
        sys.path.insert(0, str(SRC_DIR / "04_rag"))
        from simpleRAGsystem_2 import SimpleRAGSystem
        from langchain_openai import ChatOpenAI

        # LLM 초기화
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

        # RAG 시스템 초기화 (retriever_k=3으로 상위 3개 문서 검색)
        rag_system = SimpleRAGSystem(_vectorstore, llm, retriever_k=3)

        st.success("✅ RAG 시스템 초기화 완료")
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


# ==================== UI 렌더링 ====================

def render_header(keyword_manager, rag_system=None):
    """헤더: 제목 & 트렌드 키워드

    Args:
        keyword_manager: KeywordManager 객체 또는 None
        rag_system: SimpleRAGSystem 객체 또는 None (키워드 기반 검색용)
    """
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

    # 트렌드 키워드 영역
    st.markdown("### 🔥 트렌딩 키워드")
    
    # keyword_manager 구현 전 더미 데이터 사용
    if keyword_manager is None:
        st.info("💡 키워드 데이터를 불러오는 중...")
        # 데모용 더미 데이터
        trending = [
            ("LLM", 45), ("Transformer", 38), ("RAG", 32),
            ("Vision", 28), ("Diffusion", 25), ("Agent", 22),
            ("Multimodal", 20), ("RL", 18), ("NLP", 15), ("CV", 12)
        ]
    else:
        trending = keyword_manager.get_trending_keywords(
            year=2025,
            weeks=[49, 48, 47, 46, 45],
            top_n=10
        )

    # st.pills로 키워드 표시 (단일 선택으로 변경)
    keyword_labels = [f"{kw} ({count})" for kw, count in trending]

    selected = st.pills(
        label="키워드를 선택하여 관련 논문을 검색하세요",
        options=keyword_labels,
        selection_mode="single"  # 단일 선택으로 변경
    )

    # 선택된 키워드 처리
    if selected:
        # "키워드 (count)" → "키워드" 추출 (단일 선택이므로 문자열)
        keyword = selected.split(" (")[0]

        # 상태 업데이트 (단일 키워드)
        if keyword != st.session_state.get("selected_keyword", None):
            st.session_state.selected_keyword = keyword

            # 키워드 기반 검색 메시지
            query = f"📌 선택한 키워드: {keyword}"

            # RAG 시스템 사용 여부에 따라 응답 생성
            if rag_system is None:
                # RAG 시스템이 없으면 예시 응답 사용
                result_text = get_example_keyword_response(keyword)
            else:
                # 실제 RAG 시스템으로 키워드 기반 질문 생성
                keyword_query = f"{keyword}에 대한 최신 연구 동향을 알려주세요."
                result_text = rag_system.ask(keyword_query)

            add_message("user", query)
            add_message("assistant", result_text)

            st.rerun()

    st.markdown("---")


def render_chat_interface(rag_system):
    """채팅 인터페이스

    Args:
        rag_system: SimpleRAGSystem 객체 또는 None
    """
    # 채팅 헤더
    st.markdown("### 💬 논문 검색 채팅")
    
    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤗" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    # 사용자 입력
    user_input = st.chat_input(
        placeholder="🔍 논문에 대해 질문하거나 키워드를 입력하세요..."
    )

    if user_input:
        # 사용자 메시지 추가
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        add_message("user", user_input)

        # AI 응답 생성
        with st.chat_message("assistant", avatar="🤗"):
            with st.spinner("🔎 논문을 검색하고 답변을 생성하는 중..."):
                # RAG 시스템이 없으면 예시 응답 사용
                if rag_system is None:
                    # simpleRAGsystem_2.py의 출력 형식을 시뮬레이션
                    result = get_example_rag_response(user_input)
                    response_text = result['answer']
                else:
                    # 실제 RAG 시스템 호출
                    result = rag_system.ask_with_sources(user_input)
                    response_text = result['answer']

            # 응답 표시
            st.markdown(response_text)

            # 참조 논문 표시
            sources = result.get('sources', [])
            if sources:
                st.markdown("---")
                with st.expander(f"📚 참조된 논문 ({len(sources)}개)", expanded=True):
                    for i, source in enumerate(sources, 1):
                        render_paper_card(source, i)
            else:
                st.info("💡 검색된 관련 논문이 없습니다.")

        # 어시스턴트 응답 저장 (답변만 저장, 출처는 제외)
        add_message("assistant", response_text)

        st.rerun()


def render_sidebar(keyword_manager):
    """사이드바: 설정 & 통계

    Args:
        keyword_manager: KeywordManager 객체 또는 None
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
            st.rerun()

        # 캐시 초기화 버튼
        if st.button("🔄 캐시 초기화", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")

        # 통계 섹션
        st.markdown("### 📊 통계")

        # keyword_manager 구현 전 N/A 표시
        if keyword_manager is None:
            st.metric("📄 논문 개수", "로딩 중...")
            st.metric("🏷️ 키워드 개수", "로딩 중...")
            st.info("💡 데이터를 로드하는 중입니다.")
        else:
            keyword_stats = keyword_manager.get_keyword_stats()
            total_keywords = len(keyword_stats)
            total_papers = len(keyword_manager.documents)

            # 메트릭 표시
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 논문", total_papers)
            with col2:
                st.metric("🏷️ 키워드", total_keywords)

            st.metric("평균 키워드/논문", f"{total_keywords/total_papers:.1f}")

            # 전체 키워드 TOP 20 차트 (선택적)
            with st.expander("🏆 인기 키워드 TOP 20"):
                import pandas as pd

                top_keywords = sorted(
                    keyword_stats.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]

                df = pd.DataFrame(
                    top_keywords,
                    columns=["키워드", "논문 수"]
                )

                st.bar_chart(df.set_index("키워드"))

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
