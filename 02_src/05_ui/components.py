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

# ==================== 리소스 로딩 ====================

@st.cache_resource
def load_vectorstore():
    """VectorDB 로드 (캐싱)

    Returns:
        VectorStore 또는 None: ChromaDB 벡터 저장소
    """
    try:
        # TODO: VectorDB 로드 로직 구현
        # 예시:
        # from langchain_openai import OpenAIEmbeddings
        # from langchain_chroma import Chroma
        #
        # embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        # chroma_path = DATA_DIR / "chroma_db"
        # vectorstore = Chroma(
        #     collection_name='huggingface_papers',
        #     embedding_function=embeddings,
        #     persist_directory=str(chroma_path)
        # )
        # return vectorstore

        st.info("⚠️ VectorDB 로드 대기 중 - RAG 시스템 미구현")
        return None

    except Exception as e:
        st.error(f"VectorDB 로드 실패: {e}")
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
            st.warning("VectorDB가 로드되지 않아 RAG 시스템을 초기화할 수 없습니다.")
            return None

        # TODO: RAG 시스템 초기화 로직 구현
        # 예시:
        # import sys
        # sys.path.insert(0, str(SRC_DIR / "04_rag"))
        # from rag_system import SimpleRAGSystem
        # from langchain_openai import ChatOpenAI
        #
        # llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        # rag_system = SimpleRAGSystem(_vectorstore, llm, retriever_k=5)
        # return rag_system

        st.info("⚠️ RAG 시스템 초기화 대기 중")
        return None

    except Exception as e:
        st.error(f"RAG 시스템 초기화 실패: {e}")
        return None


# ==================== 세션 초기화 ====================

def init_session_state():
    """세션 상태 초기화

    Streamlit session_state에 필요한 변수들을 초기화합니다.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = []

    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "chat"  # "chat" or "keyword"


# ==================== UI 렌더링 ====================

def render_header(keyword_manager):
    """헤더: 제목 & 트렌드 키워드

    Args:
        keyword_manager: KeywordManager 객체 또는 None
    """
    st.title("📚 HuggingFace DailyPapers 논문 챗봇")
    st.markdown("RAG 기반 최신 ML/DL/LLM 논문 검색")
    st.markdown("---")

    # 트렌드 키워드 영역
    st.subheader("🔥 최근 5주 트렌드 키워드")

    # keyword_manager 구현 전 더미 데이터 사용
    if keyword_manager is None:
        st.info("키워드 데이터를 불러오는 중...")
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

    # st.pills로 키워드 표시 (다중 선택)
    keyword_labels = [f"{kw} ({count})" for kw, count in trending]

    selected = st.pills(
        label="키워드 클릭 → 관련 논문 표시",
        options=keyword_labels,
        selection_mode="multi"
    )

    # 선택된 키워드 처리
    if selected:
        # "키워드 (count)" → "키워드" 추출
        keywords = [label.split(" (")[0] for label in selected]

        # 상태 업데이트
        if keywords != st.session_state.selected_keywords:
            st.session_state.selected_keywords = keywords

            # 키워드 기반 검색 메시지
            query = f"📌 선택한 키워드: {', '.join(keywords)}"
            response = f"'{', '.join(keywords)}' 키워드 관련 논문을 검색합니다.\n\n(RAG 시스템 구현 후 실제 논문이 표시됩니다)"

            add_message("user", query)
            add_message("assistant", response)

            st.rerun()


def render_chat_interface(rag_system):
    """채팅 인터페이스

    Args:
        rag_system: SimpleRAGSystem 객체 또는 None
    """
    st.subheader("💬 채팅")

    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    user_input = st.chat_input(
        placeholder="논문 검색 또는 질문을 입력하세요..."
    )

    if user_input:
        # 사용자 메시지 추가
        with st.chat_message("user"):
            st.markdown(user_input)
        add_message("user", user_input)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # RAG 시스템 구현 전 더미 응답
                if rag_system is None:
                    response_text = f"질문을 받았습니다: '{user_input}'\n\n(RAG 시스템이 구현되면 관련 논문과 답변을 제공합니다)"
                else:
                    # TODO: 실제 RAG 시스템 호출
                    result = rag_system.ask_with_sources(user_input)
                    response_text = result['answer']

                    # 참조 논문 표시 (옵션)
                    if result.get('sources'):
                        with st.expander("📚 참조된 논문 (클릭)"):
                            for i, source in enumerate(result['sources'], 1):
                                render_paper_card(source, i)

            # 응답 표시
            st.markdown(response_text)

        # 어시스턴트 응답 저장
        add_message("assistant", response_text)

        st.rerun()


def render_sidebar(keyword_manager):
    """사이드바: 설정 & 통계

    Args:
        keyword_manager: KeywordManager 객체 또는 None
    """
    with st.sidebar:
        st.title("⚙️ 설정")

        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_keywords = []
            st.rerun()

        # 캐시 초기화 버튼
        if st.button("🔄 캐시 초기화", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")

        # 통계
        st.subheader("📊 통계")

        # keyword_manager 구현 전 N/A 표시
        if keyword_manager is None:
            st.metric("📄 논문 개수", "N/A")
            st.metric("🏷️ 키워드 개수", "N/A")
        else:
            keyword_stats = keyword_manager.get_keyword_stats()
            total_keywords = len(keyword_stats)
            total_papers = len(keyword_manager.documents)

            st.metric("📄 논문 개수", total_papers)
            st.metric("🏷️ 키워드 개수", total_keywords)
            st.metric("평균 키워드/논문", f"{total_keywords/total_papers:.1f}")

            # 전체 키워드 TOP 20 차트 (선택적)
            with st.expander("🏆 전체 키워드 TOP 20"):
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
    """논문 카드 렌더링

    Args:
        source: 논문 정보 딕셔너리
        index: 논문 순서 번호
    """
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(f"**{index}. {source['title']}**")
        st.caption(source['content'][:150] + "...")

        tags = source['metadata'].get('tags', [])
        if tags:
            st.caption(f"🏷️ {', '.join(tags[:3])}")

    with col2:
        st.metric("👍", source['upvote'])

        hf_url = source['metadata'].get('huggingface_url', '#')
        st.markdown(f"[논문 링크]({hf_url})")
