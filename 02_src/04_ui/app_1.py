"""
HuggingFace DailyPapers RAG 챗봇 - Streamlit 메인 애플리케이션

이 애플리케이션은 HuggingFace DailyPapers 데이터를 기반으로
RAG(Retrieval-Augmented Generation) 패턴을 사용하여
최신 ML/DL/LLM 논문을 검색하고 추천하는 챗봇입니다.
"""

import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경 변수 로드
load_dotenv()

# 컴포넌트 임포트 
try:
    from components import (
        load_vectorstore,
        load_rag_system,
        init_session_state,
        render_header,
        render_chat_interface,
        render_sidebar
    )
except ImportError as e:
    st.error(f"컴포넌트 임포트 실패: {e}")
    st.info("components.py 파일이 동일 디렉토리에 있는지 확인해주세요.")
    st.stop()


def main():
    """메인 애플리케이션"""

    # 페이지 설정
    st.set_page_config(
        page_title="HuggingFace DailyPapers 챗봇",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 세션 초기화
    init_session_state()

    # 리소스 로드
    vectorstore = load_vectorstore()
    rag_system = load_rag_system(vectorstore)

    # UI 렌더링
    render_header()
    render_chat_interface(rag_system)
    render_sidebar(rag_system)


if __name__ == "__main__":
    main()