# 목차
# 0. 필요 라이브러리 불러오기 및 환경 세팅
# 1. 환경 설정 및 상수 정의
# 2. STATE : RAG State 및 초기화 함수
# 3. NODES : LangGraph 노드 함수 정의
# 4. 조건 분기 함수 구축
# 5. 랭그래프 구축
# 6. 메인 실행 블록

import os
import warnings
from pathlib import Path
from typing import List, Literal, TypedDict
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.retrievers import TavilySearchAPIRetriever

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 경고 무시
warnings.filterwarnings("ignore")

# 환경 변수 로드
load_dotenv()

if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('TAVILY_API_KEY'):
    # Tavily는 웹 검색 노드에 필요합니다.
    # OPENAI_API_KEY가 없는 경우에만 에러를 발생시키는 원래 로직을 유지합니다.
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.')
    
# 모델 및 설정 상수
LLM_MODEL = 'gpt-4o-mini'
EMBEDDING_MODEL = 'text-embedding-3-small'
TEMPERATURE = 0.0
SIMILARITY_THRESHOLD = 0.6  # 문서 평가를 위한 유사도 임계값
INTERNAL_DOC_K = 8         # 내부 문서 검색 갯수
WEB_SEARCH_K = 5           # 웹 검색 문서 갯수

# 벡터 DB 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
PERSIST_DIR = PROJECT_ROOT / "01_data" / "vector_db"
COLLECTION_NAME = 'chroma_OpenAI_200_20'


# --- 2. RAG State 및 초기화 함수 ---

class RAGState(TypedDict):
    """
    LangGraph 상태를 정의하는 TypedDict입니다.
    """
    question: str
    documents: List[Document]
    doc_scores: List[float]
    search_type: str
    answer: str

def initialize_vectorstore():
    """
    Chroma 벡터 저장소를 초기화하고 반환합니다.
    """
    if not os.path.exists(PERSIST_DIR):
        raise ValueError(f'이전 단계에서 벡터 DB 디렉터리를 생성해야 합니다: {PERSIST_DIR}')

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model
    )
    return vectorstore, embedding_model


# --- 3. LangGraph 노드 함수 정의 ---

def retrieve_node(state: RAGState, vectorstore: Chroma) -> dict:
    """내부 문서 검색 노드: 하이브리드 검색을 수행합니다."""
    question = state['question']
    
    # 유사도 점수와 함께 문서 검색 (점수 높은 순서로 정렬됨)
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=INTERNAL_DOC_K)

    documents = [doc for doc, score in docs_with_scores]
    scores = [score for doc, score in docs_with_scores]

    print(f" [retriever] {len(documents)}개 문서 검색됨")
    return {'documents': documents, 'doc_scores': scores, 'search_type': 'internal'}

def grade_documents_node(state: RAGState) -> dict:
    """문서 평가 노드: 유사도 임계값을 기준으로 문서를 필터링합니다."""
    filtered_data = []
    
    # 임계값보다 높은 점수의 문서만 유지
    for doc, score in zip(state['documents'], state['doc_scores']):
        if score >= SIMILARITY_THRESHOLD:
            filtered_data.append((doc, score))

    final_documents = [item[0] for item in filtered_data]
    final_scores = [item[1] for item in filtered_data]

    print(f"[grade] {len(state['documents'])}개 --> {len(final_documents)}개 문서 유지 (임계값: {SIMILARITY_THRESHOLD:.2f})")
    return {'documents': final_documents, 'doc_scores': final_scores}

def web_search_node(state: RAGState) -> dict:
    """웹 검색 노드: Tavily를 사용하여 최신 웹 검색 결과를 가져옵니다."""
    
    # Tavily Retriever 초기화 및 검색 실행
    retriever = TavilySearchAPIRetriever(k=WEB_SEARCH_K)
    search_results: List[Document] = retriever.invoke(state['question'])

    processed_documents: List[Document] = []
    for i, doc in enumerate(search_results):
        # 웹 검색 결과 Document의 메타데이터를 통일성 있게 가공
        processed_doc = Document(
            page_content=doc.page_content, 
            metadata={
                'paper_name': doc.metadata.get('title', 'web_search_tavily_unknown'),
                'source': doc.metadata.get('source', 'web_search_tavily_unknown'), 
                'source_type': 'web',
                'index': i,
                'doc_score': doc.metadata.get('score', 0.0) # 점수가 없는 경우 기본값
            }
        )
        processed_documents.append(processed_doc)
        
    print(f" [web_search] {len(processed_documents)}개 웹 문서 검색됨")
    return {
        'documents': processed_documents, 
        'search_type': 'web'
    }

def generate_node(state: RAGState, llm: ChatOpenAI) -> dict:
    """생성 노드: 검색된 문맥과 질문을 기반으로 최종 답변을 생성합니다.""" 
    
    # 검색된 문서가 없으면 "NO_RELEVANT_PAPERS" 문맥 사용
    if not state['documents']:
        context = "NO_RELEVANT_PAPERS"
    else:
        context = '\n'.join([doc.page_content for doc in state['documents']])
    
    # --- 프롬프트 정의 (역할 및 출력 구조는 원본 유지) ---
    system_prompt = (
        "You are **\"AI Tech Trend Navigator\"**, an expert assistant for AI/ML research papers.\n\n"
        "[Role]\n"
        "- You help users understand and leverage recent AI/ML papers collected from HuggingFace DailyPapers.\n"
        "- Your main goals are:\n"
        "- Summarize and compare relevant papers clearly.\n"
        "- Explain core ideas in simple terms.\n"
        "- Highlight practical use-cases and implications for real-world services or products.\n\n"
        "[Context Handling]\n"
        "- If the context is **\"NO_RELEVANT_PAPERS\"**, answer purely from your general AI/ML knowledge.\n"
        "- Do NOT fabricate specific paper titles, authors, datasets, or numerical results.\n"
        "- If the context contains papers, prefer to base your answer on them.\n\n"
        "[Style]\n"
        "- Answer in the **SAME LANGUAGE** as the user's question (Korean).\n"
        "- Prefer clear, concise sentences.\n"
        "- Never fabricate paper titles, authors, datasets, or numerical results.\n"
    )

    human_prompt = (
        "[QUESTION]\n{question}\n\n"
        "[CONTEXT]\n======== START ========\n{context}\n======== END =========\n\n"
        "Please structure your answer as follows (flexible, but try to follow this):\n\n"
        "1) One-line summary \n"
        "2) Key insights (3-6 bullets) \n"
        "3) Related papers (top 1~3) \n"
        "4) Detailed explanation \n"
        "5) Sources summary\n\n"
        "⚠ Do not hallucinate papers or details not shown in context.\n"
        "Respond by Korean.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({'context': context, 'question': state['question']})
    
    return {'answer': answer}


# --- 4. 조건 분기 함수 구축 ---

def decide_to_generate(state: RAGState) -> Literal['generate', 'web_search']:
    """조건부 분기 함수: 내부 문서가 없으면 웹 검색으로, 있으면 생성으로 분기합니다."""
    
    if state['documents'] and len(state['documents']) > 0:
        print(f" [decide] {len(state['documents'])}개 문서 있음. -> generate")
        return 'generate'
    else: 
        print(f" [decide] 0개 문서 확인. (내부 문서 유사도 낮음) -> 웹 서칭을 합니다.")
        return 'web_search'

# --- 5. 랭그래프 구축 ---
def build_rag_graph():
    """LangGraph RAG 그래프를 구축하고 컴파일합니다."""
    # LLM 초기화
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    
    # 벡터스토어 초기화
    vectorstore, _ = initialize_vectorstore()

    # 노드 함수를 람다로 래핑하여 인자를 주입 (LangGraph의 노드 인자 타입 맞추기 위함)
    retriever_node_with_vs = lambda state: retrieve_node(state, vectorstore)
    generate_node_with_llm = lambda state: generate_node(state, llm)

    # 그래프 구축
    graph = StateGraph(RAGState)
    graph.add_node('retriever', retriever_node_with_vs)
    graph.add_node('grade', grade_documents_node)
    graph.add_node('web_search', web_search_node)
    graph.add_node('generate', generate_node_with_llm)

    # 엣지 정의
    graph.add_edge(START, 'retriever')
    graph.add_edge('retriever', 'grade')
    
    # 조건부 엣지 정의
    graph.add_conditional_edges(
        'grade',
        decide_to_generate,
        { 'generate': 'generate', 'web_search': 'web_search'}
    )
    
    graph.add_edge('web_search', 'generate')
    graph.add_edge('generate', END)

    # 그래프 컴파일
    app = graph.compile()
    
    print("\n[INFO] LangGraph RAG 앱 컴파일 완료.")
    
    # RAG 워크플로우 다이어그램 (추가 정보 가독성 증진을 위해 삽입)
    
    
    return app


# --- 6. 메인 실행 블록 ---

if __name__ == '__main__':
    try:
        # 컴파일된 LangGraph 앱을 가져오기
        rag_app = build_rag_graph()

        print("\n=== 챗봇 시작: AI Tech Trend Navigator ===")
        print(f" (LLM: {LLM_MODEL}, 임계값: {SIMILARITY_THRESHOLD})")
        print("종료하려면 'exit' 또는 'quit' 입력\n")

        while True:
            user_question = input("You: ")

            if user_question.lower() in ["exit", "quit"]:
                print("챗봇 종료!")
                break

            # RAGState 초기 상태 정의
            initial_state = RAGState(
                question=user_question,
                documents=[],
                doc_scores=[],
                search_type="",
                answer=""
            )

            # LangGraph 앱 실행
            # .invoke()를 사용하여 전체 워크플로우를 실행하고 최종 상태를 얻습니다.
            result = rag_app.invoke(initial_state)
            
            # 결과 출력
            answer = result['answer']
            search_type = result.get('search_type', 'N/A')
            doc_count = len(result.get('documents', []))

            print(f"\nAssistant: {answer}")
            print(f" (검색 유형: **{search_type}**, 참조 문서: **{doc_count}**개)\n")
            
    except ValueError as ve:
        print(f"\n[오류] 설정 오류: {ve}")
    except Exception as e:
        print(f"\n[오류] 예상치 못한 오류 발생: {e}")