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
import hashlib
from pathlib import Path
from typing import List, Literal, TypedDict
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.retrievers import TavilySearchAPIRetriever, BM25Retriever

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 경고 무시
warnings.filterwarnings("ignore")

# --- 1. 환경 설정 및 상수 정의 ---

load_dotenv()

if not os.environ.get('OPENAI_API_KEY'):
    # OPENAI_API_KEY가 없는 경우 에러 발생
    raise ValueError('OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.')
    
# 모델 및 설정 상수
LLM_MODEL = 'gpt-4o-mini'
EMBEDDING_MODEL = 'text-embedding-3-small'
TEMPERATURE = 0.0

# 하이브리드 검색(Hybrid Search) 상수
VECTOR_SEARCH_K = 3         # 벡터 검색 결과 개수
BM25_SEARCH_K = 3           # BM25 검색 결과 개수
RRF_RANK_BIAS = 60          # RRF(Reciprocal Rank Fusion)의 랭크 편향 상수 (K)
RRF_GRADE_THRESHOLD = 0.018 # RRF 점수를 기준으로 문서를 필터링하는 임계값

# 웹 검색 상수
WEB_SEARCH_K = 5            # 웹 검색 결과 개수

# 벡터 DB 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
PERSIST_DIR = PROJECT_ROOT / "01_data" / "vector_db"
COLLECTION_NAME = 'chroma_OpenAI_200_20'


# --- 2. STATE : RAG State 및 초기화 함수 ---

class RAGState(TypedDict):
    """
    LangGraph 상태를 정의하는 TypedDict입니다.
    """
    question: str
    documents: List[Document]
    doc_scores: List[float]
    search_type: str
    answer: str

def initialize_retrievers():
    """
    Chroma 벡터 저장소와 BM25 Retriever를 초기화합니다.
    """
    if not os.path.exists(PERSIST_DIR):
        raise ValueError(f'이전 단계에서 벡터 DB 디렉터리를 생성해야 합니다: {PERSIST_DIR}')

    # 임베딩 모델
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # 1. Chroma Vectorstore 초기화
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model
    )
    
    # Vector Retriever (유사도 기반)
    vector_retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': VECTOR_SEARCH_K}
    )

    # 2. BM25 Retriever를 위한 전체 문서 로드
    print(" [BM25] BM25 Retriever를 위한 전체 문서 로드 및 인덱스 생성 시작...")
    collection_data = vectorstore._collection.get(include=['documents', 'metadatas'])
    
    all_documents = []
    for content, metadata in zip(collection_data['documents'], collection_data['metadatas']):
        all_documents.append(Document(page_content=content, metadata=metadata))
        
    if not all_documents:
        raise ValueError('Chroma DB에 문서가 없어 BM25 인덱스 생성이 불가합니다.')

    # BM25 Retriever 초기화 (전체 문서를 사용하여 인덱스 생성)
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = BM25_SEARCH_K
    print(" [BM25] BM25 인덱스 생성 완료.")
    
    return vector_retriever, bm25_retriever


def get_doc_hash_key(doc: Document) -> str:
    """
    Document 객체에 대한 고유 해시 키를 생성합니다. 
    내용(최대 1000자)과 출처를 결합하여 사용합니다.
    """
    content = doc.page_content[:1000] 
    source = doc.metadata.get('source', '')
    data_to_hash = f"{content}|{source}"
    return hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()


# --- 3. NODES : LangGraph 노드 함수 정의 ---

def retrieve_node(state: RAGState, vector_retriever, bm25_retriever) -> dict:
    """
    하이브리드 검색 노드 (RRF: Reciprocal Rank Fusion 수동 구현)
    - BM25 키워드 검색과 벡터 유사도 검색 결과를 병합합니다.
    """
    question = state['question']
    
    # 1. 검색 수행
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    fusion_scores = {}
    doc_map = {}
    
    # 2. RRF 점수 계산 및 문서 매핑
    
    # 2-1. 벡터 검색 결과 처리
    for rank, doc in enumerate(vector_docs):
        doc_key = get_doc_hash_key(doc)
        
        if doc_key not in doc_map:
            doc_map[doc_key] = doc
            
        # RRF 점수 계산: 1 / (K + rank + 1)
        score = 1 / (RRF_RANK_BIAS + rank + 1)
        fusion_scores[doc_key] = fusion_scores.get(doc_key, 0) + score
            
    # 2-2. BM25 검색 결과 처리
    for rank, doc in enumerate(bm25_docs):
        doc_key = get_doc_hash_key(doc)

        if doc_key not in doc_map:
            doc_map[doc_key] = doc
        
        # RRF 점수 계산
        score = 1 / (RRF_RANK_BIAS + rank + 1)
        fusion_scores[doc_key] = fusion_scores.get(doc_key, 0) + score
            
    # 3. 점수로 정렬 및 Document 객체 추출
    sorted_items = sorted(
        fusion_scores.items(), key=lambda x : x[1], reverse=True
    )

    # 결과 준비 (상위 문서만 사용)
    docs = []
    scores = []
    # RRF는 순위를 기반으로 점수를 매기므로, 상위 3개 문서만 사용하도록 제한
    for doc_key, score in sorted_items: # 제한 없이 전체 RRF 결과를 grade 노드로 보냄
        docs.append(doc_map[doc_key]) 
        scores.append(score)

    print(f" [retrieve] 하이브리드 검색 결과 {len(docs)}개 문서와 점수 추출 완료.")
    
    return {
        'documents': docs,
        'doc_scores': scores,
        'search_type': 'hybrid'
    }

def grade_documents_node(state: RAGState) -> dict:
    """
    문서 평가 노드: RRF 점수 임계값을 기준으로 문서를 필터링합니다.
    """
    filtered_data = []
    # RRF 점수는 일반적인 유사도 점수와 다르므로 별도 임계값 사용
    threshold = RRF_GRADE_THRESHOLD
    
    for doc, score in zip(state['documents'], state['doc_scores']):
        if score > threshold:
            filtered_data.append((doc, score))

    # 문서와 점수를 다시 분리
    final_documents = [item[0] for item in filtered_data]
    final_scores = [item[1] for item in filtered_data]

    print(f"[grade] {len(state['documents'])}개 --> {len(final_documents)}개 문서 유지 (RRF 임계값: {threshold})")
    return {'documents': final_documents, 'doc_scores': final_scores}

def web_search_node(state: RAGState) -> dict:
    """
    웹 검색 노드: Tavily를 사용하여 질문에 대한 최신 웹 검색 결과를 가져옵니다.
    """
    # Tavily Retriever 생성 및 초기화
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
                'doc_score': doc.metadata.get('score', 0.0)
            }
        )
        processed_documents.append(processed_doc)
    
    print(f" [web_search] {len(processed_documents)}개 웹 문서 검색됨")
    return {
        'documents': processed_documents, 
        'search_type': 'web'
    }

def generate_node(state: RAGState, llm: ChatOpenAI) -> dict:
    """
    생성 노드: 검색된 문맥과 질문을 기반으로 최종 답변을 생성합니다.
    """ 
    # 검색된 문서가 없으면 "NO_RELEVANT_PAPERS" 문맥 사용
    if not state['documents']:
        context = "NO_RELEVANT_PAPERS"
    else:
        context = '\n'.join([doc.page_content for doc in state['documents']])
        
    prompt = ChatPromptTemplate.from_messages([
        ('system', """
        You are **"AI Tech Trend Navigator"**, an expert assistant for AI/ML research papers.

        [Role]
        - You help users understand and leverage recent AI/ML papers.
        - Your main goals are to summarize, compare, and explain core ideas simply.
        - You must rely only on the given context and general AI/ML knowledge. Do NOT invent specific paper details.

        [Context Handling]
        - If the context is **"NO_RELEVANT_PAPERS"**, answer purely from your general AI/ML knowledge.
        - If context contains papers, base your answer on them.

        [Style]
        - Answer in the **SAME LANGUAGE** as the user's question (Korean).
        - Prefer clear, concise sentences.
        - Never fabricate paper titles, authors, datasets, or numerical results.
        """),
        ('human', f"""
        [QUESTION]
        {state['question']}
        
        [CONTEXT]
        ======== START ========
        {context}
        ======== END =========

        Please structure your answer as follows (flexible, but try to follow this):

        1) One-line summary 
        2) Key insights (3-6 bullets) 
        3) Related papers (top 1~3) 
        4) Detailed explanation 
        5) Sources summary

        ⚠ Do not hallucinate papers or details not shown in context.
        Respond by Korean.
        """)
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({'context': context, 'question': state['question']})
    
    return {'answer': answer}


# --- 4. 조건 분기 함수 구축 ---

def decide_to_generate(state: RAGState) -> Literal['generate', 'web_search']:
    """
    조건부 분기 함수: 내부 문서가 없으면 웹 검색으로, 있으면 생성으로 분기합니다.
    """ 
    if state['documents'] and len(state['documents']) > 0:
        print(f" [decide] {len(state['documents'])}개 문서 있음. -> generate")
        return 'generate'
    else: 
        print(f" [decide] 0개 문서 확인. (내부 문서 유사도 낮음) -> 웹 서칭을 합니다.")
        return 'web_search'


# --- 5. 랭그래프 구축 ---

def langgraph_rag():
    """
    LangGraph RAG 그래프를 구축하고 컴파일합니다.
    """
    # LLM 초기화
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    
    # Retriever 초기화
    vector_retriever, bm25_retriever = initialize_retrievers()

    # 노드 함수를 람다로 래핑하여 인자를 주입 (LangGraph 노드 인자 타입 맞추기 위함)
    retriever_node_with_r = lambda state: retrieve_node(state, vector_retriever, bm25_retriever)
    generate_node_with_llm = lambda state: generate_node(state, llm)

    # 그래프 구축 
    graph = StateGraph(RAGState)
    graph.add_node('retriever', retriever_node_with_r)
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
    
    return app


# --- 6. 메인 실행 블록 ---

if __name__ == '__main__':
    try:
        # 컴파일된 LangGraph 앱을 가져오기
        rag_app = langgraph_rag()

        print("\n=== 챗봇 시작: AI Tech Trend Navigator (Hybrid RAG) ===")
        print(f" (LLM: {LLM_MODEL}, RRF K: {RRF_RANK_BIAS}, Grade Threshold: {RRF_GRADE_THRESHOLD})")
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