# pip install langgraph
# pip install langchain-community
# pip install tavily-python
from langgraph.graph import StateGraph, START, END# 필요라이브러리 설치
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Literal
from typing_extensions import TypedDict
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

# 환경설정
load_dotenv()

if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError('key check')

def langgraph_rag():
    # llm모델 초기화 및 생성
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)

    ## State 클래스 정의
    class RAGState(TypedDict):
        question:str
        documents : List[Document]
        doc_scores : List[float]
        search_type : str
        answer : str

    # 임베딩 모델 생성
    embedding_model = OpenAIEmbeddings(
        model = 'text-embedding-3-small'
    )

    # 벡터 DB가져오기
    project_root = Path(__file__).resolve().parent.parent.parent 
    persist_dir = project_root / "01_data" / "vector_db"
    if os.path.exists(persist_dir):
        vectorstore = Chroma(
            persist_directory = persist_dir,
            collection_name = 'persistent_rag',
            embedding_function = embedding_model
        )
    else:
        raise ValueError('이전단계 chroma_db_reg2 디렉터리 생성 필요')

    ## Node 함수 + 조건 분기 함수 생성
        # 검색 노드
        # 문서평가 노드
        # 웹검색 노드
        # 생성 노드
        # 조건 분기 함수
    def retrieve_node(state:RAGState)->dict:
        '''내부문서 검색 노드'''
        question = state['question']
        docs_with_scores = vectorstore.similarity_search_with_score(question, k = 3)
        documents =  [ doc for doc,score in docs_with_scores]
        scores =  [ 1-score for doc,score in docs_with_scores]

        print(f' [retriever] {len(documents)}개 문서 검색됨')        
        return {'documents': documents, 'doc_scores':scores, 'search_type':'internal'}   # state 업데이트

    def grade_documents_node(state:RAGState)->dict:
        '''문서평가 노드'''
        threshold = 0.3
        filtered_docs, filtered_scores = [],[]
        for doc, score in zip(state['documents'],state['doc_scores']):
            if score >= threshold:
                filtered_docs.append(doc); filtered_scores.append(score)
        print(f"[grade] {len(state['documents'])}개 --> {len(filtered_docs)}개 문서 유지")
        return {'documents' : filtered_docs, 'doc_scores':filtered_scores}

    def web_search_node(state: dict) -> dict:
        '''웹검색 노드: Tavily를 사용하여 질문에 대한 최신 웹 검색 결과를 가져옵니다.'''
        
        # Tavily Retriever 생성 및 초기화
        retriever = TavilySearchAPIRetriever()
        
        # 웹 검색 실행 (state['question'] 사용)
        # Tavily Retriever는 일반적으로 LangChain의 Document 객체 리스트를 반환합니다.
        search_results: List[Document] = retriever.invoke(state['question'])
        
        # 검색 결과를 처리하고 메타데이터 추가
        processed_documents: List[Document] = []
        
        for i, doc in enumerate(search_results):
            # 검색 결과 Document의 내용을 확인하고 출처(source)를 추가합니다.
            # LangChain Document는 'metadata' 속성에 출처(source) 정보(예: URL)가 이미 포함되어 있을 수 있습니다.
            
            # 새로운 Document 객체 생성 (RAG 시스템에 맞게 metadata를 통일시킵니다)
            web_doc = Document(
                page_content=doc.page_content, 
                metadata={
                    'source': doc.metadata.get('source', 'web_search_tavily_unknown'), # 기존 source가 있다면 사용
                    'search_type': 'web',
                    'doc_score': 0.8, # 모든 웹 검색 결과에 임의의 점수 0.8 부여
                    'index': i # 결과 순서
                }
            )
            processed_documents.append(web_doc)
        
        # 4. 결과 반환
        # 검색결과를 Document 객체 리스트를 반환합니다.
        # doc_scores는 Document 수에 맞게 리스트로 생성합니다.
        doc_scores = [doc.metadata['doc_score'] for doc in processed_documents]
        
        return {
            'documents': processed_documents,
            'doc_scores': doc_scores,
            'search_type': 'web'
        }

    def generate_node(state:RAGState)->dict:
        '''생성노드'''  
        context = '\n'.join([ doc.page_content for doc in state['documents']])
        prompt = ChatPromptTemplate.from_messages([
            ('system','Answer in Korean based on the provided context.'),
            ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'context':context, 'question' : state['question'] })
        return {'answer':answer}
    
    def decide_to_generate(state:RAGState)-> Literal['generate','web_search']:
        '''조건부 분기 함수 : 문서내에 참고할 내용이 없다면 web으로 검색한다.'''    
        if state['documents'] and len(state['documents']) > 0:
            return 'generate'
        else:
            return 'web_search'

    # 그래프 구축(add_node  add_edge  add_conditional_edges)
    graph = StateGraph(RAGState)
    graph.add_node('retriever',retrieve_node)
    graph.add_node('grade',grade_documents_node)
    graph.add_node('web_search',web_search_node)
    graph.add_node('generate',generate_node)

    graph.add_edge(START, 'retriever')
    graph.add_edge('retriever', 'grade')
    graph.add_conditional_edges(
        'grade',
        decide_to_generate,
        { 'generate':'generate', 'web_search': 'web_search'}
    )
    graph.add_edge('web_search', 'generate')
    graph.add_edge('generate', END)

    # 그래프 컴파일
    app = graph.compile()
    return app

if __name__ == '__main__':
    # 컴파일된 LangGraph 앱을 가져오기
    rag_app = langgraph_rag()

    print("\n=== AI Tech Trend Navigator Chatbot ===")
    print("종료하려면 'exit' 또는 'quit' 입력\n")

    while True:
        try:
            user_question = input("You: ")

            if user_question.lower() in ["exit", "quit"]:
                print("챗봇 종료!")
                break

            # RAGState 초기 상태 정의 (매번 새 질문으로 처리)
            initial_state = {
                'question': user_question,
                'documents': [],
                'doc_scores': [],
                'search_type': "",
                'answer': ""
            }

            # LangGraph 앱 실행
            result = rag_app.invoke(initial_state)
            
            # 결과 출력
            answer = result['answer']
            search_type = result.get('search_type', 'N/A')
            doc_count = len(result.get('documents', []))

            print(f"\nAssistant: {answer}")
            print(f" (검색유형: {search_type}, 참조문서: {doc_count}개)\n")
            
        except Exception as e:
            print(f"\n오류 발생: {e}\n")