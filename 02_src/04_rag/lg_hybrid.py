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

# 하이브리드 검색 관련 임포트
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever

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
    
    # 2. BM25 Retriever를 위한 전체 문서 준비 (추가된 부분)
    # Chroma에서 모든 문서를 가져옵니다. (BM25 인덱스 생성용)
    print(" [BM25] BM25 Retriever를 위한 전체 문서 로드 및 인덱스 생성 시작...")
    
    # Chroma._collection.get()을 사용하여 Document 리스트를 직접 구성합니다.
    collection_data = vectorstore._collection.get(include=['documents', 'metadatas'])
    
    # LangChain Document 객체 리스트로 변환
    all_documents = []
    for content, metadata in zip(collection_data['documents'], collection_data['metadatas']):
        all_documents.append(Document(page_content=content, metadata=metadata))
        
    if not all_documents:
        raise ValueError('Chroma DB에 문서가 없습니다. BM25 인덱스 생성이 불가합니다.')

    # BM25 Retriever 초기화 (전체 문서를 사용하여 인덱스 생성)
    # **주의:** 문서가 많으면 이 단계에서 메모리를 많이 사용하고 시간이 오래 걸립니다.
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 3 # BM25 검색 결과 개수 설정
    
    # 리트리버
    retriever = vectorstore.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k' : 3}
    )
    
    ## Node 함수 + 조건 분기 함수 생성
        # 검색 노드
        # 문서평가 노드
        # 웹검색 노드
        # 생성 노드
        # 조건 분기 함수
    def retrieve_node(state:RAGState)->dict:
        '''
        하이브리드 검색 노드 (RRF: Reciprocal Rank Fusion 수동 구현)
        - BM25 키워드 검색과 벡터 유사도 검색 결과를 병합합니다.
        '''
        question = state['question']
        
        # 백터 검색
        vector_docs = retriever.invoke(question)
        # BM25 검색
        bm25_docs = bm25_retriever.invoke(question)
        fusion_scores = {}
        # 백터 검색 결과 점수
        for rank, doc in enumerate(vector_docs):
            doc_key = doc.page_content[:50]
            score = 1 / (60 + rank)
            fusion_scores[doc_key] = fusion_scores.get(doc_key,0) + score
        # BM25 검색 결과 점수
        for rank, doc in enumerate(bm25_docs):
            doc_key = doc.page_content[:50]
            score = 1 / (60 + rank)
            fusion_scores[doc_key] = fusion_scores.get(doc_key,0) + score
        
        # 점수로 정렬
        sorted_docs =  sorted(
            fusion_scores.items(), key=lambda x : x[1], reverse=True
        )

        # print(f'fusion docs 결과 상위 3개 : {sorted_docs[:3]}')
        docs = []
        scores =[]
        for doc,score in sorted_docs[:3]:
            docs.append(doc)
            scores.append(score)

        return {
            'documents': docs, 
            'doc_scores': scores,
        }

    def grade_documents_node(state:RAGState)->dict:
        '''문서평가 노드'''
        threshold = 0.3
        filtered_data = []
        for doc, score in zip(state['documents'],state['doc_scores']):
            if score >= threshold:
                filtered_data.append((doc, score))

        # .similarity_search_with_score 에서 반환하는 값이 높은 점수 순으로 반환되지않는 것을 확인 :
            #  높은 점수순으로 문서정렬 로직 추가하는 코드
        # sorted() 함수를 사용하여 두 번째 요소(score, 인덱스 1)를 기준으로 내림차순(reverse=True) 정렬
        sorted_filtered_data = sorted(
            filtered_data,
            key=lambda item: item[1], # item[1]은 score
            reverse=True              # 내림차순 정렬 (높은 점수부터)
        )

        # 문서와 점수를 다시 분리
        filtered_docs = [doc for doc, score in sorted_filtered_data]
        filtered_scores = [score for doc, score in sorted_filtered_data]

        print(f"[grade] {len(state['documents'])}개 --> {len(filtered_docs)}개 문서 유지 (점수 내림차순 정렬 완료)") ### 출력 조정
        return {'documents': filtered_docs, 'doc_scores': filtered_scores}
    
##
    def web_search_node(state: dict) -> dict:
        '''웹검색 노드: Tavily를 사용하여 질문에 대한 최신 웹 검색 결과를 가져옵니다.'''
        
        # Tavily Retriever 생성 및 초기화, 검색 문서 갯수 k : 5
        retriever = TavilySearchAPIRetriever(k=5)
        
        # 웹 검색 실행 (state['question'] 사용)
        # Tavily Retriever는 일반적으로 LangChain의 Document 객체 리스트를 반환
        search_results: List[Document] = retriever.invoke(state['question'])
        # tavily 웹검색 반환값 : <class 'langchain_core.documents.base.Document'>
            # page_content
            # source
            # score
            # images
        
        # 검색 결과를 처리하고 메타데이터 추가
        processed_documents: List[Document] = []

        for i, doc in enumerate(search_results):
            # 검색 결과 Document의 내용을 확인하고 출처(source)를 추가합니다.
            # LangChain Document는 'metadata' 속성에 출처(source) 정보(예: URL)가 이미 포함되어 있습니다.
            source_url = doc.metadata.get('source', 'web_search_tavily_unknown')
            paper_name = doc.metadata.get('title', 'web_search_tavily_unknown')
            doc_score = doc.metadata.get('score', 'web_search_tavily_unknown')
            # 새로운 Document 객체 생성 및 메타데이터 형식 통일
            web_doc = Document(
                page_content=doc.page_content, 
                metadata={
                    # 내부 문서와 유사한 키를 사용하거나, 새 키를 정의
                    'paper_name': paper_name,
                    'source': source_url, 
                    
                    # 추가적인 식별 키
                    'source_type': 'web',
                    'index': i,
                    'doc_score': doc_score,
                    
                    
                    # 내부 문서에는 없지만, 나중에 필요할 경우를 대비해 None으로 초기화
                    'upvote': None,
                    'tags': [],
                    'year': None
                }
            )
        processed_documents.append(web_doc)
        
        return {
            'documents': processed_documents, 
            'search_type': 'web'
        }
    
    def generate_node(state:RAGState)->dict: ##
        '''생성노드'''  
        context = '\n'.join([ doc.page_content for doc in state['documents']])
        prompt = ChatPromptTemplate.from_messages([
            ('system','Answer in Korean based on the provided context.'), 
            ('human', 'context:\n{context}\n\nquestion:{question}\n\nanswer:')
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'context':context, 'question' : state['question'] })
        return {'answer':answer}

    def decide_to_generate(state:RAGState)-> Literal['generate','web_search']:
        '''조건부 분기 함수 : 문서내에 참고할 내용이 없다면 web으로 검색한다.'''    
        # if state['documents'] and len(state['documents']) > 0:
        #     return 'generate'
        # else:
        #     return 'web_search'
        if state['documents'] and len(state['documents']) > 0:
            print(f" [decide] {len(state['documents'])}개 문서 있음. -> generate")
            return 'generate'
        else:
            print(f" [decide] 0개 문서 확인. (내부 문서 유사도 낮음) -> 웹 서칭을 합니다.")
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