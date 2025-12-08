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
import hashlib
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
from langchain_community.retrievers import BM25Retriever

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
        chat_history: str

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

    # 하단 노드함수에 들어갈 고유키 생성 함수 : vector_docs에 고유키로 사용할만한 id항목이 있지만, 무조건 하나의 문서에 id가 있도록 할겸
                                           # content와 source를 사용해 고유키를 생성하고 키 통합
    def get_doc_hash_key(doc: Document) -> str:
        # 내용(최대 1000자)과 출처를 결합하여 고유 해시 생성
        content = doc.page_content[:1000] 
        source = doc.metadata.get('source', '')
        data_to_hash = f"{content}|{source}"
        return hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()
    
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

        # 벡터 검색 결과
        # Document객체
        # id = 고유id 존재
        # metadata = {'source': ' '}
        # page_content = 내용

        # BM25 검색 결과
        # metadata = {'source': ' '}
        # page_content = 내용

        fusion_scores = {}
        doc_map = {}
        # RRF 점수 계산 및 문서 매핑
        RRF_K = 60
        # RRF_RANK_BIAS: RRF 순위 편향(K) 상수
        # 이 상수가 클수록 순위가 낮은 문서(rank)도 최종 점수에 더 큰 기여를 합니다.
        # 일반적으로 60이 권장되지만, 튜닝이 필요할 수 있습니다.
        
        # 1. 벡터 검색 결과 처리
        for rank, doc in enumerate(vector_docs):
            # 고유키 지정 : 벡터 검색 결과 Document 내부 id 사용
            doc_key = get_doc_hash_key(doc)
            
            # 키와 Document 객체 매핑 저장
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            score = 1 / (RRF_K + rank +1) # rank는 0부터 시작하므로 +1
            fusion_scores[doc_key] = fusion_scores.get(doc_key, 0) + score
            
        # 2. BM25 검색 결과 처리
        for rank, doc in enumerate(bm25_docs):
            # 고유키 지정 : 벡터 검색 결과 Document 내부 id 사용
            doc_key = get_doc_hash_key(doc)

            # 키와 Document 객체 매핑 저장
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            
            score = 1 / (RRF_K + rank +1)
            fusion_scores[doc_key] = fusion_scores.get(doc_key, 0) + score
            
        # 3. 점수로 정렬 및 Document 객체 추출
        sorted_items = sorted(
            fusion_scores.items(), key=lambda x : x[1], reverse=True
        )

        # 상위 3개 문서의 Document 객체와 점수 추출
        docs = []
        scores =[]
        for doc_key, score in sorted_items[:3]:
            # doc_map을 사용하여 Document 객체를 찾아와 docs 리스트에 추가
            # 점수 정규화
            normalized_score = score
            docs.append(doc_map[doc_key]) 
            scores.append(normalized_score)

        print(f" [retrieve] 하이브리드 검색 결과 {len(docs)}개 문서와 점수 추출 완료.")
        
        return {
            'documents': docs,
            'doc_scores': scores,
            'search_type' : 'hybrid'
        }

    def grade_documents_node(state:RAGState)->dict:
        '''문서평가 노드'''
        threshold = 0.16
        filtered_data = []
        for doc, score in zip(state['documents'],state['doc_scores']):
            if score >= threshold:
                filtered_data.append((doc, score))

        # 문서와 점수를 다시 분리
        filtered_docs = [doc for doc, score in filtered_data]
        filtered_scores = [score for doc, score in filtered_data]

        print(f"[grade] {len(state['documents'])}개 --> {len(filtered_docs)}개 문서 유지 (임계값 필터링 완료)") ### 출력 나오게
        return {'documents': filtered_docs, 'doc_scores': filtered_scores}
    
    def web_search_node(state: dict) -> dict:
        '''웹검색 노드: Tavily를 사용하여 질문에 대한 최신 웹 검색 결과를 가져옵니다.'''
        
        # Tavily Retriever 생성 및 초기화, 검색 문서 갯수 k : 3
        retriever = TavilySearchAPIRetriever(k=3)
        
        # 웹 검색 실행 (state['question'] 사용)TRivia
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
                    
                    
                    # 내부 문서에는 해당내용이 없지만, 나중에 필요할 경우를 대비해 None으로 초기화
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
            ('system',"""
        You are **"AI Tech Trend Navigator"**, an expert assistant for AI/ML research papers.

        [Role]
        - You help users understand and leverage recent AI/ML papers collected from HuggingFace DailyPapers.
        - Your main goals are:
        - Summarize and compare relevant papers clearly.
        - Explain core ideas in simple terms.
        - Highlight practical use-cases and implications for real-world services or products.

        [Inputs]
        The system provides:
        - user_question: the user’s question.
        - context: a set of retrieved documents, formatted as a single text block.
            - Sometimes the context may be exactly the string "NO_RELEVANT_PAPERS".
        - page_content: main text (abstract or summary)
        - metadata:
            - paper_name
            - github_url (optional)
            - huggingface_url (optional)
            - upvote (integer, popularity signal)
            - tags: list of keywords
            - year, week, and other fields.

        You must rely only on:
        - the given context, and
        - general, high-level AI/ML knowledge.
        Do NOT invent specific paper titles, authors, datasets, metrics, or numerical results
        that are not supported by the context.


        [Context Handling]
        - If the context is **"NO_RELEVANT_PAPERS"**, it means:
        - The retrieval system could not find any clearly relevant papers.
        - In this case, you may answer **purely from your own general AI/ML knowledge**.
        - Do NOT fabricate specific paper titles, authors, datasets, or numerical results.
        - You may skip the "Related papers" section or keep it very generic.

        - If the context contains one or more papers:
        - Prefer to base your answer on those papers.
        - Use only the papers that are reasonably related to the user’s question.
                    
        [Main Tasks]

        1. Understand the user’s intent
        - Roughly classify the question as one of:
            - (a) concept/background explanation
            - (b) single-paper summary
            - (c) comparison or trend analysis across multiple papers
            - (d) practical application and use-case ideas
        - If the intent is ambiguous, make a reasonable assumption and continue.
            You may briefly state what you assumed.

        2. Use only the relevant papers
        - Focus on the most relevant 1–3 papers in the given context.
        - If some papers look only weakly related to the question, you may ignore them.
        - If nothing is clearly relevant, say that the context does not directly answer the question.

        3. Summarize each selected paper
        For each paper you rely on, briefly cover:
        - What problem it tries to solve.
        - What approach/model/idea it uses.
        - What seems new or strong compared to typical or baseline methods.
        - Any obvious limitations, trade-offs, or caveats that are visible from the context.

        4. Produce a synthesized answer
        - Do not just list papers. Synthesize them to directly answer the user’s question.
        - When possible, cover:
            - Common themes or trends across the papers.
            - How these ideas relate to topics such as RAG, long-context, multimodal models, etc.,
            when relevant.
            - How someone could apply these ideas in a real-world project, prototype, or product.

        5. Be honest about uncertainty
        - If the given context is not enough to answer precisely, say so.
        - Suggest what extra information, papers, or queries would be helpful.

        [Style]
        - Answer in the SAME LANGUAGE as the user’s question.
        (If the question is in Korean, answer in Korean. If it is in English, answer in English.)
        - Prefer clear, concise sentences over heavy academic wording.
        - Briefly explain technical terms when needed.
        - Never fabricate paper titles, authors, datasets, or numerical results.
        """),
                    
        ("human", """
        [QUESTION]
        {question}
        
        [CHAT HISTORY]
        {chat_history}

        [Context]
        The following CONTEXT block may contain 0 or more papers. 
        If it is "NO_RELEVANT_PAPERS", please answer from your general AI/ML knowledge.
        
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
        answer = chain.invoke({'context':context, 'question' : state['question'],'chat_history': state['chat_history']})
        return {'answer':answer}

    def decide_to_generate(state:RAGState)-> Literal['generate','web_search']:
        '''조건부 분기 함수 : 문서내에 참고할 내용이 없다면 web으로 검색한다.'''    
        if state['documents'] and len(state['documents']) > 0:
            print(f" [decide] {len(state['documents'])}개 문서 있음. -> generate")
            return 'generate'
        else: # 검색된 문서가 0개인경우, len(state['documents'] == 0
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

    full_chat_history = ""
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
                'answer': "",
                'chat_history': full_chat_history
            }

            # LangGraph 앱 실행
            result = rag_app.invoke(initial_state)
            
            # 결과 출력
            answer = result['answer']
            search_type = result.get('search_type', 'N/A')
            doc_count = len(result.get('documents', []))

            print(f"\nAssistant: {answer}")
            print(f" (검색유형: {search_type}, 참조문서: {doc_count}개)\n")

            # 현재 질문과 답변을 대화 기록에 누적
            full_chat_history += f"Human: {user_question}\n"
            full_chat_history += f"Assistant: {answer}\n"
        except Exception as e:
            print(f"\n오류 발생: {e}\n")