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
            collection_name = 'chroma_OpenAI_200_20',
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
        '''내부문서 검색 노드 : 하이브리드 검색'''
        question = state['question']
        docs_with_scores = vectorstore.similarity_search_with_score(question, k = 8)

        # similarity_search_with_score 함수가 반환하는 docs_with_scores 결과는 점수(score)가 높은 순서대로 정렬
        documents =  [ doc for doc,score in docs_with_scores]
        scores =  [ score for doc,score in docs_with_scores]

        print(f' [retriever] {len(documents)}개 문서 검색됨')        
        return {'documents': documents, 'doc_scores':scores, 'search_type':'internal'}   # state 업데이트

    def grade_documents_node(state:RAGState)->dict:
        '''문서평가 노드'''
        threshold = 0.7 ### 하이퍼파라미터
        filtered_data = []
        for doc, score in zip(state['documents'],state['doc_scores']):
            if score >= threshold:
                filtered_data.append((doc, score))

        # 문서와 점수를 다시 분리
        final_documents = [item[0] for item in filtered_data]
        final_scores = [item[1] for item in filtered_data]

        print(f"[grade] {len(state['documents'])}개 --> {len(final_documents)}개 문서 유지")
        return {'documents': final_documents, 'doc_scores': final_scores}
    
    def web_search_node(state: dict) -> dict:
        '''웹검색 노드: Tavily를 사용하여 질문에 대한 최신 웹 검색 결과를 가져옵니다.'''
        
        # Tavily Retriever 생성 및 초기화, 검색 문서 갯수 5개 (k : 5) 설정
        retriever = TavilySearchAPIRetriever(k=5)
        
        # 웹 검색 실행
        # Tavily Retriever는 LangChain의 Document 객체 리스트를 반환
        # tavily 웹검색 반환값 : <class 'langchain_core.documents.base.Document'>
        # page_content
        # source
        # score
        # images
        
        # 주어진 질문에서 Tavily Retriever 웹검색 실행
        search_results: List[Document] = retriever.invoke(state['question'])

        # 처리한 검색결과 를 저장할 객체 생성
        processed_documents: List[Document] = []
        for i, doc in enumerate(search_results):
            # 검색 결과 Document의 내용을 확인하고 출처(source)를 추가합니다.
            # LangChain Document는 'metadata' 속성에 출처(source) 정보(예: URL)가 이미 포함되어 있다. (101 줄 참고)
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
                    'doc_score': doc_score
                }
            )
            processed_documents.append(web_doc)
        
        return {
            'documents': processed_documents, 
            'search_type': 'web'
        }
    
    def generate_node(state:RAGState)->dict: ### 하이퍼파라미터 영역 : 프롬프트 부분
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
        """), 
            ('human', 'context:\n{context}\n\nquestion:{question}\n\nanswer:')
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'context':context, 'question' : state['question']})
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