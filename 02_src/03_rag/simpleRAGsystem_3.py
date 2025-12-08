import os
import warnings
import sys
from dotenv import load_dotenv

# 필수 라이브러리 로드
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import requests


# 경고메세지 삭제
warnings.filterwarnings('ignore')
load_dotenv()

# openapi key 확인
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError('.env확인,  key없음')

# vectordb 모듈 import
SRC_DIR=Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR / "02_utils"))
from vectordb import load_vectordb


class SimpleRAGSystem:
    '''간단한 RAG 시스템 래퍼 클래스'''
    def __init__(self, vectorstore, llm, retriever_k=3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_type = 'similarity', search_kwargs={'k':retriever_k})
        self.chain = self._build_chain()
        self.chat_history = []
    

    def _build_chain(self): ### ---------> 최종 사용자에게 전달되는 프롬프트 수정
        '''RAG 체인 구성''' 
        
        prompt = ChatPromptTemplate.from_messages([
                ("system",     """
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
      - metadata(Extract the following items from the VectorDB file and display them to the user.):
        - title
        - github_url (optional)
        - huggingface_url
        - upvote (integer)
        - tags
        - authors
        - publication_year
        - total_chunks
        - doc_id
        - chunk_index

    You must rely only on:
    - the given context, and
    - general, high-level AI/ML knowledge.
    
    Do NOT invent specific paper titles, authors, datasets, metrics, or numerical results
    that are not supported by the context.

    - In case of "NO_RELEVANT_PAPERS", mark it as "HuggingFace paper: None" and enter the content searched on the web using duckduckgo.
        - One-line summary
        - Detailed explanation
        - source : DuckDuckgo (Two URL links related to the answer)


    [Context Handling]

    - If the context is **"NO_RELEVANT_PAPERS"**, it means:
    - The retrieval system could not find any clearly relevant papers.
    - In this case, Search and answer user questions on duckduckgo from general AI/ML knowledge.
    - Do NOT fabricate specific paper titles, authors, datasets, or numerical results.
    - You may skip the "Related papers" section or keep it generic.
    - WebResults are NOT research papers and MUST NOT appear in the "Sources summary". 
    - WebResults must NOT override or replace papers when papers are present.

    - If the context contains one or more papers:
    - Treat these as the primary and most reliable information source.
    - Use only the papers that are reasonably related to the user's question.
    - If some papers are weakly related, ignore them.
    - If nothing seems clearly relevant, say that the context does not directly answer the question.




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
       - Focus on the most relevant 1-3 papers in the given context.
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
    Each paper in the context may contain:
    - page_content (text)
    - metadata: paper_name, tags, upvote, github_url, huggingface_url, etc.

    [CONTEXT]
    ======== START ========
    {context}
    ======== END =========

    Please structure your answer as follows (flexible, but try to follow this):
     

    - If there is a paper that is suitable for the user's question,

        1) One-line summary
        2) Key insights (3-6 bullets)
        3) Related papers (top 1~3) : only title
        4) Detailed explanation
        5) Sources summary
        - Organize the papers used above **based on metadata**
        - Output each paper in the following format:
            · title: (title or “No information") 
            · Authors: (authors if available, “No information” otherwise)
            · huggingface_url: (huggingface_url or “No information”)
            · github_url: (github_url (optional) or “No information”)
            · upvote: (upvote (integer) or “No information”)
            · tags: (tags;separated by commas, “No information” if none)

        ⚠ For information not present in the metadata (e.g., if authors are missing), DO NOT invent it; instead, write “No information available.”
        ⚠ Do not hallucinate papers or details not shown in context.
        ⚠ Regardless of the input language, ALWAYS respond in Korean.
     

     
    - If you do not find a paper that matches your question (In case of "NO_RELEVANT_PAPERS") and use duckduckgo web search,
     
        1) One-line summary
        2) Detailed explanation
        3) source : DuckDuckgo (Two URL links related to the answer)
        
        ⚠ Do not hallucinate papers or details not shown in context.
        ⚠ Regardless of the input language, ALWAYS respond in Korean.


    """)
            ])
        return (
            prompt
            | self.llm
            | StrOutputParser()
        )
    
    
    def _web_search(self, query: str, num_results: int = 5):
        """DuckDuckGo API로 검색"""
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
        }
        res = requests.get(url, params=params)

        if res.status_code != 200:
            return []

        data = res.json()
        results = []

        # DuckDuckGo는 주요 검색 결과가 'RelatedTopics'에 들어감
        for item in data.get("RelatedTopics", []):
            if "Text" in item:
                results.append({
                    "title": item.get("Text", ""),
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", "")
                })

            # 일부는 내부 topics 형태로 들어있을 수 있음
            if "Topics" in item:
                for t in item["Topics"]:
                    results.append({
                        "title": t.get("Text", ""),
                        "url": t.get("FirstURL", ""),
                        "snippet": t.get("Text", "")
                    })

        return results[:num_results]

    def _format_web_results(self, results):
        """검색 결과 → LLM 프롬프트용 텍스트"""
        if not results:
            return "NO_WEB_RESULTS"

        blocks = []
        for i, r in enumerate(results, 1):
            blocks.append(f"""
[WebResult {i}]
title: {r['title']}
url: {r['url']}

snippet:
{r['snippet']}
""")
        return "\n\n".join(blocks)
    



    @staticmethod
    def _format_docs(docs):
        """retriever가 반환한 Document들을 프롬프트용 텍스트로 변환"""
        if not docs:
            return "NO_RELEVANT_PAPERS"

        lines = []

        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}

            block = f"""
    [Paper {i}]
    title: {md.get("title", "No information")}
    authors: {md.get("authors", "No information")}
    huggingface_url: {md.get("huggingface_url", "No information")}
    github_url: {md.get("github_url", "No information")}
    upvote: {md.get("upvote", "No information")}
    tags: {md.get("tags", "No information")}
    publication_year: {md.get("publication_year", "No information")}
    doc_id: {md.get("doc_id", "No information")}
    chunk_index: {md.get("chunk_index", "No information")}
    

    content:
    {doc.page_content}
    """

            lines.append(block)

        return "\n\n".join(lines)



    def _format_chat_history(self):
        """저장된 대화 리스트를 하나의 문자열로 구성"""
        if not self.chat_history:
            return "(no previous conversation)"

        history_lines = []
        for turn in self.chat_history:
            history_lines.append(f"User: {turn['user']}")
            history_lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(history_lines)

    def chat(self, user_message: str) -> str:
        """대화 모드: 히스토리 저장 + RAG 답변 (최적화: retrieval 1회만 수행)"""
        # retrieval 수행
        source_docs = self.retriever.invoke(user_message)
        if not source_docs:
            web_results = self._web_search(user_message)
            context = self._format_web_results(web_results)
        else:
            context = self._format_docs(source_docs)

        response = self.chain.invoke({
            "question": user_message,
            "context": context,
            "chat_history": self._format_chat_history()
        })

        self.chat_history.append({
            "user": user_message,
            "assistant": response
        })


        # chain 실행
        response = self.chain.invoke({
            "question": user_message,
            "context": context,
            "chat_history": self._format_chat_history()
        })

        # 히스토리에 저장
        self.chat_history.append({
            "user": user_message,
            "assistant": response
        })

        return response



    def ask(self, question: str) -> str:
        '''질문에 답변 (최적화: retrieval 1회만 수행)'''
        # 1) 벡터DB 검색. retrieval 수행
        source_docs = self.retriever.invoke(question)
        context = self._format_docs(source_docs)
        
        # 2) context 결정
        if not source_docs:
            # --- 내부 문서 없음 → 웹 검색 ---
            web_results = self._web_search(question)
            context = self._format_web_results(web_results)
        else:
            context = self._format_docs(source_docs)

        # 3) chain 실행
        return self.chain.invoke({
            "question": question,
            "context": context,
            "chat_history": self._format_chat_history()
        })

    

    def ask_with_sources(self, question: str, stream: bool = False):
        """질문에 답변 + 출처 반환 (최적화: retrieval 1회만 수행)

        Args:
            question: 사용자 질문
            stream: True이면 스트리밍 응답 생성기 반환, False이면 전체 답변 반환

        Returns:
            stream=False: {"answer": str, "sources": list}
            stream=True: {"answer_stream": generator, "sources": list}
        """
        # 1. retrieval을 한 번만 수행
        source_docs = self.retriever.invoke(question)

        if not source_docs:
            # 내부문서 없음 → 웹으로 대체
            web_results = self._web_search(question)
            context = self._format_web_results(web_results)

            # 웹은 metadata가 없으므로 sources에도 웹 정보 담아줌
            sources = [{
                "paper_name": r["title"],
                "huggingface_url": None,
                "github_url": None,
                "upvote": None,
                "tags": [],
                "url": r["url"]
            } for r in web_results]

        else:
            # DB가 있으면 기존 로직
            context = self._format_docs(source_docs)
            # 3. 출처 정보 구성
            sources = []
            for doc in source_docs:
                md = doc.metadata or {}
                tags = [
                    md.get("tag1"),
                    md.get("tag2"),
                    md.get("tag3"),
                ]
                tags = [t for t in tags if t]

                sources.append(
                    {
                        "paper_name": md.get("paper_name", "(no title)"),
                        "huggingface_url": md.get("huggingface_url"),
                        "github_url": md.get("github_url"),
                        "upvote": md.get("upvote"),
                        "tags": tags,
                    }
                )

        # 4. chain 실행 (스트리밍 or 일반)
        chain_input = {
            "question": question,
            "context": context,
            "chat_history": self._format_chat_history()
        }

        if stream:
            # 스트리밍 응답 생성기 반환
            return {
                "answer_stream": self.chain.stream(chain_input),
                "sources": sources,
            }
        else:
            # 전체 답변 반환
            answer = self.chain.invoke(chain_input)
            return {
                "answer": answer,
                "sources": sources,
            }

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []
   


if __name__ == '__main__':
    # chunk 파일로 임시 확인
    def get_project_root():
        curr = Path().resolve()
        for parent in [curr] + list(curr.parents):
            if (parent / ".git").exists():
                return parent
        raise FileNotFoundError("프로젝트 루트 찾기 실패")

    MODEL_NAME = os.getenv("MODEL_NAME")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
    KEYWORD_METHOD = os.getenv("KEYWORD_EXTRACTION_METHOD").lower()
    
    vectorstore = load_vectordb(
            model_name=MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            method=KEYWORD_METHOD
    )
  
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    rag_system = SimpleRAGSystem(vectorstore, llm)
    user_question = "벡터DB가 뭐야?"
    result = rag_system.ask_with_sources(user_question)

    print(f"질문: {user_question}")
    print("\n[답변]\n")
    print(result["answer"])

    print("\n[출처]\n")
    for i, src in enumerate(result["sources"], start=1):
        print(f"- [{i}] {src['paper_name']}")
        if src["huggingface_url"]:
            print(f"  HF: {src['huggingface_url']}")
        if src["github_url"]:
            print(f"  GitHub: {src['github_url']}")
        if src["tags"]:
            print(f"  tags: {', '.join(src['tags'])}")
        if src["upvote"] is not None:
            print(f"  upvote: {src['upvote']}")


    # --------------------------------------------------------
    # 🔥 여기 아래 챗봇 모드 입력 루프 넣으면 됨!
    # --------------------------------------------------------

    print("\n=== AI Tech Trend Navigator Chatbot ===")
    print("종료하려면 'exit' 또는 'quit' 입력\n")

    while True:
        user_msg = input("You: ")

        if user_msg.lower() in ["exit", "quit"]:
            print("챗봇 종료!")
            break

        answer = rag_system.chat(user_msg)
        print(f"\nAssistant:\n{answer}\n")