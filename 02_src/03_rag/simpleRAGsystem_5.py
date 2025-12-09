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
                ("system", """
You are **"AI Tech Trend Navigator"**, an expert assistant for AI/ML research papers.

[Role]
- You help users understand and leverage recent AI/ML papers collected from HuggingFace Weekly Papers.
- Your goals:
  - Summarize papers clearly.
  - Explain core ideas simply.
  - Highlight practical use-cases for real products/services.

[Inputs Provided]
The system supplies:
- user_question
- chat_history
- context (either a set of papers OR EXACT STRING: "NO_RELEVANT_PAPERS")

**IMPORTANT MODE SWITCH**
If context == "NO_RELEVANT_PAPERS":
    → You MUST answer using ONLY general AI/ML knowledge + DuckDuckGo results.
    → You MUST output using strictly the <web search useing> format.
    → You MUST NOT output:
        - "Sources summary"
        - Any paper list
        - Any paper title
        - ANY reference to HuggingFace papers or metadata

If context contains papers:
    → You MUST ignore DuckDuckGo rules COMPLETELY.
    → You MUST answer ONLY based on:
         (1) given context papers
         (2) general high-level knowledge (no invented details)
    → You MUST output using the "paper mode" format:
         1) One-line summary
         2) Key insights
         3) Related papers
         4) Detailed explanation
         5) Sources summary (based strictly on metadata)

[PAPER MODE — detailed behavior]
- Use only relevant papers (1–3 typically).
- DO NOT hallucinate titles, authors, datasets, years, URLs, metrics, or numbers.
- If metadata items are missing, write “No information”.
- If papers do not directly answer the user’s question, explicitly say so.

Paper Output Format:
1) One-line summary
2) Key insights (3~6 bullets)
3) Related papers (only titles)
4) Detailed explanation
5) Sources summary
   For each used paper:
     · title:
     · authors:
     · huggingface_url:
     · github_url:
     · upvote:

[WEB SEARCH MODE — detailed behavior]
Triggered ONLY when context == "NO_RELEVANT_PAPERS".

Output MUST follow this EXACT structure:
1) One-line summary
2) Detailed explanation
3) source : DuckDuckgo
   - First URL
   - Second URL

RULES FOR WEB SEARCH MODE:
- NEVER output “Sources summary”
- NEVER output a paper title
- NEVER mention HuggingFace papers
- Treat results as general information, not research papers.

[Style]
- ALWAYS respond in Korean.
- Keep explanations clear and non-academic.
- Briefly explain technical terms when helpful.
"""),


("human", """
[QUESTION]
{question}

[CHAT HISTORY]
{chat_history}

[Context]
======= START =======
{context}
======= END =======

Follow the output rules based on whether papers exist.
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
        """대화 모드: 히스토리 저장 + RAG 답변 (similarity score 후처리)"""
        # 1. similarity_search_with_score 수행
        docs_and_scores = self.vectorstore.similarity_search_with_score(user_message, k=5)

        # 2. score 기준 필터링
        score_threshold = 0.7
        relevant_docs = [doc for doc, score in docs_and_scores if score >= score_threshold]

        # 3. context 결정
        if not relevant_docs:
            context = "NO_RELEVANT_PAPERS"
            web_results = self._web_search(user_message)
            context = self._format_web_results(web_results)
        else:
            context = self._format_docs(relevant_docs)

        # 4. chain 실행
        response = self.chain.invoke({
            "question": user_message,
            "context": context,
            "chat_history": self._format_chat_history()
        })

        # 5. 히스토리 저장
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

    

    def ask_with_sources(self, question: str, stream: bool = False, score_threshold: float = 0.7):
        """질문에 답변 + 출처 반환 (score 후처리 방식)"""
        # 1. similarity_search_with_score 사용
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=5)
        
        # 2. score 기준 필터링
        relevant_docs = [doc for doc, score in docs_and_scores if score >= score_threshold]

        # 3. context 결정
        if not relevant_docs:
            context = "NO_RELEVANT_PAPERS"
            web_results = self._web_search(question)
            context = self._format_web_results(web_results)
            
            sources = [{
                "paper_name": r["title"],
                "huggingface_url": None,
                "github_url": None,
                "upvote": None,
                "url": r["url"]
            } for r in web_results]

        else:
            context = self._format_docs(relevant_docs)
            sources = []
            for doc in relevant_docs:
                md = doc.metadata or {}
                tags = [md.get("tag1"), md.get("tag2"), md.get("tag3")]
                tags = [t for t in tags if t]
                sources.append({
                    "paper_name": md.get("title", "(no title)"),
                    "huggingface_url": md.get("huggingface_url"),
                    "github_url": md.get("github_url"),
                    "upvote": md.get("upvote"),
                })


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
    
    vectorstore = load_vectordb(
            model_name=MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
    )
  
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    rag_system = SimpleRAGSystem(vectorstore, llm)
    user_question = "벡터DB가 뭐야?"
    result = rag_system.ask_with_sources(user_question)

    print(f"질문: {user_question}")
    print("\n[답변]\n")
    print(result["answer"])

    # print("\n[출처]\n")
    # for i, src in enumerate(result["sources"], start=1):
    #     print(f"- [{i}] {src['paper_name']}")
    #     if src["huggingface_url"]:
    #         print(f"  HF: {src['huggingface_url']}")
    #     if src["github_url"]:
    #         print(f"  GitHub: {src['github_url']}")
    #     if src["upvote"] is not None:
    #         print(f"  upvote: {src['upvote']}")


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