import os
import warnings
import pickle    # chunk, vectorDB 저장한것 사용
from dotenv import load_dotenv

# 경고메세지 삭제
warnings.filterwarnings('ignore')
load_dotenv()

# openapi key 확인
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('.env확인,  key없음')

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from pathlib import Path


class SimpleRAGSystem:
    '''간단한 RAG 시스템 래퍼 클래스'''
    def __init__(self, vectorstore, llm, retriever_k=3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_type = 'similarity', search_kwargs={'k':retriever_k})
        # self.retriever_chain = self._retriever_basic_chain()
        self.chain = self._build_chain()
    

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
    """)
            ])
        return (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: ""
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    
    @staticmethod
    def _format_docs(docs):
        """retriever가 반환한 Document들을 프롬프트용 텍스트로 변환"""
        if not docs:
            # ⚠️ 컨텍스트가 전혀 없을 때는 이 문자열로 보냄
            return "NO_RELEVANT_PAPERS"

        lines = []
        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}

            # tag1, tag2, tag3 → tags 리스트로 재구성
            tags = [
                md.get("tag1"),
                md.get("tag2"),
                md.get("tag3"),
            ]
            tags = [t for t in tags if t]  # None/빈 값 제거

            tag_str = f"tags: {', '.join(tags)}" if tags else "tags: (none)"

            paper_name = md.get("paper_name", "(no title)")
            hf_url = md.get("huggingface_url", "")
            gh_url = md.get("github_url", "")

            link_lines = []
            if hf_url:
                link_lines.append(f"HuggingFace: {hf_url}")
            if gh_url:
                link_lines.append(f"GitHub: {gh_url}")
            links_block = "\n".join(link_lines) if link_lines else ""

            block = f"""[{i}] {paper_name}
                                {tag_str}
                                {links_block}

                                {doc.page_content}"""
            lines.append(block)

        return "\n\n".join(lines)


    def ask(self, question:str) -> str:
        '''질문에 답변'''
        return self.chain.invoke(question)
    

    def ask_with_sources(self, question: str) -> dict:
        """질문에 답변 + 출처 반환"""
        answer = self.chain.invoke(question)
        source_docs = self.retriever.invoke(question)

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

        return {
            "answer": answer,
            "sources": sources,
        }
   


if __name__ == '__main__' :
    # chunk 파일로 임시 확인
    def get_project_root():
        curr = Path().resolve()
        for parent in [curr] + list(curr.parents):
            if (parent / ".git").exists():
                return parent
        raise FileNotFoundError("프로젝트 루트 찾기 실패")

    PROJECT_ROOT = get_project_root()
    DATA_DIR = PROJECT_ROOT / "01_data/chunks"

    chunks_path = DATA_DIR / "chunks_all.pkl"

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name='test',
        embedding=OpenAIEmbeddings(model='text-embedding-3-small')
    )

  
    llm = ChatOpenAI( model = 'gpt-4o-mini', temperature=0 )

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