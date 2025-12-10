import os
import warnings
import sys
from dotenv import load_dotenv

# 필수 라이브러리 로드
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from duckduckgo_search import DDGS  # pip install duckduckgo-search

from langsmith import Client, wrappers, evaluate
from openai import OpenAI

# 경고메세지 삭제
warnings.filterwarnings('ignore')
load_dotenv()

# openapi key 확인
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError('.env확인, key없음')

# vectordb 모듈 import
SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SRC_DIR / "02_utils"))
from vectordb import load_vectordb

client = Client()

dataset = client.create_dataset(
    dataset_name="ds-pertinent-fiesta-37", description="A sample dataset in LangSmith."
)
examples = [
    {
        "inputs": {"question": "ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration"},
        "outputs": {"answer": """title : ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration
huggingface_url : https://huggingface.co/papers/2511.21689
git_url : https://github.com/NVlabs/ToolOrchestra/
upvote:99
authors : Hongjin Su, Shizhe Diao, Ximing Lu"""},
    },
    {
        "inputs": {"question": "RFT를 LVLMs (large video language models) 으로 확장하는 방법은 무엇이 있나요?"},
        "outputs": {"answer": """title: VIDEOP2R: Video Understanding from Perception to Reasoning
huggingface_url:https://huggingface.co/papers/2511.11113
git_url: 없음
authors : Yifan Jiang, Yueying Wang, Rui Zhao, Toufiq Parag
upvote:111"""},
    },
    {
        "inputs": {"question": "LLM에서 긴 문맥의 추론을 향상시키는 GSW (Generative Semantic Workspace)에 대한 논문이 있다면 소개시켜주세요"},
        "outputs": {"answer": """title: Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces
hugginfFace_url: https://huggingface.co/papers/2511.07587
git_url: 없음
Authors: Shreyas Rajesh, Pavan Holur, Chenda Duan, David Chong
upvote:8"""}
    },
    {
        "inputs": {"question": "GUI-360: A Comprehensive Dataset and Benchmark for Computer-Using Agents"},
        "outputs": {"answer": """title: GUI-360: A Comprehensive Dataset and Benchmark for Computer-Using Agents
huggingface_url: https://huggingface.co/papers/2511.04307
git_url: 없음
authors: Jian Mu, Chaoyun Zhang, Chiming Ni, Lu Wang
upvote:14"""}
    },
    {
        "inputs": {"question": "오디오 기반 애니메이션의 정체성을 유지하는 방법이 있나요?"},
        "outputs": {"answer": """title: https://huggingface.co/papers/2510.23581
huggingface_url : https://huggingface.co/papers/2510.23581
git_url: 없음
authors : Junyoung Seo, Rodrigo Mira, Alexandros Haliassos
upvote:41"""}
    },
    {
        "inputs": {"question": "core attention disaggregation 은 무엇인가요?"},
        "outputs": {"answer": """title : Efficient Long-context Language Model Training by Core Attention Disaggregation
huggingface_url: https://huggingface.co/papers/2510.18121
git_url: 없음
authors:Yonghao Zhuang, Junda Chen, Bo Pang, Yi Gu
upvote:121"""}
    },
    {
        "inputs": {"question": "LLM에서 환각탐지를 할 수 있는 데이터셋을 알려주세요"},
        "outputs": {"answer": """title: When Models Lie, We Learn: Multilingual Span-Level Hallucination Detection with PsiloQA
huggingface_url : https://huggingface.co/papers/2510.04849
git_url : https://github.com/s-nlp/PsiloQA
authors : Elisei Rykov, Kseniia Petrushina, Maksim Savkin"""}
    },
    {
        "inputs": {"question": "LLM에서 캐시와 관련된 논문이 있나요?"},
        "outputs": {"answer": """title: Cache-to-Cache: Direct Semantic Communication Between Large Language Models
huggingface_url: https://huggingface.co/papers/2510.03215
git_url: https://github.com/thu-nics/C2C
authors: Tianyu Fu, Zihan Min, Hanling Zhang
upvote: 97"""}
    },
    {
        "inputs": {"question": "Adrian Kosowski 저자의 최근 논문이 있나요?"},
        "outputs": {"answer": """title: The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain
huggingface_url:https://huggingface.co/papers/2509.26507
git_url: https://github.com/pathwaycom/bdh
authors: Adrian Kosowski, Przemysław Uznański, Jan Chorowski
upvote:535"""}
    },
    {
        "inputs": {"question": "해리포터 줄거리 알려주세요"},
        "outputs": {"answer": """해당없다"""}
    },
    {
        "inputs": {"question": "최근 공개된 논문에서 좋아요수를 300개 이상 받은 논문은 몇개인가요? (5개 이하인경우, 5개 논문에 대해서 소개해주세요)"},
        "outputs": {"answer": ""}
    }

]
client.create_examples(dataset_id=dataset.id, examples=examples)

# Wrap the OpenAI client for LangSmith tracing
openai_client = wrappers.wrap_openai(OpenAI())


# Define the application logic to evaluate.
# Dataset inputs are automatically sent to this target function.
def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": response.choices[0].message.content}

# Define an LLM-as-a-judge evaluator to evaluate correctness of the output
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    LLM을 사용하여 답변의 정확성을 평가하는 함수
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    eval_prompt = f"""You are an expert evaluator. Compare the predicted answer with the reference answer.

Question: {inputs.get('question', '')}
Predicted Answer: {outputs.get('answer', '')}
Reference Answer: {reference_outputs.get('answer', '')}

Evaluate if the predicted answer is correct and relevant. Provide a score from 0 to 1 where:
- 1.0 means perfect match or fully correct
- 0.5 means partially correct
- 0.0 means completely incorrect

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}
"""

    try:
        response = llm.invoke(eval_prompt)
        import json
        result = json.loads(response.content)
        return {
            "key": "correctness",
            "score": result.get("score", 0.0),
            "comment": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"평가 오류: {e}")
        return {
            "key": "correctness",
            "score": 0.0,
            "comment": f"평가 실패: {str(e)}"
        }

class SimpleRAGSystem:
    '''간단한 RAG 시스템 래퍼 클래스'''
    def __init__(self, vectorstore, llm, retriever_k=3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever_k = retriever_k
        self.retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': retriever_k}
        )
        self.chain = self._build_chain()
        self.chat_history = []
    
    def _build_chain(self):
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
- context:
    - If there are relevant papers:
        → a concatenation of [Paper i] blocks.
    - If there are NO relevant papers:
        → a string that begins with the line EXACTLY:
        NO_RELEVANT_PAPERS
        followed by one or more [WebResult i] blocks from DuckDuckGo.

**IMPORTANT MODE SWITCH**
If the FIRST LINE of context is EXACTLY "NO_RELEVANT_PAPERS":
    → You MUST answer using ONLY general AI/ML knowledge + DuckDuckGo results.
    → You MUST output using strictly the <web search using> format.
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
- Use only relevant papers (1-3 typically).
- DO NOT hallucinate titles, authors, datasets, years, URLs, metrics, or numbers.
- If metadata items are missing, write "No information".
- If papers do not directly answer the user's question, explicitly say so.

Paper Output Format:
1) One-line summary
2) Key insights (3~6 bullets)
3) Matched papers (only title from md.get("title"))
4) Detailed explanation
5) Sources summary
   For each used paper:
     · title:
     · authors:
     · huggingface_url:
     · github_url:
     · upvote:

[WEB SEARCH MODE — detailed behavior]
Triggered ONLY when the FIRST LINE of context is EXACTLY "NO_RELEVANT_PAPERS".

Below that line, you will see one or more blocks like:
    [WebResult 1]
    title: ...
    url: ...
    snippet: ...
Use them as your ONLY external information.

Output MUST follow this EXACT structure:
1) One-line summary
2) Detailed explanation
3) source : DuckDuckgo
   - First URL
   - Second URL

URL RULES (VERY IMPORTANT):
- ONLY two URL use
- The URLs written under "source : DuckDuckgo" MUST be copied **exactly** from the `url:` fields.
- DO NOT invent or fabricate URLs.
- If only one valid URL exists, output only one URL line.
- If NO valid URLs exist, output:
    3) source : DuckDuckgo
       - (no URL available)

You MUST NOT:
- Output "Sources summary" in this mode
- Mention HuggingFace papers
- Output any paper titles

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
        """DuckDuckGo로 검색 (duckduckgo-search 라이브러리 사용)"""
        try:
            ddgs = DDGS()
            results = []
            
            # text() 메서드로 검색 수행
            search_results = ddgs.text(query, max_results=num_results)
            
            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", "")
                })
            
            return results
        except Exception as e:
            print(f"웹 검색 오류: {e}")
            return []

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

            title = md.get("title") or md.get("paper_name") or "No information"

            raw_authors = md.get("authors")
            if isinstance(raw_authors, list):
                authors = ", ".join(raw_authors)
            else:
                authors = raw_authors or "No information"

            title = md.get("title") or "No information"
            authors = md.get("authors") or "No information"
            huggingface_url = md.get("huggingface_url") or "No information"
            github_url = md.get("github_url") or "No information"
            upvote = md.get("upvote") or "No information"
            publication_year = md.get("publication_year") or "No information"
            doc_id = md.get("doc_id") or "No information"
            chunk_index = md.get("chunk_index") or "No information"

            block = f"""
[Paper {i}]
title: {title}
authors: {authors}
huggingface_url: {huggingface_url}
github_url: {github_url}
upvote: {upvote}
publication_year: {publication_year}
doc_id: {doc_id}
chunk_index: {chunk_index}

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

    def chat(self, user_message: str, score_threshold: float = 1.2) -> str:
        """
        대화 모드: 히스토리 저장 + RAG 답변
        
        Args:
            user_message: 사용자 질문
            score_threshold: 유사도 임계값 (낮을수록 유사함, 기본값 1.0)
        """
        # 1. similarity_search_with_score 수행
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            user_message, 
            k=self.retriever_k
        )
        
        # 디버깅용 출력
        print(f"\n[DEBUG] 검색된 문서 수: {len(docs_and_scores)}")
        for i, (doc, score) in enumerate(docs_and_scores):
            title = doc.metadata.get("title", "제목없음")
            print(f"  문서 {i+1}: score={score:.4f}, title={title}")

        # 2. score 기준 필터링 (낮을수록 유사 → <= 사용)
        relevant_docs = [doc for doc, score in docs_and_scores if score <= score_threshold]
        
        print(f"[DEBUG] 임계값 {score_threshold} 이하 문서: {len(relevant_docs)}개\n")

        # 3. context 결정
        if not relevant_docs:
            # 웹 검색 모드
            print("[INFO] 관련 논문 없음 → 웹 검색 모드 실행")
            web_results = self._web_search(user_message)
            web_block = self._format_web_results(web_results)
            context = "NO_RELEVANT_PAPERS\n\n" + web_block
        else:
            # 논문 모드
            print(f"[INFO] 논문 모드 실행 (관련 논문 {len(relevant_docs)}개)")
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

    def ask(self, question: str, score_threshold: float = 1.2) -> str:
        """
        질문에 답변 (단발성)
        
        Args:
            question: 질문 내용
            score_threshold: 유사도 임계값
        """
        # 1. 벡터DB 검색
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question,
            k=self.retriever_k
        )
        
        # 2. score 필터링
        relevant_docs = [doc for doc, score in docs_and_scores if score <= score_threshold]

        # 3. context 결정
        if not relevant_docs:
            web_results = self._web_search(question)
            web_block = self._format_web_results(web_results)
            context = "NO_RELEVANT_PAPERS\n\n" + web_block
        else:
            context = self._format_docs(relevant_docs)

        # 4. chain 실행
        return self.chain.invoke({
            "question": question,
            "context": context,
            "chat_history": self._format_chat_history()
        })

    def ask_with_sources(self, question: str, stream: bool = False, score_threshold: float = 1.2):
        """질문에 답변 + 출처 반환"""
        # 1. similarity_search_with_score 사용
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question,
            k=self.retriever_k
        )
        
        # 2. score 기준 필터링 (낮을수록 유사)
        relevant_docs = [doc for doc, score in docs_and_scores if score <= score_threshold]

        # 3. context 결정
        if not relevant_docs:
            # 웹 검색 모드
            web_results = self._web_search(question)
            web_block = self._format_web_results(web_results)
            context = "NO_RELEVANT_PAPERS\n\n" + web_block

            sources = [{
                "paper_name": r["title"],
                "huggingface_url": None,
                "github_url": None,
                "upvote": None,
                "url": r["url"]
            } for r in web_results]
        else:
            # 논문 모드
            context = self._format_docs(relevant_docs)
            sources = []
            for doc in relevant_docs:
                md = doc.metadata or {}
                title = md.get("title") or md.get("paper_name") or "(no title)"

                raw_authors = md.get("authors")
                if isinstance(raw_authors, list):
                    authors = ", ".join(raw_authors)
                else:
                    authors = raw_authors or "No information"

                sources.append({
                    "paper_name": title,
                    "authors": authors,
                    "huggingface_url": md.get("huggingface_url"),
                    "github_url": md.get("github_url"),
                    "upvote": md.get("upvote"),
                })

        # 4. chain 실행
        chain_input = {
            "question": question,
            "context": context,
            "chat_history": self._format_chat_history()
        }

        if stream:
            return {
                "answer_stream": self.chain.stream(chain_input),
                "sources": sources,
            }
        else:
            answer = self.chain.invoke(chain_input)
            return {
                "answer": answer,
                "sources": sources,
            }

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []


# if __name__ == '__main__':
#     # chunk 파일로 임시 확인
#     def get_project_root():
#         curr = Path().resolve()
#         for parent in [curr] + list(curr.parents):
#             if (parent / ".git").exists():
#                 return parent
#         raise FileNotFoundError("프로젝트 루트 찾기 실패")

#     MODEL_NAME = os.getenv("MODEL_NAME")
#     CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
#     CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
    
#     vectorstore = load_vectordb(
#             model_name=MODEL_NAME,
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#     )
  
#     llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

#     rag_system = SimpleRAGSystem(vectorstore, llm)
#     user_question = "해리포터 줄거리 알려줘"
#     result = rag_system.ask_with_sources(user_question)

#     print(f"질문: {user_question}")
#     print("\n[답변]\n")
#     print(result["answer"])

#     # --------------------------------------------------------
#     # 🔥 여기 아래 챗봇 모드 입력 루프 넣으면 됨!
#     # --------------------------------------------------------

#     print("\n=== AI Tech Trend Navigator Chatbot ===")
#     print("종료하려면 'exit' 또는 'quit' 입력\n")

#     while True:
#         user_msg = input("You: ")

#         if user_msg.lower() in ["exit", "quit"]:
#             print("챗봇 종료!")
#             break

#         answer = rag_system.chat(user_msg)
#         print(f"\nAssistant:\n{answer}\n")

if __name__ == '__main__':
    MODEL_NAME = os.getenv("MODEL_NAME")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

    print("VectorDB 로딩 중...")
    vectorstore = load_vectordb(
        model_name=MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    rag_system = SimpleRAGSystem(vectorstore, llm, retriever_k=5)

    print("RAG 시스템 초기화 완료\n")

    # target 함수를 RAG 시스템을 사용하도록 재정의
    def rag_target(inputs: dict) -> dict:
        """RAG 시스템을 사용하여 답변을 생성하는 target 함수"""
        question = inputs["question"]
        result = rag_system.ask(question, score_threshold=1.2)
        return {"answer": result}

    print("LangSmith 평가 실행 중...")
    try:
        experiment_results = evaluate(
            rag_target,  # RAG 시스템을 사용하는 함수
            data="ds-pertinent-fiesta-37",
            evaluators=[correctness_evaluator],
            experiment_prefix="experiment-rag-system",
            max_concurrency=2,
        )
        print("평가 완료!")
        print(f"실험 결과: {experiment_results}")
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        print("평가를 건너뛰고 챗봇 모드로 진행합니다.\n")

    # 테스트
    print("\n=== 테스트 1: 논문 검색 ===")
    result = rag_system.ask_with_sources("transformer architecture", score_threshold=1.2)
    print(f"\n[답변]\n{result['answer']}\n")
    print(f"[출처 수]: {len(result['sources'])}")

    print("\n=== AI Tech Trend Navigator Chatbot ===")
    print("종료하려면 'exit' 또는 'quit' 입력\n")

    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("챗봇 종료!")
            break

        # score_threshold 조정 가능 (기본값 1.2)
        answer = rag_system.chat(user_msg, score_threshold=1.2)
        print(f"\nAssistant:\n{answer}\n")