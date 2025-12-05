from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

# 임베딩 모델 생성
embedding_model = OpenAIEmbeddings(
    model = 'text-embedding-3-small'
)

# 저장된 피클 데이터 가져오기
chunks_path = Path(r".\01_data\chunks\chunks_all.pkl")
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

# 프로젝트 루트 기준 상대경로
persist_dir = os.path.join("01_data", "vector_db")

# 디렉토리가 없으면 생성
os.makedirs(persist_dir, exist_ok=True)  # exist_ok=True면 이미 있어도 에러 안 남
print(f"Directory ready: {persist_dir}")

# VectorStore 영구 저장
vectorstore_persistent = Chroma.from_documents(
    documents=chunks,
    collection_name='persistent_rag',
    embedding=embedding_model,
    persist_directory=persist_dir
)