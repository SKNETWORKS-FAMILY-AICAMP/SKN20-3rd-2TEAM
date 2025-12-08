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

# 저장경로 설정
current_folder = Path(__file__).resolve().parent
project_root = current_folder.parent.parent 
chunks_path = project_root / "01_data" / "chunks" / "chunks_all.pkl"

# 저장된 피클 데이터 가져오기
# chunks_path = Path(r".\01_data\chunks\chunks_all.pkl")
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent.parent 
persist_dir = project_root / "01_data" / "vector_db"

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