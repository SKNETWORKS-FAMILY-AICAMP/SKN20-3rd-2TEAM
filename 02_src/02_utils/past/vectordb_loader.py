from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

embedding_model = OpenAIEmbeddings(
    model = 'text-embedding-3-small'
)

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

# 확인겸, 저장된 모든 문서 로드
all_docs = vectorstore._collection.get(include=['documents', 'metadatas'])
documents = all_docs['documents']
metadatas = all_docs['metadatas']

print("문서 내용 예시:")
for doc in documents[:5]:  # 처음 5개만
    print(doc)

print("메타데이터 예시:")
for meta in metadatas[:5]:
    print(meta)
