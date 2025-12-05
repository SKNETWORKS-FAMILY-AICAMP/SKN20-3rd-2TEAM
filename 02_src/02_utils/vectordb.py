# pip install chromadb
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.config import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from tqdm import tqdm

from embedding import load_embeddings

# Chroma 설정
# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "01_data"

CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "rag-collection"
BATCH_SIZE = 100  # Chroma upsert 배치 크기

def create_chroma_client():
    """Chroma 클라이언트 생성"""
    print(f"Chroma DB 경로: {CHROMA_DB_PATH}")
    
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    return client

def create_or_get_collection(client):
    """Chroma 컬렉션 생성 또는 가져오기"""
    
    # 기존 컬렉션 확인
    existing_collections = [col.name for col in client.list_collections()]
    
    if COLLECTION_NAME in existing_collections:
        print(f"기존 컬렉션 사용: {COLLECTION_NAME}")
        # 기존 컬렉션 삭제 여부 확인
        user_input = input("기존 컬렉션을 삭제하고 새로 만드시겠습니까? (y/n): ")
        if user_input.lower() == 'y':
            client.delete_collection(name=COLLECTION_NAME)
            print("기존 컬렉션 삭제 완료")
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"새 컬렉션 생성: {COLLECTION_NAME}")
        else:
            collection = client.get_collection(name=COLLECTION_NAME)
    else:
        print(f"새 컬렉션 생성: {COLLECTION_NAME}")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
        )
    
    return collection

def upload_to_chroma(embedded_chunks, collection):
    """임베딩된 청크를 Chroma에 업로드"""
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    print("\n데이터 준비 중...")
    for i, chunk in enumerate(tqdm(embedded_chunks, desc="데이터 변환 중")):
        ids.append(f"chunk_{i}")
        embeddings.append(chunk["vector"])
        documents.append(chunk["text"])
        
        # 메타데이터 준비 (Chroma는 문자열, 숫자, 불린만 지원)
        metadata = {}
        for key, value in chunk["metadata"].items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)  # 다른 타입은 문자열로 변환
        
        metadatas.append(metadata)
    
    # 배치 단위로 업로드
    print("\nChroma DB에 업로드 중...")
    total_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(ids), BATCH_SIZE), total=total_batches, desc="업로드 진행"):
        batch_end = min(i + BATCH_SIZE, len(ids))
        
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    print(f"\n총 {len(embedded_chunks)}개의 벡터가 업로드되었습니다.")
    
    # 컬렉션 통계 확인
    print(f"컬렉션 총 문서 수: {collection.count()}")


def test_search(collection, query_text="AI", n_results=3):
    """검색 테스트"""
    
    
    print("\n" + "=" * 60)
    print("검색 테스트")
    print("=" * 60)
    print(f"검색어: {query_text}")
    
    # 쿼리 임베딩 생성
    embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
    query_vector = embedding_model.get_text_embedding(query_text)
    
    # 검색 수행
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"\n상위 {n_results}개 검색 결과:")
    print("-" * 60)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n[결과 {i+1}]")
        print(f"유사도 점수: {1 - distance:.4f}")  # 거리를 유사도로 변환
        print(f"메타데이터: {metadata}")
        print(f"내용: {doc[:200]}...")
        print("-" * 60)


def main():
    """벡터DB 저장 메인 함수"""
    
    print("=" * 60)
    print("임베딩 결과 로드")
    print("=" * 60)
    embedded_chunks = load_embeddings("embeddings.pkl")
    
    print("\n" + "=" * 60)
    print("Chroma 클라이언트 생성")
    print("=" * 60)
    client = create_chroma_client()
    
    print("\n" + "=" * 60)
    print("Chroma 컬렉션 생성/연결")
    print("=" * 60)
    collection = create_or_get_collection(client)
    
    print("\n" + "=" * 60)
    print("벡터 업로드")
    print("=" * 60)
    upload_to_chroma(embedded_chunks, collection)
    
    # 검색 테스트 (선택사항)
    test_search(collection)
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"Chroma DB 저장 경로: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()