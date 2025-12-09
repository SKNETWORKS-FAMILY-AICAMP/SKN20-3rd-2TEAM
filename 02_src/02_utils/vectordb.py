"""
VectorDB 관리 모듈

이 모듈은 문서 청크를 임베딩하여 Chroma 벡터 데이터베이스에 저장하고 로드하는 기능을 제공합니다.
다양한 임베딩 모델을 지원하며, OpenAI와 HuggingFace 모델을 사용할 수 있습니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import torch
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chunking import load_chunks_from_pkl
from clustering import cluster_documents

# 환경 변수 로드
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "OpenAI")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 100))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 10))


# 저장경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "01_data" / "chunks"
VECTORDB_DIR = PROJECT_ROOT / "01_data" / "vector_db"

# 지원하는 임베딩 모델 딕셔너리
# 키: 모델 별칭, 값: HuggingFace 모델명 또는 OpenAI 모델명
embedding_models = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "MsMarco": "sentence-transformers/msmarco-MiniLM-L-6-v3",
    "SPECTER": "sentence-transformers/allenai-specter",
    "OpenAI": "text-embedding-3-small",
    "BGE-M3": "BAAI/bge-m3",
    "Paraphrase-Multi": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}


def vectordb_save(model_name: str, chunk_size: int = 100, chunk_overlap: int = 10, use_weeks: int = 10):
    """
    문서 청크를 임베딩하여 Chroma 벡터 데이터베이스에 저장합니다.
    클러스터링을 수행하여 각 청크에 cluster_id를 추가합니다.

    Args:
        model_name (str): 사용할 임베딩 모델의 이름 (embedding_models 딕셔너리의 키)
        chunk_size (int, optional): 청크 크기. 기본값은 100
        chunk_overlap (int, optional): 청크 간 겹치는 크기. 기본값은 10
        use_weeks (int, optional): 클러스터링에 사용할 주차 수. 기본값은 10

    Returns:
        None

    Raises:
        ValueError: 임베딩 모델이 지정되지 않았을 때
        Exception: 모델 로딩 실패 시
    """
    # pkl 파일에서 문서 청크 로드
    documents = load_chunks_from_pkl(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 클러스터링 수행 (문서 단위)
    print("\n" + "=" * 60)
    print("[CLUSTERING] 문서 클러스터링 수행 중...")
    print("=" * 60)
    doc_cluster_map = cluster_documents(use_weeks=use_weeks)
    print(f"[CLUSTERING] {len(doc_cluster_map)}개 문서에 대한 클러스터 매핑 완료")

    # 메타데이터 전처리 및 cluster_id 추가
    chunks = []
    for doc in documents:
        metadata = doc.metadata.copy()

        # List로 되어있는 metadata를 쉼표와 공백으로 구분 된 문자열로 변환
        metadata["authors"] = ", ".join(metadata["authors"])
        metadata["tags"] = ", ".join(metadata["tags"])

        # doc_id를 기반으로 cluster_id 추가
        doc_id = metadata.get("doc_id", "")
        if doc_id in doc_cluster_map:
            metadata["cluster_id"] = doc_cluster_map[doc_id]
        else:
            metadata["cluster_id"] = -1  # 클러스터 매핑 없는 경우

        chunks.append(Document(page_content=doc.page_content, metadata=metadata))

    # 벡터 DB 저장 디렉토리 생성
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    print(f"  Directory ready: {VECTORDB_DIR}")

    # 임베딩 모델 초기화
    if model_name == "OpenAI":
        model = OpenAIEmbeddings(model=embedding_models[model_name])
        print(f"[LOADING] {model_name} 로딩 완료")
    else:
        try:
            # HuggingFace 임베딩 모델 로드 (CUDA 사용 가능 시 GPU 사용)
            model = HuggingFaceEmbeddings(
                model_name=embedding_models[model_name],
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print(f"[LOADING] {model_name} 로딩 완료")
        except Exception as e:
            print(f"[FAILED] {model_name} 로딩 실패: {e}")
            return

    # VectorStore 생성 및 저장
    if model:
        collection_name = f"chroma_{model_name}_{chunk_size}_{chunk_overlap}_C"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            collection_name=collection_name,
            embedding=model,
            persist_directory=VECTORDB_DIR,
        )
        print(f"[SUCCESS] vectordb '{collection_name}' 저장 완료")
        print(f"[INFO] 총 {len(chunks)}개 청크 저장 (cluster_id 포함)")
    else:
        raise ValueError("embedding_model이 지정되지 선택되지 않았습니다.")


def load_vectordb(model_name: str, chunk_size: int = 100, chunk_overlap: int = 10) -> Chroma:
    """
    저장된 Chroma 벡터 데이터베이스를 로드하고 샘플 데이터를 출력합니다.

    Args:
        model_name (str): 사용할 임베딩 모델의 이름 (embedding_models 딕셔너리의 키)
        chunk_size (int): 청크 크기. 기본값=100
        chunk_overlap (int): 청크 간 겹치는 크기. 기본값=10

    Returns:
        Chroma: 로드된 Chroma 벡터스토어 객체

    Raises:
        ValueError: 벡터 DB 디렉토리가 존재하지 않을 때
    """
    # 벡터 DB 디렉토리 존재 확인
    if not os.path.exists(VECTORDB_DIR):
        raise ValueError("[ERROR] 폴더가 존재하지 않음")

    # 임베딩 모델 초기화
    if model_name == "OpenAI":
        embedding_function = OpenAIEmbeddings(model=embedding_models[model_name])
        print(f"[LOADING] {model_name} 로딩 완료")
    else:
        try:
            # HuggingFace 임베딩 모델 로드 (CUDA 사용 가능 시 GPU 사용)
            embedding_function = HuggingFaceEmbeddings(
                model_name=embedding_models[model_name],
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print(f"[LOADING] {model_name} 로딩 완료")
        except Exception as e:
            print(f"[FAILED] {model_name} 로딩 실패: {e}")
            raise

    # Chroma 벡터스토어 로드
    collection_name = f"chroma_{model_name}_{chunk_size}_{chunk_overlap}_C"
    vectorstore = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )
    print("[SUCCESS] VectorDB 로딩 완료\n")

    # 컬렉션에서 문서와 메타데이터 가져오기
    docs = vectorstore._collection.get(include=["documents", "metadatas"])
    documents = docs["documents"]
    metadatas = docs["metadatas"]

    # 처음 5개의 문서와 메타데이터 출력 (샘플 확인용)
    if __name__ == "__main__":
        for doc, meta in zip(documents[:5], metadatas[:5]):
            print(f"Doc: {doc[:50]}\nMeta: {meta}")

    return vectorstore


if __name__ == "__main__":
    # 메인 실행 블록: 벡터 DB 생성 및 로드 테스트

    # 임베딩을 사용하여 벡터 DB 저장 (클러스터링 포함)
    vectordb_save(MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, use_weeks=10)

    # 저장된 벡터 DB 로드 및 확인
    load_vectordb(MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)
    