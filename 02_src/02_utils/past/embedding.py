# pip install llama-index-embeddings-openai
import os
from pathlib import Path
import pickle
from dotenv import load_dotenv

load_dotenv()

from llama_index.embeddings.openai import OpenAIEmbedding
from tqdm import tqdm

from chunking import load_chunks_from_pkl

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "01_data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHUNKS_DIR = DATA_DIR / "chunks"
DOCUMENTS_DIR = DATA_DIR / "documents"

# 파일 경로
CHUNKS_FILE = CHUNKS_DIR / "chunks_all.pkl"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.pkl"

BATCH_SIZE = 40   # 20~60 정도가 가장 안전하고 빠름

def embed_chunks_batch(chunks, embedding_model, batch_size=BATCH_SIZE):
    """청크를 배치 단위로 임베딩"""
    results = []
    batch_texts = []
    batch_metadata = []
    
    # tqdm으로 진행상황 표시
    for chunk in tqdm(chunks, desc="임베딩 진행 중"):
        batch_texts.append(chunk.page_content)
        batch_metadata.append(chunk.metadata)

        # 배치가 꽉 차면 내용 추가
        if len(batch_texts) == batch_size:
            try:
                vectors = embedding_model.get_text_embedding_batch(batch_texts)
                for text, meta, vec in zip(batch_texts, batch_metadata, vectors):
                    results.append({
                        "text": text,
                        "metadata": meta,
                        "vector": vec
                    })
                batch_texts, batch_metadata = [], []
            except Exception as e:
                print(f"\n배치 임베딩 중 오류 발생: {e}")
                # 실패한 배치는 건너뛰고 계속 진행
                batch_texts, batch_metadata = [], []
                continue

    # 마지막 남은 데이터 처리
    if batch_texts:
        try:
            vectors = embedding_model.get_text_embedding_batch(batch_texts)
            for text, meta, vec in zip(batch_texts, batch_metadata, vectors):
                results.append({
                    "text": text,
                    "metadata": meta,
                    "vector": vec
                })
        except Exception as e:
            print(f"\n마지막 배치 임베딩 중 오류 발생: {e}")

    return results


def save_embeddings(embedded_chunks, filename="embeddings.pkl"):
    """임베딩 결과를 pickle 파일로 저장"""
    # 디렉토리 생성
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = EMBEDDINGS_DIR / filename  # config에서 가져온 경로 사용
    with open(output_path, 'wb') as f:
        pickle.dump(embedded_chunks, f)
    print(f"\n임베딩 결과 저장 완료: {output_path}")
    print(f"총 {len(embedded_chunks)}개의 청크가 저장되었습니다.")


def load_embeddings(filename="embeddings.pkl"):
    """저장된 임베딩 결과를 로드"""
    input_path = EMBEDDINGS_DIR / filename  # config에서 가져온 경로 사용
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} 파일을 찾을 수 없습니다.")
    
    with open(input_path, 'rb') as f:
        embedded_chunks = pickle.load(f)
    print(f"임베딩 결과 로드 완료: {len(embedded_chunks)}개의 청크")
    return embedded_chunks


def embedding_function():
    """임베딩 실행 함수"""
    
    # 경로 정보 출력
    print("=" * 60)
    print("경로 정보")
    print("=" * 60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"임베딩 저장 경로: {EMBEDDINGS_DIR}")
    print(f"청크 파일 경로: {CHUNKS_FILE}")
    
    # API KEY 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    print(f"\nAPI KEY 확인: {api_key[:10]}...")

    print("\n" + "=" * 60)
    print("임베딩 모델 초기화")
    print("=" * 60)
    embedding_model = OpenAIEmbedding(model="text-embedding-3-small")

    print("\n" + "=" * 60)
    print("청크 파일 로드")
    print("=" * 60)
    chunks = load_chunks_from_pkl(str(CHUNKS_FILE))  # config 경로 사용
    print(f"총 {len(chunks)}개의 청크를 로드했습니다.")

    print("\n" + "=" * 60)
    print("청크 임베딩 (배치 모드)")
    print("=" * 60)
    embedded_chunks = embed_chunks_batch(chunks, embedding_model, batch_size=BATCH_SIZE)

    print("\n" + "=" * 60)
    print("임베딩 완료!")
    print("=" * 60)
    print(f"총 {len(embedded_chunks)}개의 청크가 임베딩되었습니다.")
    print(f"\n첫 번째 청크 샘플:")
    print(f"텍스트: {embedded_chunks[0]['text'][:100]}...")
    print(f"벡터 차원: {len(embedded_chunks[0]['vector'])}")
    print(f"메타데이터: {embedded_chunks[0]['metadata']}")
    
    # 임베딩 결과 저장
    save_embeddings(embedded_chunks, "embeddings.pkl")
    
    return embedded_chunks


if __name__ == "__main__":
    embedding_function()