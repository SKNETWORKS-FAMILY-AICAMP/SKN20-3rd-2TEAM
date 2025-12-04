"""
임베딩 모델 비교 평가 스크립트

5개의 임베딩 모델을 비교하여 최적의 모델을 선택합니다.

평가 모델:
1. sentence-transformers/all-MiniLM-L6-v2
2. sentence-transformers/all-mpnet-base-v2
3. sentence-transformers/msmarco-MiniLM-L-6-v3
4. sentence-transformers/allenai-specter (scientific papers)
5. OpenAI text-embedding-3-small
6. BAAI/bge-m3
7. E5-Mistral / Jina-embeddings (Jina v2, v3)
8. paraphrase-multilingual-mpnet-base-v2

평가 방법:
- 테스트 쿼리에 대한 검색 성능 측정
- Cosine similarity 기반 Recall@k, MRR 계산
- 각 쿼리에 대해 관련 문서가 상위에 포함되는지 평가

Version: 1.0
Author: SKN20-3rd-2TEAM
"""

import os
import pickle
import time
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==================== 로깅 설정 ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 평가용 데이터 클래스 ====================

@dataclass
class TestQuery:
    """평가용 테스트 쿼리"""
    query: str
    relevant_keywords: List[str]  # 관련 키워드 (metadata.tags에서 매칭)
    relevant_doc_ids: List[str] = None  # 특정 문서 ID (optional)

    def __repr__(self):
        return f"TestQuery(query='{self.query[:50]}...', keywords={self.relevant_keywords})"


@dataclass
class EvaluationResult:
    """모델 평가 결과"""
    model_name: str
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    avg_time: float  # 평균 검색 시간 (초)

    def __repr__(self):
        return (f"EvaluationResult(model='{self.model_name}', "
                f"R@5={self.recall_at_5:.3f}, R@10={self.recall_at_10:.3f}, "
                f"MRR={self.mrr:.3f}, time={self.avg_time:.2f}s)")


# ==================== 테스트 쿼리 정의 ====================

TEST_QUERIES = [
    TestQuery(
        query="최신 vision transformer 모델과 이미지 분류 성능",
        relevant_keywords=["vision", "transformer", "image"]
    ),
    TestQuery(
        query="대규모 언어 모델의 fine-tuning 기법",
        relevant_keywords=["llm", "fine-tuning", "training"]
    ),
    TestQuery(
        query="code generation을 위한 LLM 모델",
        relevant_keywords=["code", "generation", "llm"]
    ),
    TestQuery(
        query="multimodal learning과 vision-language 모델",
        relevant_keywords=["multimodal", "vision", "language"]
    ),
    TestQuery(
        query="reinforcement learning from human feedback",
        relevant_keywords=["reinforcement", "rlhf", "feedback"]
    ),
    TestQuery(
        query="diffusion models for image generation",
        relevant_keywords=["diffusion", "image", "generation"]
    ),
    TestQuery(
        query="efficient transformers and model compression",
        relevant_keywords=["efficient", "transformer", "compression"]
    ),
    TestQuery(
        query="graph neural networks and molecular modeling",
        relevant_keywords=["graph", "neural", "molecular"]
    ),
    TestQuery(
        query="video understanding and temporal modeling",
        relevant_keywords=["video", "temporal", "understanding"]
    ),
    TestQuery(
        query="zero-shot and few-shot learning methods",
        relevant_keywords=["zero-shot", "few-shot", "learning"]
    ),
]

# ==================== 임베딩 모델 초기화 ====================

def init_models() -> Dict[str, any]:
    """
    8개 임베딩 모델 초기화

    Returns:
        Dict[str, embedding_model]: 모델명 -> 임베딩 모델 객체
    """
    models = {}

    logger.info("[모델 초기화] 5개 임베딩 모델 로딩 시작...")

    # 1. all-MiniLM-L6-v2 (384 dim, 빠름)
    try:
        models["MiniLM-L6"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ MiniLM-L6-v2 로딩 완료")
    except Exception as e:
        logger.error(f"✗ MiniLM-L6-v2 로딩 실패: {e}")

    # 2. all-mpnet-base-v2 (768 dim, 높은 품질)
    try:
        models["MPNet"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ MPNet-base-v2 로딩 완료")
    except Exception as e:
        logger.error(f"✗ MPNet-base-v2 로딩 실패: {e}")

    # 3. msmarco-MiniLM-L-6-v3 (384 dim, 검색 최적화)
    try:
        models["MsMarco"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-MiniLM-L-6-v3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ MsMarco-MiniLM 로딩 완료")
    except Exception as e:
        logger.error(f"✗ MsMarco-MiniLM 로딩 실패: {e}")

    # 4. allenai-specter (768 dim, scientific papers 특화)
    try:
        models["SPECTER"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/allenai-specter",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ SPECTER (scientific) 로딩 완료")
    except Exception as e:
        logger.error(f"✗ SPECTER 로딩 실패: {e}")

    # 5. OpenAI text-embedding-3-small (1536 dim)
    try:
        if os.getenv("OPENAI_API_KEY"):
            models["OpenAI-small"] = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            logger.info("✓ OpenAI text-embedding-3-small 로딩 완료")
        else:
            logger.warning("✗ OPENAI_API_KEY not found, skipping OpenAI model")
    except Exception as e:
        logger.error(f"✗ OpenAI 모델 로딩 실패: {e}")

    # 6. BAAI/bge-m3 (1024 dim, 중국어 및 영어 지원)
    try:
        models["BGE-M3"] = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ BGE-M3 로딩 완료")
    except Exception as e:
        logger.error(f"✗ BGE-M3 로딩 실패: {e}")

    # 7. E5-Mistral / Jina-embeddings (Jina v2, v3)
    try:
        models["E5-Base"] = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ E5-Base 로딩 완료")
    except Exception as e:
        logger.error(f"✗ E5-Base 로딩 실패: {e}")

    # 8. paraphrase-multilingual-mpnet-base-v2 (768 dim, 다국어 지원)
    try:
        models["Paraphrase-Multi"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ Paraphrase-Multilingual 로딩 완료")
    except Exception as e:
        logger.error(f"✗ Paraphrase-Multilingual 로딩 실패: {e}")

    logger.info(f"[모델 초기화 완료] {len(models)}개 모델 준비됨")
    return models


# ==================== 평가 메트릭 계산 ====================

def calculate_recall_at_k(
    relevant_indices: List[int],
    retrieved_indices: List[int],
    k: int
) -> float:
    """
    Recall@k 계산

    Args:
        relevant_indices: 관련 문서 인덱스 목록
        retrieved_indices: 검색된 문서 인덱스 목록 (상위 k개)
        k: 상위 k개

    Returns:
        float: Recall@k 점수
    """
    if not relevant_indices:
        return 0.0

    retrieved_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)

    hits = len(retrieved_k & relevant_set)
    recall = hits / len(relevant_set)

    return recall


def calculate_mrr(
    relevant_indices: List[int],
    retrieved_indices: List[int]
) -> float:
    """
    Mean Reciprocal Rank 계산

    Args:
        relevant_indices: 관련 문서 인덱스 목록
        retrieved_indices: 검색된 문서 인덱스 목록

    Returns:
        float: MRR 점수
    """
    relevant_set = set(relevant_indices)

    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_set:
            return 1.0 / rank

    return 0.0


# ==================== 검색 및 평가 ====================

def find_relevant_docs(
    chunks: List[Document],
    test_query: TestQuery
) -> List[int]:
    """
    테스트 쿼리에 대한 관련 문서 인덱스 찾기

    Args:
        chunks: 전체 문서 청크
        test_query: 테스트 쿼리

    Returns:
        List[int]: 관련 문서 인덱스 목록
    """
    relevant_indices = []

    for idx, doc in enumerate(chunks):
        tags = doc.metadata.get('tags', [])

        # 관련 키워드가 tags에 포함되어 있는지 확인
        if any(keyword.lower() in [tag.lower() for tag in tags]
               for keyword in test_query.relevant_keywords):
            relevant_indices.append(idx)

    return relevant_indices


def retrieve_top_k(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int = 10
) -> List[int]:
    """
    Cosine similarity 기반 상위 k개 문서 인덱스 검색

    Args:
        query_embedding: 쿼리 임베딩 (1, dim)
        doc_embeddings: 문서 임베딩 (n, dim)
        k: 상위 k개

    Returns:
        List[int]: 상위 k개 문서 인덱스
    """
    # Cosine similarity 계산
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # 상위 k개 인덱스
    top_k_indices = np.argsort(similarities)[::-1][:k]

    return top_k_indices.tolist()


def evaluate_model(
    model_name: str,
    embedding_model: any,
    chunks: List[Document],
    test_queries: List[TestQuery],
    top_k: int = 10
) -> EvaluationResult:
    """
    단일 모델 평가

    Args:
        model_name: 모델명
        embedding_model: 임베딩 모델
        chunks: 전체 문서 청크
        test_queries: 테스트 쿼리 목록
        top_k: 상위 k개 검색

    Returns:
        EvaluationResult: 평가 결과
    """
    logger.info(f"\n[평가 시작] {model_name}")

    # 1. 모든 문서 임베딩 생성
    logger.info(f"  - 문서 임베딩 생성 중... ({len(chunks)}개)")
    start_time = time.time()

    try:
        # 배치로 임베딩 (메모리 효율성)
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch_docs = chunks[i:i+batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]

            if hasattr(embedding_model, 'embed_documents'):
                batch_embeddings = embedding_model.embed_documents(batch_texts)
            else:
                # OpenAI의 경우
                batch_embeddings = [embedding_model.embed_query(text) for text in batch_texts]

            all_embeddings.extend(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"    진행: {i+len(batch_docs)}/{len(chunks)}")

        doc_embeddings = np.array(all_embeddings)
        embedding_time = time.time() - start_time
        logger.info(f"  - 임베딩 완료 ({embedding_time:.2f}초)")

    except Exception as e:
        logger.error(f"  ✗ 임베딩 생성 실패: {e}")
        return EvaluationResult(model_name, 0.0, 0.0, 0.0, 0.0)

    # 2. 각 테스트 쿼리 평가
    recall_5_scores = []
    recall_10_scores = []
    mrr_scores = []
    query_times = []

    for query_idx, test_query in enumerate(test_queries, 1):
        logger.info(f"  - 쿼리 {query_idx}/{len(test_queries)}: '{test_query.query[:50]}...'")

        # 관련 문서 찾기
        relevant_indices = find_relevant_docs(chunks, test_query)

        if not relevant_indices:
            logger.warning(f"    경고: 관련 문서 없음")
            continue

        logger.info(f"    관련 문서: {len(relevant_indices)}개")

        # 쿼리 임베딩 및 검색
        try:
            start_time = time.time()
            query_embedding = np.array([embedding_model.embed_query(test_query.query)])
            retrieved_indices = retrieve_top_k(query_embedding, doc_embeddings, k=top_k)
            query_time = time.time() - start_time
            query_times.append(query_time)

            # 메트릭 계산
            recall_5 = calculate_recall_at_k(relevant_indices, retrieved_indices, k=5)
            recall_10 = calculate_recall_at_k(relevant_indices, retrieved_indices, k=10)
            mrr = calculate_mrr(relevant_indices, retrieved_indices)

            recall_5_scores.append(recall_5)
            recall_10_scores.append(recall_10)
            mrr_scores.append(mrr)

            logger.info(f"    R@5={recall_5:.3f}, R@10={recall_10:.3f}, MRR={mrr:.3f}")

        except Exception as e:
            logger.error(f"    ✗ 쿼리 평가 실패: {e}")
            continue

    # 3. 평균 계산
    avg_recall_5 = np.mean(recall_5_scores) if recall_5_scores else 0.0
    avg_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
    avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    avg_time = np.mean(query_times) if query_times else 0.0

    result = EvaluationResult(
        model_name=model_name,
        recall_at_5=avg_recall_5,
        recall_at_10=avg_recall_10,
        mrr=avg_mrr,
        avg_time=avg_time
    )

    logger.info(f"[평가 완료] {result}")
    return result


# ==================== 메인 실행 ====================

def main():
    """메인 실행 함수"""
    logger.info("=" * 80)
    logger.info("임베딩 모델 비교 평가 시작")
    logger.info("=" * 80)

    # 1. 청크 데이터 로드
    chunks_path = "01_data/chunks/chunks_all.pkl"
    logger.info(f"\n[데이터 로드] {chunks_path}")

    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)

    logger.info(f"  - 총 청크 수: {len(chunks)}")

    # 샘플링 (전체 데이터는 시간이 오래 걸림)
    # 전체 평가를 원하면 이 부분 제거
    if len(chunks) > 2000:
        logger.warning(f"  - 평가 시간 단축을 위해 2000개 샘플링")
        np.random.seed(42)
        sample_indices = np.random.choice(len(chunks), 2000, replace=False)
        chunks = [chunks[i] for i in sample_indices]

    # 2. 모델 초기화
    models = init_models()

    if not models:
        logger.error("사용 가능한 모델이 없습니다!")
        return

    # 3. 각 모델 평가
    results = []

    for model_name, embedding_model in models.items():
        try:
            result = evaluate_model(
                model_name=model_name,
                embedding_model=embedding_model,
                chunks=chunks,
                test_queries=TEST_QUERIES,
                top_k=10
            )
            results.append(result)
        except Exception as e:
            logger.error(f"모델 {model_name} 평가 실패: {e}")

    # 4. 결과 출력
    logger.info("\n" + "=" * 80)
    logger.info("최종 평가 결과")
    logger.info("=" * 80)

    # 정렬 (Recall@10 기준)
    results.sort(key=lambda x: x.recall_at_10, reverse=True)

    print("\n{:<20} {:>10} {:>10} {:>10} {:>12}".format(
        "Model", "R@5", "R@10", "MRR", "Avg Time(s)"
    ))
    print("-" * 65)

    for result in results:
        print("{:<20} {:>10.3f} {:>10.3f} {:>10.3f} {:>12.2f}".format(
            result.model_name,
            result.recall_at_5,
            result.recall_at_10,
            result.mrr,
            result.avg_time
        ))

    # 최고 성능 모델
    if results:
        best_model = results[0]
        logger.info(f"\n🏆 최고 성능 모델: {best_model.model_name}")
        logger.info(f"   - Recall@10: {best_model.recall_at_10:.3f}")
        logger.info(f"   - MRR: {best_model.mrr:.3f}")
        logger.info(f"   - 평균 검색 시간: {best_model.avg_time:.2f}초")


if __name__ == "__main__":
    main()
