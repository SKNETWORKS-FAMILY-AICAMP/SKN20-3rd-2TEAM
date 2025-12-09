"""
임베딩 모델 및 청크 전략 비교 평가 스크립트

여러 임베딩 모델과 청크 전략 조합을 비교하여 최적의 조합을 선택합니다.

평가 모델:
1. sentence-transformers/all-MiniLM-L6-v2 (384 dim, 빠름)
2. sentence-transformers/all-mpnet-base-v2 (768 dim, 고품질)
3. sentence-transformers/msmarco-MiniLM-L-6-v3 (384 dim, 검색 최적화)
4. sentence-transformers/allenai-specter (768 dim, scientific papers 특화)
5. OpenAI text-embedding-3-small (1536 dim)
6. BAAI/bge-m3 (1024 dim, 다국어)
7. paraphrase-multilingual-mpnet-base-v2 (768 dim, 다국어)

평가 청크 전략:
- 자동으로 01_data/chunks/ 디렉토리의 모든 *_C.pkl 파일 탐색
- 다양한 chunk_size와 overlap 조합 평가

평가 메트릭:
- Hit Rate@K: 상위 K개 결과 중 최소 1개의 관련 문서 포함 비율
- MRR (Mean Reciprocal Rank): 첫 번째 관련 문서의 역순위 평균
- NDCG@K: 순위를 고려한 검색 품질 측정
- Avg Time: 쿼리당 평균 검색 시간

평가 방법:
- 테스트 쿼리 10개에 대한 검색 성능 측정
- Cosine similarity 기반 검색
- 각 모델 × 청크 전략 조합에 대해 평가 수행

출력:
- 터미널에 결과 테이블 출력
- embedding_evaluation_report.md 마크다운 보고서 자동 생성
  - 전체 결과 표
  - 최고 성능 조합 (NDCG@10, Hit Rate@10, MRR, 속도 기준)
  - 청크 전략별/모델별 비교
  - 프로덕션 사용 권장사항
"""

import os
import pickle
import time
import logging
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==================== 로깅 설정 ====================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    chunk_strategy: str  # e.g., "100_15"
    hit_rate_at_5: float
    hit_rate_at_10: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_5: float
    ndcg_at_10: float
    avg_time: float  # 평균 검색 시간 (초)

    def __repr__(self):
        return (
            f"EvaluationResult(model='{self.model_name}', "
            f"chunk={self.chunk_strategy}, "
            f"HR@5={self.hit_rate_at_5:.3f}, HR@10={self.hit_rate_at_10:.3f}, "
            f"MRR={self.mrr:.3f}, NDCG@5={self.ndcg_at_5:.3f}, NDCG@10={self.ndcg_at_10:.3f}, "
            f"time={self.avg_time:.2f}s)"
        )


# ==================== 테스트 쿼리 정의 ====================
# 실제 클러스터 메타데이터 기반 개선된 TEST_QUERIES 사용

from improved_test_queries import IMPROVED_TEST_QUERIES as TEST_QUERIES

# 클러스터 커버리지: 18개 클러스터 중 18개 (100%)
# 주요 고품질 클러스터 포함:
#   - Cluster 17 (LLM training, avg_upvote: 44.3)
#   - Cluster 11 (Attention, avg_upvote: 42.9)
#   - Cluster 5 (Long context, avg_upvote: 38.7)
#   - Cluster 15 (Video/multimodal, avg_upvote: 39.6)
# 총 14개 쿼리로 모든 클러스터 커버

# ==================== 임베딩 모델 초기화 ====================


def init_models() -> Dict[str, any]:
    """
    7개 임베딩 모델 초기화

    Returns:
        Dict[str, embedding_model]: 모델명 -> 임베딩 모델 객체
    """
    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("[모델 초기화] 7개 임베딩 모델 로딩 시작...")
    logger.info(f"[GPU 사용 여부] {device}")

    # 1. all-MiniLM-L6-v2 (384 dim, 빠름)
    try:
        models["MiniLM-L6"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✓ MiniLM-L6-v2 로딩 완료")
    except Exception as e:
        logger.error(f"✗ MiniLM-L6-v2 로딩 실패: {e}")

    # 2. all-mpnet-base-v2 (768 dim, 높은 품질)
    try:
        models["MPNet"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✓ MPNet-base-v2 로딩 완료")
    except Exception as e:
        logger.error(f"✗ MPNet-base-v2 로딩 실패: {e}")

    # 3. msmarco-MiniLM-L-6-v3 (384 dim, 검색 최적화)
    try:
        models["MsMarco"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-MiniLM-L-6-v3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✓ MsMarco-MiniLM 로딩 완료")
    except Exception as e:
        logger.error(f"✗ MsMarco-MiniLM 로딩 실패: {e}")

    # 4. allenai-specter (768 dim, scientific papers 특화)
    try:
        models["SPECTER"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/allenai-specter",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✓ SPECTER (scientific) 로딩 완료")
    except Exception as e:
        logger.error(f"✗ SPECTER 로딩 실패: {e}")

    # 5. OpenAI text-embedding-3-small (1536 dim)
    try:
        if os.getenv("OPENAI_API_KEY"):
            models["OpenAI-small"] = OpenAIEmbeddings(model="text-embedding-3-small")
            logger.info("✓ OpenAI text-embedding-3-small 로딩 완료")
        else:
            logger.warning("✗ OPENAI_API_KEY not found, skipping OpenAI model")
    except Exception as e:
        logger.error(f"✗ OpenAI 모델 로딩 실패: {e}")

    # 6. BAAI/bge-m3 (1024 dim, 중국어 및 영어 지원)
    try:
        models["BGE-M3"] = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✓ BGE-M3 로딩 완료")
    except Exception as e:
        logger.error(f"✗ BGE-M3 로딩 실패: {e}")

    # 7. paraphrase-multilingual-mpnet-base-v2 (768 dim, 다국어 지원)
    try:
        models["Paraphrase-Multi"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✓ Paraphrase-Multilingual 로딩 완료")
    except Exception as e:
        logger.error(f"✗ Paraphrase-Multilingual 로딩 실패: {e}")

    logger.info(f"[모델 초기화 완료] {len(models)}개 모델 준비됨")
    return models


# ==================== 평가 메트릭 계산 ====================


def calculate_hit_rate_at_k(
    relevant_indices: List[int], retrieved_indices: List[int], k: int
) -> float:
    """
    Hit Rate@k 계산
    최소 1개의 관련 문서가 상위 k개에 포함되면 1, 아니면 0

    Args:
        relevant_indices: 관련 문서 인덱스 목록
        retrieved_indices: 검색된 문서 인덱스 목록 (상위 k개)
        k: 상위 k개

    Returns:
        float: Hit Rate@k 점수 (0 or 1)
    """
    if not relevant_indices:
        return 0.0

    retrieved_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)

    # 교집합이 있으면 1, 없으면 0
    return 1.0 if len(retrieved_k & relevant_set) > 0 else 0.0


def calculate_mrr(relevant_indices: List[int], retrieved_indices: List[int]) -> float:
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


def calculate_ndcg_at_k(
    relevant_indices: List[int], retrieved_indices: List[int], k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain@k 계산

    Args:
        relevant_indices: 관련 문서 인덱스 목록
        retrieved_indices: 검색된 문서 인덱스 목록
        k: 상위 k개

    Returns:
        float: NDCG@k 점수
    """
    if not relevant_indices:
        return 0.0

    relevant_set = set(relevant_indices)
    retrieved_k = retrieved_indices[:k]

    # DCG 계산 (relevance는 binary: relevant=1, irrelevant=0)
    dcg = 0.0
    for i, idx in enumerate(retrieved_k, 1):
        if idx in relevant_set:
            # rel_i / log2(i+1)
            dcg += 1.0 / np.log2(i + 1)

    # IDCG 계산 (이상적인 순서: 모든 relevant 문서가 앞에)
    idcg = 0.0
    for i in range(1, min(len(relevant_indices), k) + 1):
        idcg += 1.0 / np.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ==================== 클러스터 메타데이터 로딩 ====================

# 전역 변수로 클러스터 메타데이터 캐싱
_cluster_assignments = None
_cluster_metadata = None


def load_cluster_metadata():
    """클러스터 메타데이터를 로드 (한 번만 로드)"""
    global _cluster_assignments, _cluster_metadata

    if _cluster_assignments is not None and _cluster_metadata is not None:
        return _cluster_assignments, _cluster_metadata

    import json
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # cluster_assignments.json 로드
    assignments_path = PROJECT_ROOT / "01_data" / "clusters" / "cluster_assignments.json"
    with open(assignments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        _cluster_assignments = data["assignments"]

    # cluster_metadata.json 로드
    metadata_path = PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        _cluster_metadata = data["clusters"]

    logger.info(f"[CLUSTER] 클러스터 메타데이터 로드 완료: {len(_cluster_assignments)} documents, {len(_cluster_metadata)} clusters")

    return _cluster_assignments, _cluster_metadata


# ==================== 검색 및 평가 ====================


def find_relevant_docs(chunks: List[Document], test_query: TestQuery) -> List[int]:
    """
    테스트 쿼리에 대한 관련 문서 인덱스 찾기

    클러스터 메타데이터를 사용하여 관련성 판단:
    1. 문서의 doc_id로 cluster_id 찾기
    2. cluster_id로 cluster keywords 가져오기
    3. test_query의 relevant_keywords와 cluster keywords 매칭

    Args:
        chunks: 전체 문서 청크
        test_query: 테스트 쿼리

    Returns:
        List[int]: 관련 문서 인덱스 목록
    """
    # 클러스터 메타데이터 로드
    cluster_assignments, cluster_metadata = load_cluster_metadata()

    relevant_indices = []

    for idx, doc in enumerate(chunks):
        doc_id = doc.metadata.get("doc_id", "")

        # 문서의 청크 ID 제거 (doc2549001_chunk_0 -> doc2549001)
        if "_chunk_" in doc_id:
            base_doc_id = doc_id.split("_chunk_")[0]
        else:
            base_doc_id = doc_id

        # cluster_id 가져오기
        cluster_id = cluster_assignments.get(base_doc_id, None)

        if cluster_id is None:
            continue

        # cluster keywords 가져오기
        cluster_info = cluster_metadata.get(str(cluster_id), {})
        cluster_keywords = cluster_info.get("keywords", [])

        # 관련 키워드가 cluster keywords 내 부분 문자열로 포함되어 있는지 확인
        if any(
            keyword.lower() in cluster_keyword.lower()
            for keyword in test_query.relevant_keywords
            for cluster_keyword in cluster_keywords
        ):
            relevant_indices.append(idx)

    return relevant_indices


def retrieve_top_k(
    query_embedding: np.ndarray, doc_embeddings: np.ndarray, k: int = 10
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
    chunk_strategy: str,
    top_k: int = 10,
) -> EvaluationResult:
    """
    단일 모델 평가

    Args:
        model_name: 모델명
        embedding_model: 임베딩 모델
        chunks: 전체 문서 청크
        test_queries: 테스트 쿼리 목록
        chunk_strategy: 청크 전략 (e.g., "100_15")
        top_k: 상위 k개 검색

    Returns:
        EvaluationResult: 평가 결과
    """
    logger.info(f"\n[평가 시작] {model_name} (chunk: {chunk_strategy})")

    # 1. 모든 문서 임베딩 생성
    logger.info(f"  - 문서 임베딩 생성 중... ({len(chunks)}개)")
    start_time = time.time()

    try:
        # 배치로 임베딩 (메모리 효율성)
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch_docs = chunks[i : i + batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]

            if hasattr(embedding_model, "embed_documents"):
                batch_embeddings = embedding_model.embed_documents(batch_texts)
            else:
                # OpenAI의 경우
                batch_embeddings = [
                    embedding_model.embed_query(text) for text in batch_texts
                ]

            all_embeddings.extend(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"    진행: {i+len(batch_docs)}/{len(chunks)}")

        doc_embeddings = np.array(all_embeddings)
        embedding_time = time.time() - start_time
        logger.info(f"  - 임베딩 완료 ({embedding_time:.2f}초)")

    except Exception as e:
        logger.error(f"  ✗ 임베딩 생성 실패: {e}")
        return EvaluationResult(model_name, chunk_strategy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 2. 각 테스트 쿼리 평가
    hit_rate_5_scores = []
    hit_rate_10_scores = []
    mrr_scores = []
    ndcg_5_scores = []
    ndcg_10_scores = []
    query_times = []

    for query_idx, test_query in enumerate(test_queries, 1):
        logger.info(
            f"  - 쿼리 {query_idx}/{len(test_queries)}: '{test_query.query[:50]}...'"
        )

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
            hit_rate_5 = calculate_hit_rate_at_k(relevant_indices, retrieved_indices, k=5)
            hit_rate_10 = calculate_hit_rate_at_k(relevant_indices, retrieved_indices, k=10)
            mrr = calculate_mrr(relevant_indices, retrieved_indices)
            ndcg_5 = calculate_ndcg_at_k(relevant_indices, retrieved_indices, k=5)
            ndcg_10 = calculate_ndcg_at_k(relevant_indices, retrieved_indices, k=10)

            hit_rate_5_scores.append(hit_rate_5)
            hit_rate_10_scores.append(hit_rate_10)
            mrr_scores.append(mrr)
            ndcg_5_scores.append(ndcg_5)
            ndcg_10_scores.append(ndcg_10)

            logger.info(
                f"    HR@5={hit_rate_5:.3f}, HR@10={hit_rate_10:.3f}, MRR={mrr:.3f}, "
                f"NDCG@5={ndcg_5:.3f}, NDCG@10={ndcg_10:.3f}"
            )

        except Exception as e:
            logger.error(f"    ✗ 쿼리 평가 실패: {e}")
            continue

    # 3. 평균 계산
    avg_hit_rate_5 = np.mean(hit_rate_5_scores) if hit_rate_5_scores else 0.0
    avg_hit_rate_10 = np.mean(hit_rate_10_scores) if hit_rate_10_scores else 0.0
    avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    avg_ndcg_5 = np.mean(ndcg_5_scores) if ndcg_5_scores else 0.0
    avg_ndcg_10 = np.mean(ndcg_10_scores) if ndcg_10_scores else 0.0
    avg_time = np.mean(query_times) if query_times else 0.0

    result = EvaluationResult(
        model_name=model_name,
        chunk_strategy=chunk_strategy,
        hit_rate_at_5=avg_hit_rate_5,
        hit_rate_at_10=avg_hit_rate_10,
        mrr=avg_mrr,
        ndcg_at_5=avg_ndcg_5,
        ndcg_at_10=avg_ndcg_10,
        avg_time=avg_time,
    )

    logger.info(f"[평가 완료] {result}")
    return result


# ==================== 청크 전략 관리 ====================


def find_chunk_files(chunks_dir: str = "01_data/chunks") -> List[str]:
    """
    chunks 디렉토리에서 사용 가능한 청크 파일 찾기
    _C suffix (clustering 적용) 파일만 사용

    Args:
        chunks_dir: 청크 디렉토리 경로

    Returns:
        List[str]: 청크 파일 경로 목록
    """
    import glob

    pattern = os.path.join(chunks_dir, "chunks_*_*_C.pkl")
    chunk_files = glob.glob(pattern)

    logger.info(f"[청크 파일 탐색] {len(chunk_files)}개 파일 발견")
    for f in chunk_files:
        logger.info(f"  - {os.path.basename(f)}")

    return sorted(chunk_files)


def extract_chunk_strategy(chunk_file_path: str) -> str:
    """
    청크 파일명에서 전략 추출 (e.g., "chunks_100_15_C.pkl" -> "100_15")

    Args:
        chunk_file_path: 청크 파일 경로

    Returns:
        str: 청크 전략 (size_overlap)
    """
    import re

    basename = os.path.basename(chunk_file_path)
    match = re.search(r"chunks_(\d+)_(\d+)_C\.pkl", basename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return "unknown"


# ==================== 마크다운 보고서 생성 ====================


def generate_markdown_report(
    results: List[EvaluationResult], output_path: str = "embedding_evaluation_report.md"
) -> None:
    """
    평가 결과를 마크다운 보고서로 생성

    Args:
        results: 평가 결과 목록
        output_path: 출력 파일 경로
    """
    from datetime import datetime

    with open(output_path, "w", encoding="utf-8") as f:
        # 헤더
        f.write("# 임베딩 모델 및 청크 전략 평가 보고서\n\n")
        f.write(f"**생성 날짜**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 평가 개요
        f.write("## 1. 평가 개요\n\n")
        f.write("### 평가 모델\n\n")
        unique_models = sorted(set(r.model_name for r in results))
        for i, model in enumerate(unique_models, 1):
            f.write(f"{i}. {model}\n")

        f.write("\n### 평가 청크 전략\n\n")
        unique_strategies = sorted(set(r.chunk_strategy for r in results))
        for strategy in unique_strategies:
            size, overlap = strategy.split("_")
            f.write(f"- **{strategy}**: chunk_size={size}, overlap={overlap}\n")

        f.write("\n### 평가 메트릭\n\n")
        f.write("- **Hit Rate@K**: 상위 K개 결과 중 최소 1개의 관련 문서 포함 비율 (0~1)\n")
        f.write("- **MRR (Mean Reciprocal Rank)**: 첫 번째 관련 문서의 역순위 평균\n")
        f.write(
            "- **NDCG@K (Normalized Discounted Cumulative Gain)**: 순위를 고려한 검색 품질 (0~1)\n"
        )
        f.write("- **Avg Time**: 쿼리당 평균 검색 시간 (초)\n\n")

        f.write("---\n\n")

        # 전체 결과 표
        f.write("## 2. 전체 평가 결과\n\n")
        f.write("| 모델 | 청크 전략 | HR@5 | HR@10 | MRR | NDCG@5 | NDCG@10 | Avg Time(s) |\n")
        f.write("|------|-----------|------|-------|-----|--------|---------|-------------|\n")

        for result in results:
            f.write(
                f"| {result.model_name} | {result.chunk_strategy} | "
                f"{result.hit_rate_at_5:.3f} | {result.hit_rate_at_10:.3f} | "
                f"{result.mrr:.3f} | {result.ndcg_at_5:.3f} | {result.ndcg_at_10:.3f} | "
                f"{result.avg_time:.2f} |\n"
            )

        f.write("\n---\n\n")

        # 최고 성능 모델
        f.write("## 3. 최고 성능 조합\n\n")

        # NDCG@10 기준
        best_ndcg10 = max(results, key=lambda x: x.ndcg_at_10)
        f.write("### NDCG@10 기준 (종합 검색 품질)\n\n")
        f.write(f"- **모델**: {best_ndcg10.model_name}\n")
        f.write(f"- **청크 전략**: {best_ndcg10.chunk_strategy}\n")
        f.write(f"- **NDCG@10**: {best_ndcg10.ndcg_at_10:.3f}\n")
        f.write(f"- **HR@10**: {best_ndcg10.hit_rate_at_10:.3f}\n")
        f.write(f"- **MRR**: {best_ndcg10.mrr:.3f}\n")
        f.write(f"- **평균 검색 시간**: {best_ndcg10.avg_time:.2f}초\n\n")

        # Hit Rate@10 기준
        best_hr10 = max(results, key=lambda x: x.hit_rate_at_10)
        f.write("### Hit Rate@10 기준 (검색 재현율)\n\n")
        f.write(f"- **모델**: {best_hr10.model_name}\n")
        f.write(f"- **청크 전략**: {best_hr10.chunk_strategy}\n")
        f.write(f"- **HR@10**: {best_hr10.hit_rate_at_10:.3f}\n")
        f.write(f"- **NDCG@10**: {best_hr10.ndcg_at_10:.3f}\n")
        f.write(f"- **MRR**: {best_hr10.mrr:.3f}\n\n")

        # MRR 기준
        best_mrr = max(results, key=lambda x: x.mrr)
        f.write("### MRR 기준 (첫 번째 관련 문서 순위)\n\n")
        f.write(f"- **모델**: {best_mrr.model_name}\n")
        f.write(f"- **청크 전략**: {best_mrr.chunk_strategy}\n")
        f.write(f"- **MRR**: {best_mrr.mrr:.3f}\n")
        f.write(f"- **NDCG@10**: {best_mrr.ndcg_at_10:.3f}\n\n")

        # 가장 빠른 모델
        fastest = min(results, key=lambda x: x.avg_time)
        f.write("### 검색 속도 (가장 빠름)\n\n")
        f.write(f"- **모델**: {fastest.model_name}\n")
        f.write(f"- **청크 전략**: {fastest.chunk_strategy}\n")
        f.write(f"- **평균 검색 시간**: {fastest.avg_time:.2f}초\n")
        f.write(f"- **NDCG@10**: {fastest.ndcg_at_10:.3f}\n\n")

        f.write("---\n\n")

        # 청크 전략별 비교
        f.write("## 4. 청크 전략별 성능 비교\n\n")
        for strategy in unique_strategies:
            strategy_results = [r for r in results if r.chunk_strategy == strategy]
            if not strategy_results:
                continue

            f.write(f"### 청크 전략: {strategy}\n\n")
            f.write("| 모델 | HR@10 | NDCG@10 | MRR |\n")
            f.write("|------|-------|---------|-----|\n")

            for result in sorted(strategy_results, key=lambda x: x.ndcg_at_10, reverse=True):
                f.write(
                    f"| {result.model_name} | {result.hit_rate_at_10:.3f} | "
                    f"{result.ndcg_at_10:.3f} | {result.mrr:.3f} |\n"
                )

            f.write("\n")

        f.write("---\n\n")

        # 모델별 비교
        f.write("## 5. 모델별 성능 비교\n\n")
        for model in unique_models:
            model_results = [r for r in results if r.model_name == model]
            if not model_results:
                continue

            f.write(f"### 모델: {model}\n\n")
            f.write("| 청크 전략 | HR@10 | NDCG@10 | MRR |\n")
            f.write("|-----------|-------|---------|-----|\n")

            for result in sorted(model_results, key=lambda x: x.ndcg_at_10, reverse=True):
                f.write(
                    f"| {result.chunk_strategy} | {result.hit_rate_at_10:.3f} | "
                    f"{result.ndcg_at_10:.3f} | {result.mrr:.3f} |\n"
                )

            f.write("\n")

        f.write("---\n\n")

        # 권장사항
        f.write("## 6. 권장사항\n\n")
        f.write("### 프로덕션 사용 권장\n\n")
        f.write(
            f"**종합 평가 최고 성능**: `{best_ndcg10.model_name}` + `chunk_strategy={best_ndcg10.chunk_strategy}`\n\n"
        )

        # 성능-속도 트레이드오프 분석
        f.write("### 성능-속도 트레이드오프\n\n")

        # 상위 3개 NDCG@10 모델 선별
        top3_ndcg = sorted(results, key=lambda x: x.ndcg_at_10, reverse=True)[:3]

        f.write("**상위 3개 조합 (NDCG@10 기준)**:\n\n")
        for i, result in enumerate(top3_ndcg, 1):
            f.write(
                f"{i}. **{result.model_name} + {result.chunk_strategy}** - "
                f"NDCG@10={result.ndcg_at_10:.3f}, Time={result.avg_time:.2f}s\n"
            )

        f.write("\n")

        # 속도 우선시
        fast_and_good = [r for r in results if r.avg_time < 0.1 and r.ndcg_at_10 > 0.5]
        if fast_and_good:
            best_fast = max(fast_and_good, key=lambda x: x.ndcg_at_10)
            f.write("**속도 우선 (검색 시간 < 0.1초 && NDCG@10 > 0.5)**:\n\n")
            f.write(
                f"- {best_fast.model_name} + {best_fast.chunk_strategy} - "
                f"NDCG@10={best_fast.ndcg_at_10:.3f}, Time={best_fast.avg_time:.2f}s\n\n"
            )

        f.write("---\n\n")
        f.write(
            "*이 보고서는 `evaluate_embeddings.py`에 의해 자동 생성되었습니다.*\n"
        )

    logger.info(f"\n[보고서 생성 완료] {output_path}")


# ==================== 메인 실행 ====================


def main():
    """메인 실행 함수 - 여러 청크 전략과 임베딩 모델 조합 평가"""
    logger.info("=" * 80)
    logger.info("임베딩 모델 및 청크 전략 비교 평가 시작")
    logger.info("=" * 80)

    # 1. 청크 파일 탐색
    chunk_files = find_chunk_files("01_data/chunks")

    if not chunk_files:
        logger.error("청크 파일을 찾을 수 없습니다!")
        return

    # 2. 모델 초기화 (한 번만)
    models = init_models()

    if not models:
        logger.error("사용 가능한 모델이 없습니다!")
        return

    # 3. 모든 조합 평가
    all_results = []

    for chunk_file_path in chunk_files:
        chunk_strategy = extract_chunk_strategy(chunk_file_path)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"청크 전략: {chunk_strategy}")
        logger.info(f"파일: {os.path.basename(chunk_file_path)}")
        logger.info(f"{'=' * 80}")

        # 청크 데이터 로드
        try:
            with open(chunk_file_path, "rb") as f:
                chunks = pickle.load(f)
            logger.info(f"  - 총 청크 수: {len(chunks)}")
        except Exception as e:
            logger.error(f"  ✗ 청크 로드 실패: {e}")
            continue

        # 샘플링 (평가 시간 단축)
        # 전체 평가를 원하면 SAMPLE_SIZE = None 설정
        SAMPLE_SIZE = None  # 전체 데이터 평가
        if SAMPLE_SIZE and len(chunks) > SAMPLE_SIZE:
            logger.warning(f"  - 평가 시간 단축을 위해 {SAMPLE_SIZE}개 샘플링")
            np.random.seed(42)
            sample_indices = np.random.choice(len(chunks), SAMPLE_SIZE, replace=False)
            chunks = [chunks[i] for i in sample_indices]

        # 각 모델 평가
        for model_name, embedding_model in models.items():
            try:
                result = evaluate_model(
                    model_name=model_name,
                    embedding_model=embedding_model,
                    chunks=chunks,
                    test_queries=TEST_QUERIES,
                    chunk_strategy=chunk_strategy,
                    top_k=10,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"모델 {model_name} 평가 실패: {e}")

    # 4. 결과 출력 (터미널)
    logger.info("\n" + "=" * 80)
    logger.info("최종 평가 결과")
    logger.info("=" * 80)

    # 정렬 (NDCG@10 기준)
    all_results.sort(key=lambda x: x.ndcg_at_10, reverse=True)

    print(
        "\n{:<15} {:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
            "Model", "Chunk", "HR@5", "HR@10", "MRR", "NDCG@5", "NDCG@10", "Time(s)"
        )
    )
    print("-" * 90)

    for result in all_results:
        print(
            "{:<15} {:<10} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.2f}".format(
                result.model_name,
                result.chunk_strategy,
                result.hit_rate_at_5,
                result.hit_rate_at_10,
                result.mrr,
                result.ndcg_at_5,
                result.ndcg_at_10,
                result.avg_time,
            )
        )

    # 최고 성능 조합
    if all_results:
        best_model = all_results[0]
        logger.info(f"\n🏆 최고 성능 조합 (NDCG@10 기준):")
        logger.info(f"   - 모델: {best_model.model_name}")
        logger.info(f"   - 청크 전략: {best_model.chunk_strategy}")
        logger.info(f"   - NDCG@10: {best_model.ndcg_at_10:.3f}")
        logger.info(f"   - Hit Rate@10: {best_model.hit_rate_at_10:.3f}")
        logger.info(f"   - MRR: {best_model.mrr:.3f}")
        logger.info(f"   - 평균 검색 시간: {best_model.avg_time:.2f}초")

    # 5. 마크다운 보고서 생성
    if all_results:
        report_path = "embedding_evaluation_report.md"
        generate_markdown_report(all_results, report_path)
        logger.info(f"\n✓ 마크다운 보고서 생성 완료: {report_path}")


if __name__ == "__main__":
    main()
