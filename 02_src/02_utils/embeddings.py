"""
RAG 기반 HuggingFace Papers 챗봇 임베딩 관리 모듈

주요 기능:
1. HuggingFace sentence-transformers 임베딩 래퍼
2. 단일/배치 텍스트 임베딩 (재시도 로직 포함)
3. Lazy initialization (첫 호출 시 모델 로딩)
4. 임베딩 차원 조회 및 모델 정보 관리

CRITICAL 설정:
- HuggingFaceEmbeddings 사용 (OpenAI 임베딩 아님!)
- 모델: sentence-transformers/all-MiniLM-L6-v2
- 차원: 384 (all-MiniLM-L6-v2 기준)
- 정규화: normalize_embeddings=True (유사도 검색 최적화)
- 디바이스: CPU (GPU 사용 가능 시 'cuda'로 변경 가능)

임베딩 처리:
- 단일 텍스트: embed_text() - 최대 3회 재시도, exponential backoff
- 배치 텍스트: embed_batch() - batch_size=100, 진행 상황 로깅

모델 선택 근거 (2025-12-04 평가 완료):
✅ 5개 모델 비교 평가 결과 1위 선정
   - sentence-transformers/all-MiniLM-L6-v2 (현재 모델) ⭐ BEST
   - sentence-transformers/all-mpnet-base-v2
   - sentence-transformers/msmarco-MiniLM-L-6-v3
   - sentence-transformers/allenai-specter
   - OpenAI text-embedding-3-small

평가 결과:
   - Recall@10: 6.4% (1위) - 가장 정확한 검색
   - MRR: 0.658 (1위) - 관련 문서가 평균 상위 1.5위에 등장
   - 속도: 0.02초/쿼리 (1위) - OpenAI 대비 28배 빠름
   - 비용: 무료 (OpenAI는 유료)

상세 평가 결과: 02_src/02_utils/EMBEDDING_EVALUATION_RESULTS.md
평가 스크립트: 02_src/02_utils/evaluate_embeddings.py

Version: 2.1 (Model evaluation completed, performance validated)
Author: SKN20-3rd-2TEAM
"""

import time
import logging
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================== 전역 설정 (Inline constants) ====================

logger = logging.getLogger(__name__)

# 임베딩 모델 설정 (HuggingFace - FREE!)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"  # GPU 사용 시 "cuda"로 변경

# 재시도 설정
DEFAULT_MAX_RETRIES = 3
EXPONENTIAL_BACKOFF_BASE = 2

# 배치 처리 설정
DEFAULT_BATCH_SIZE = 100


# ==================== EmbeddingManager 클래스 ====================

class EmbeddingManager:
    """
    HuggingFace sentence-transformers 임베딩 관리 클래스

    CRITICAL: HuggingFaceEmbeddings 사용 (OpenAI 임베딩 X)
    EMBEDDING_MODEL 설정: "sentence-transformers/all-MiniLM-L6-v2"

    주요 메서드:
        - get_embeddings_model(): LangChain HuggingFaceEmbeddings 인스턴스 반환
        - embed_text(text): 단일 텍스트 임베딩 (재시도 로직)
        - embed_batch(texts): 배치 텍스트 임베딩 (효율성)
        - get_embedding_dimension(): 임베딩 차원 반환 (384)

    예시:
        >>> em = EmbeddingManager()
        >>> embedding = em.embed_text("transformer models")
        >>> len(embedding)  # 384
        384
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        임베딩 관리자 초기화

        Args:
            model_name: HuggingFace 임베딩 모델명
                       (기본값: EMBEDDING_MODEL = sentence-transformers/all-MiniLM-L6-v2)
            device: 디바이스 (cpu 또는 cuda, 기본값: cpu)

        예시:
            >>> em = EmbeddingManager()
            >>> em = EmbeddingManager("sentence-transformers/all-MiniLM-L6-v2")
            >>> em = EmbeddingManager(device="cuda")  # GPU 사용
        """
        # 모델명 설정
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or DEFAULT_DEVICE

        logger.info(f"[초기화] EmbeddingManager 모델: {self.model_name}, 디바이스: {self.device}")

        # Lazy initialization (첫 호출 시 로딩)
        self._embeddings = None

    def get_embeddings_model(self) -> HuggingFaceEmbeddings:
        """
        HuggingFaceEmbeddings 인스턴스 반환 (Lazy loading)

        CRITICAL FIX: OpenAI 임베딩 버그 수정!
        이전 버전에서는 OpenAIEmbeddings를 사용하여 비용이 발생했으나,
        이제 HuggingFaceEmbeddings를 사용하여 무료로 임베딩을 생성합니다.

        Returns:
            HuggingFaceEmbeddings: LangChain 임베딩 객체

        동작:
            - 첫 호출 시에만 모델 로딩 (이후 캐싱)
            - CPU 사용 (GPU 사용 가능 시 'cuda'로 변경 가능)
            - 정규화 활성화 (유사도 검색 최적화)

        예시:
            >>> em = EmbeddingManager()
            >>> embeddings = em.get_embeddings_model()
            >>> type(embeddings)
            <class 'langchain_community.embeddings.huggingface.HuggingFaceEmbeddings'>
        """
        if self._embeddings is None:
            logger.info(f"[모델 로딩] HuggingFace 임베딩 모델: {self.model_name}")

            try:
                # HuggingFace 임베딩 초기화 (FREE!)
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': self.device},
                    encode_kwargs={'normalize_embeddings': True}
                )

                logger.info(f"[모델 로딩 완료] {self.model_name}")

            except Exception as e:
                logger.error(f"[모델 로딩 실패] {e}")
                raise

        return self._embeddings

    def embed_text(self, text: str, max_retries: int = DEFAULT_MAX_RETRIES) -> List[float]:
        """
        단일 텍스트 임베딩 (재시도 로직 포함)

        Args:
            text: 임베딩할 텍스트
            max_retries: 최대 재시도 횟수 (기본값: 3)

        Returns:
            List[float]: 임베딩 벡터 (384차원)

        Raises:
            Exception: max_retries 초과 후 임베딩 실패 시

        재시도 로직:
            - Exponential backoff: 2^attempt 초 대기
            - 1차 실패: 2초 대기 후 재시도
            - 2차 실패: 4초 대기 후 재시도
            - 3차 실패: 예외 발생

        예시:
            >>> em = EmbeddingManager()
            >>> embedding = em.embed_text("transformer models for NLP")
            >>> len(embedding)
            384
        """
        for attempt in range(max_retries):
            try:
                embeddings_model = self.get_embeddings_model()
                embedding = embeddings_model.embed_query(text)

                logger.debug(
                    f"[임베딩 완료] 텍스트 길이: {len(text)}, 차원: {len(embedding)}"
                )
                return embedding

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = EXPONENTIAL_BACKOFF_BASE ** attempt
                    logger.warning(
                        f"[임베딩 재시도] 시도 {attempt + 1}/{max_retries}, "
                        f"{wait_time}초 후 재시도: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"[임베딩 실패] {max_retries}회 시도 후 실패: {e}"
                    )
                    raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        배치 텍스트 임베딩 (효율성)

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (기본값: 100)
            show_progress: 진행 상황 로깅 여부 (기본값: False)

        Returns:
            List[List[float]]: 임베딩 벡터 리스트

        동작:
            1. texts를 batch_size 단위로 분할
            2. 각 배치를 embed_documents()로 임베딩
            3. 모든 배치 결과 통합 반환

        예시:
            >>> em = EmbeddingManager()
            >>> texts = ["text 1", "text 2", "text 3"]
            >>> embeddings = em.embed_batch(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            384
        """
        embeddings_model = self.get_embeddings_model()
        all_embeddings = []

        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                logger.info(
                    f"[배치 임베딩] {batch_num}/{total_batches} "
                    f"({len(batch)}개 텍스트)"
                )

            try:
                batch_embeddings = embeddings_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"[배치 완료] 배치 {batch_num} 임베딩 성공")

            except Exception as e:
                logger.error(f"[배치 실패] 배치 {batch_num}: {e}")
                raise

        logger.info(f"[임베딩 완료] 총 {len(texts)}개 텍스트 임베딩됨")
        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """
        임베딩 벡터 차원 반환

        Returns:
            int: 임베딩 차원 (all-MiniLM-L6-v2: 384)

        동작:
            샘플 텍스트 "test"를 임베딩하여 차원 확인

        예시:
            >>> em = EmbeddingManager()
            >>> dim = em.get_embedding_dimension()
            >>> print(dim)
            384
        """
        # 샘플 텍스트로 차원 확인
        sample_embedding = self.embed_text("test")
        dimension = len(sample_embedding)

        logger.info(f"[임베딩 차원] {dimension}")
        return dimension

    @property
    def model(self) -> str:
        """
        현재 모델명 반환

        Returns:
            str: 모델명
        """
        return self.model_name

    def __repr__(self) -> str:
        """
        문자열 표현

        Returns:
            str: 포매팅된 문자열
        """
        return f"EmbeddingManager(model='{self.model_name}', device='{self.device}')"
