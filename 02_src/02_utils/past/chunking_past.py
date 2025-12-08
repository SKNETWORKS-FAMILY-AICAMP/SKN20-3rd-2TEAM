"""
RAG 기반 HuggingFace Papers 챗봇 텍스트 청킹 모듈

주요 기능:
1. RecursiveCharacterTextSplitter를 사용한 문서 청킹
2. Chunk 결과를 .pkl 파일로 저장/로딩
3. 메타데이터 보존 (doc_id, year, week, upvote 등)
4. 청킹 통계 및 검증

청킹 설정:
- chunk_size: 100 (기본값, 논문 초록 기준 적절한 크기)
- chunk_overlap: 10 (컨텍스트 보존)
- separators: ["\n\n", "\n", ". ", " ", ""] (문단 우선 분리)

청킹 전략:
- 문단 우선 분리로 의미 단위 보존
- 메타데이터 상속 (원본 Document의 metadata 유지)
- Chunk index 추가 (같은 문서의 청크 추적)

Version: 2.0 (Simplified)
Author: SKN20-3rd-2TEAM
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documents import load_all_documents, load_documents_by_week


# ==================== 전역 설정 (Inline constants) ====================

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "01_data" / "chunks"

# 청킹 설정
DEFAULT_CHUNK_SIZE = 100
DEFAULT_CHUNK_OVERLAP = 10
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# 파일 경로
DEFAULT_CHUNKS_PKL = "chunks_all.pkl"


# ==================== 청킹 함수 ====================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None
) -> List[Document]:
    """
    LangChain Document 리스트를 청크로 분할

    Args:
        documents: 원본 Document 리스트
        chunk_size: 청크 크기 (기본값: 500)
        chunk_overlap: 청크 오버랩 크기 (기본값: 50)
        separators: 분리자 리스트 (기본값: ["\n\n", "\n", ". ", " ", ""])

    Returns:
        List[Document]: 청크된 Document 리스트 (메타데이터 보존)

    동작:
        1. RecursiveCharacterTextSplitter 초기화
        2. 각 문서를 청크로 분할
        3. 각 청크에 chunk_index 메타데이터 추가
        4. 원본 메타데이터 상속

    예시:
        >>> from utils import load_all_documents, chunk_documents
        >>> docs = load_all_documents()
        >>> chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        >>> print(f"원본: {len(docs)}개 → 청크: {len(chunks)}개")
        원본: 506개 → 청크: 1523개
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    logger.info(
        f"[청킹 시작] {len(documents)}개 문서, "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}"
    )

    # RecursiveCharacterTextSplitter 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len
    )

    # 문서 청킹
    all_chunks = []

    for doc_idx, doc in enumerate(documents):
        try:
            # 문서를 청크로 분할
            chunks = text_splitter.split_documents([doc])

            # 각 청크에 chunk_index 추가
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = chunk_idx
                chunk.metadata["total_chunks"] = len(chunks)
                all_chunks.append(chunk)

            if (doc_idx + 1) % 100 == 0:
                logger.debug(f"[청킹 진행] {doc_idx + 1}/{len(documents)}개 문서 처리됨")

        except Exception as e:
            logger.error(f"[청킹 에러] 문서 {doc_idx}: {e}")
            # 에러 발생 시 원본 문서를 그대로 추가 (chunk_index=0)
            doc.metadata["chunk_index"] = 0
            doc.metadata["total_chunks"] = 1
            all_chunks.append(doc)

    logger.info(
        f"[청킹 완료] {len(documents)}개 문서 → {len(all_chunks)}개 청크 "
        f"(평균 {len(all_chunks)/len(documents):.1f}개/문서)"
    )

    return all_chunks


def save_chunks_to_pkl(
    chunks: List[Document],
    filename: str = DEFAULT_CHUNKS_PKL,
    output_dir: Optional[Path] = None
) -> str:
    """
    청크 리스트를 .pkl 파일로 저장

    Args:
        chunks: 저장할 Document 청크 리스트
        filename: 출력 파일명 (기본값: "chunks_all.pkl")
        output_dir: 출력 디렉토리 (기본값: 01_data/chunks/)

    Returns:
        str: 저장된 파일 경로

    동작:
        1. 출력 디렉토리 생성 (없으면)
        2. pickle로 청크 리스트 직렬화
        3. .pkl 파일로 저장

    예시:
        >>> chunks = chunk_documents(docs)
        >>> filepath = save_chunks_to_pkl(chunks, "chunks_2025_w49.pkl")
        >>> print(f"저장 완료: {filepath}")
    """
    if output_dir is None:
        output_dir = CHUNKS_DIR

    # 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    logger.info(f"[저장 시작] {len(chunks)}개 청크를 {output_path}에 저장 중...")

    try:
        with open(output_path, 'wb') as f:
            pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"[저장 완료] {output_path} ({file_size:.2f} MB)")

        return str(output_path)

    except Exception as e:
        logger.error(f"[저장 실패] {e}")
        raise


def load_chunks_from_pkl(
    filename: str = DEFAULT_CHUNKS_PKL,
    chunks_dir: Optional[Path] = None
) -> List[Document]:
    """
    .pkl 파일에서 청크 리스트 로딩

    Args:
        filename: 로딩할 파일명 (기본값: "chunks_all.pkl")
        chunks_dir: 청크 디렉토리 (기본값: 01_data/chunks/)

    Returns:
        List[Document]: 로딩된 Document 청크 리스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우

    예시:
        >>> chunks = load_chunks_from_pkl("chunks_all.pkl")
        >>> print(f"로딩 완료: {len(chunks)}개 청크")
    """
    if chunks_dir is None:
        chunks_dir = CHUNKS_DIR

    pkl_path = chunks_dir / filename

    if not pkl_path.exists():
        raise FileNotFoundError(f"[파일 없음] {pkl_path}")

    logger.info(f"[로딩 시작] {pkl_path}")

    try:
        with open(pkl_path, 'rb') as f:
            chunks = pickle.load(f)

        logger.info(f"[로딩 완료] {len(chunks)}개 청크")
        return chunks

    except Exception as e:
        logger.error(f"[로딩 실패] {e}")
        raise


def chunk_and_save(
    year: Optional[int] = None,
    week: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    filename: Optional[str] = None
) -> str:
    """
    문서 로딩 → 청킹 → 저장 (원스텝)

    Args:
        year: 연도 (None이면 전체 문서 로딩)
        week: 주차 (None이면 전체 문서 로딩)
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        filename: 출력 파일명 (자동 생성 가능)

    Returns:
        str: 저장된 파일 경로

    동작:
        1. 문서 로딩 (load_all_documents 또는 load_documents_by_week)
        2. 청킹 (chunk_documents)
        3. 저장 (save_chunks_to_pkl)

    예시:
        >>> # 전체 문서 청킹
        >>> filepath = chunk_and_save()
        >>> print(f"저장됨: {filepath}")

        >>> # 특정 주차만 청킹
        >>> filepath = chunk_and_save(year=2025, week=49, filename="chunks_2025_w49.pkl")
    """
    # 문서 로딩
    if year and week:
        logger.info(f"[문서 로딩] {year}-W{week:02d}")
        documents = load_documents_by_week(year, week, validate=False)
        default_filename = f"chunks_{year}_w{week:02d}.pkl"
    else:
        logger.info("[문서 로딩] 전체 문서")
        documents = load_all_documents(validate=False)
        default_filename = "chunks_all.pkl"

    if not documents:
        logger.warning("[경고] 로딩된 문서가 없습니다")
        return ""

    # 청킹
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 저장
    output_filename = filename or default_filename
    filepath = save_chunks_to_pkl(chunks, filename=output_filename)

    return filepath


# ==================== 통계 ====================

def get_chunk_statistics(chunks: List[Document]) -> Dict[str, Any]:
    """
    청크 통계 정보 조회

    Args:
        chunks: Document 청크 리스트

    Returns:
        Dict: 통계 정보
            - total_chunks: 총 청크 개수
            - avg_chunk_length: 평균 청크 길이
            - min_chunk_length: 최소 청크 길이
            - max_chunk_length: 최대 청크 길이
            - unique_docs: 유니크 문서 개수

    예시:
        >>> stats = get_chunk_statistics(chunks)
        >>> print(f"총 청크: {stats['total_chunks']}")
        >>> print(f"평균 길이: {stats['avg_chunk_length']:.0f}자")
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "unique_docs": 0
        }

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    unique_doc_ids = set(chunk.metadata.get("doc_id") for chunk in chunks)

    stats = {
        "total_chunks": len(chunks),
        "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
        "min_chunk_length": min(chunk_lengths),
        "max_chunk_length": max(chunk_lengths),
        "unique_docs": len(unique_doc_ids)
    }

    logger.info(
        f"[청크 통계] 총 {stats['total_chunks']}개 청크, "
        f"평균 {stats['avg_chunk_length']:.0f}자, "
        f"{stats['unique_docs']}개 문서"
    )

    return stats


# ==================== 유틸리티 ====================

def list_chunk_files(chunks_dir: Optional[Path] = None) -> List[str]:
    """
    저장된 청크 파일 목록 조회

    Args:
        chunks_dir: 청크 디렉토리 (기본값: 01_data/chunks/)

    Returns:
        List[str]: .pkl 파일명 리스트

    예시:
        >>> files = list_chunk_files()
        >>> print(files)
        ['chunks_all.pkl', 'chunks_2025_w49.pkl']
    """
    if chunks_dir is None:
        chunks_dir = CHUNKS_DIR

    if not chunks_dir.exists():
        logger.warning(f"[디렉토리 없음] {chunks_dir}")
        return []

    pkl_files = sorted([f.name for f in chunks_dir.glob("*.pkl")])

    logger.info(f"[청크 파일 목록] {len(pkl_files)}개 파일")
    return pkl_files


if __name__ == "__main__":
    # 테스트용 메인 실행부
    import logging
    logging.basicConfig(level=logging.INFO)

    # 전체 문서 청킹 및 저장
    print("[테스트] 전체 문서 청킹 시작...")
    filepath = chunk_and_save(chunk_size=100, chunk_overlap=10)
    print(f"✓ 저장 완료: {filepath}")

    # 청크 로딩 및 통계
    print("\n[테스트] 청크 로딩 및 통계...")
    chunks = load_chunks_from_pkl()
    stats = get_chunk_statistics(chunks)

    print(f"✓ 총 청크: {stats['total_chunks']}개")
    print(f"✓ 평균 길이: {stats['avg_chunk_length']:.0f}자")
    print(f"✓ 원본 문서: {stats['unique_docs']}개")

    # 샘플 청크 출력
    print("\n[테스트] 샘플 청크 (첫 3개):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- 청크 {i+1} ---")
        print(f"doc_id: {chunk.metadata.get('doc_id')}")
        print(f"chunk_index: {chunk.metadata.get('chunk_index')}/{chunk.metadata.get('total_chunks')}")
        print(f"길이: {len(chunk.page_content)}자")
        print(f"내용 미리보기: {chunk.page_content[:100]}...")
