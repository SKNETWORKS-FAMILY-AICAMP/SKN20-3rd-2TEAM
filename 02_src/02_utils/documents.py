"""
RAG 기반 HuggingFace Papers 챗봇 데이터 로더 모듈

주요 기능:
1. JSON 문서 파일 로딩 (01_data/documents 디렉토리)
2. 파일명 기반 doc_id 파싱 (doc2549001.json → year=2025, week=49, index=1)
3. JSON to LangChain Document 변환 (metadata 보강)
4. Upvote 필드 정규화 (string/int 혼재 처리)
5. 주차별/전체 문서 일괄 로딩
6. 필터링 기반 문서 조회 (year, week, upvote, tags)

문서 구조:
- 경로: 01_data/documents/{year}/{year}-W{week}/doc{YY}{ww}{NNN}.json
- 스키마: { context: str, metadata: { paper_name, github_url, huggingface_url, upvote, tags } }
- 예시: doc2549001.json → doc_id=doc2549001, year=2025, week=49, index=1

데이터 통계:
- 총 506개 논문 (2025년 W45~W49, 5주)
- Upvote 필드: string/int 혼재 (정규화 필수)
- Tags: 3개 키워드 (TF-IDF + Lemmatization 추출)

Version: 2.0 (Simplified)
Author: SKN20-3rd-2TEAM
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Union, Optional

from langchain_core.documents import Document

# import validators


# ==================== 전역 설정 (Inline constants) ====================

logger = logging.getLogger(__name__)

# Project paths (no config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "01_data" / "documents"

# 파일명 패턴 (docYYWWNNN.json)
DOC_ID_PATTERN = r"doc(\d{2})(\d{2})(\d{3})\.json"

# 주차 디렉토리 패턴 (YYYY-WWW)
WEEK_DIR_PATTERN = r"\d{4}-W(\d{2})"


# ==================== 유틸리티 함수 ====================

def parse_doc_id(filename: str) -> Dict[str, Union[str, int]]:
    """
    파일명에서 doc_id 정보 파싱

    Args:
        filename: JSON 파일명 (예: "doc2549001.json")

    Returns:
        Dict: {
            "doc_id": "doc2549001",
            "year": 2025,
            "week": 49,
            "index": 1
        }

    Raises:
        ValueError: 파일명이 패턴과 일치하지 않을 경우

    예시:
        >>> info = parse_doc_id("doc2549001.json")
        >>> print(info["year"])  # 2025
        >>> print(info["week"])  # 49
    """
    match = re.match(DOC_ID_PATTERN, filename)

    if not match:
        raise ValueError(
            f"[파일명 파싱 실패] 잘못된 파일명 형식: {filename}. "
            f"올바른 형식: docYYWWNNN.json"
        )

    yy, ww, nnn = match.groups()

    doc_id_info = {
        "doc_id": filename.replace(".json", ""),
        "year": 2000 + int(yy),  # 25 → 2025
        "week": int(ww),          # 49 → 49
        "index": int(nnn)         # 001 → 1
    }

    return doc_id_info


def normalize_upvote(upvote: Union[str, int]) -> int:
    """
    Upvote 필드 정규화 (string → int)

    CRITICAL: JSON 파일에 upvote 필드가 string("219"), int(191), 또는 "-"로 혼재되어 있음.
    이를 일관되게 int로 변환하는 정규화 함수.

    Args:
        upvote: Upvote 값 (string 또는 int)

    Returns:
        int: 정규화된 upvote 값 (실패 시 0)

    예시:
        >>> normalize_upvote("219")  # 219
        >>> normalize_upvote(191)    # 191
        >>> normalize_upvote("-")    # 0 (데이터 없음 표시)
        >>> normalize_upvote("invalid")  # 0 (경고 로그 출력)
    """
    if isinstance(upvote, int):
        return upvote

    if isinstance(upvote, str):
        # '-' 문자는 데이터 없음을 의미하므로 0으로 처리
        if upvote.strip() == '-':
            return 0
        try:
            return int(upvote)
        except ValueError:
            logger.warning(
                f"[UPVOTE 변환 실패] 문자열을 int로 변환할 수 없습니다: '{upvote}'. 기본값 0 반환"
            )
            return 0

    logger.warning(
        f"[UPVOTE 타입 오류] 잘못된 upvote 타입: {type(upvote)}. 기본값 0 반환"
    )
    return 0


# ==================== JSON 로딩 ====================

def load_json_document(file_path: str) -> Dict[str, Any]:
    """
    단일 JSON 문서 파일 로딩 및 검증

    Args:
        file_path: JSON 파일 절대 경로

    Returns:
        Dict: 검증된 JSON 문서 딕셔너리

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        json.JSONDecodeError: JSON 파싱 실패 시
        ValidationError: 필수 필드 누락 시

    예시:
        >>> data = load_json_document("01_data/documents/2025/2025-W49/doc2549001.json")
        >>> print(data["metadata"]["paper_name"])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"[파일 없음] JSON 파일을 찾을 수 없습니다: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"[JSON 파싱 에러] {file_path}: {e}")
        raise

    # Pydantic 모델로 검증
    # validators.validate_json_document(data)

    return data


def json_to_langchain_document(
    json_data: Dict[str, Any],
    doc_id_info: Dict[str, Union[str, int]]
) -> Document:
    """
    JSON 딕셔너리를 LangChain Document 객체로 변환

    Args:
        json_data: 로딩된 JSON 딕셔너리 (context, metadata 포함)
        doc_id_info: 파일명 파싱 결과 (doc_id, year, week, index)

    Returns:
        Document: LangChain Document 객체 (page_content, metadata)

    변환 과정:
        1. context → page_content
        2. metadata 복사 및 보강 (doc_id, year, week, index 추가)
        3. upvote 필드 정규화 (string → int)

    예시:
        >>> data = load_json_document("doc2549001.json")
        >>> doc_id_info = parse_doc_id("doc2549001.json")
        >>> doc = json_to_langchain_document(data, doc_id_info)
        >>> print(doc.metadata["doc_id"])  # "doc2549001"
        >>> print(doc.metadata["year"])    # 2025
    """
    # page_content 추출
    page_content = json_data["context"]

    # metadata 복사 및 보강
    metadata = json_data["metadata"].copy()

    # Upvote 필드 정규화 (CRITICAL)
    metadata["upvote"] = normalize_upvote(metadata["upvote"])

    # doc_id 정보 추가
    metadata["doc_id"] = doc_id_info["doc_id"]
    metadata["year"] = doc_id_info["year"]
    metadata["week"] = doc_id_info["week"]
    metadata["index"] = doc_id_info["index"]

    # LangChain Document 생성
    document = Document(
        page_content=page_content,
        metadata=metadata
    )

    return document


# ==================== 주차별 로딩 ====================

def load_documents_by_week(
    year: int,
    week: int,
    validate: bool = True
) -> List[Document]:
    """
    특정 주차의 모든 문서 로딩

    Args:
        year: 연도 (예: 2025)
        week: 주차 번호 (1~53)
        validate: 검증 여부 (기본값: True, False 시 에러 무시)

    Returns:
        List[Document]: LangChain Document 리스트

    경로 구조:
        01_data/documents/{year}/{year}-W{week:02d}/*.json

    예시:
        >>> docs = load_documents_by_week(2025, 49)
        >>> print(f"[로딩 완료] {len(docs)}개 문서")
    """
    week_str = f"{year}-W{week:02d}"
    week_dir = DOCUMENTS_DIR / str(year) / week_str

    if not week_dir.exists():
        logger.warning(f"[디렉토리 없음] 주차 디렉토리를 찾을 수 없습니다: {week_dir}")
        return []

    logger.info(f"[로딩 시작] {week_str} 문서 로딩 중...")

    documents = []
    json_files = sorted(week_dir.glob("*.json"))

    for json_file in json_files:
        try:
            # 파일명 파싱
            doc_id_info = parse_doc_id(json_file.name)

            # JSON 로딩
            json_data = load_json_document(str(json_file))

            # Document 변환
            document = json_to_langchain_document(json_data, doc_id_info)

            documents.append(document)

        except Exception as e:
            logger.error(f"[로딩 에러] {json_file.name}: {e}")
            if validate:
                raise
            continue

    logger.info(f"[로딩 완료] {week_str}: {len(documents)}개 문서")
    return documents


def load_documents_batch(
    year: int,
    start_week: int,
    end_week: int,
    batch_size: int = 100
) -> Generator[List[Document], None, None]:
    """
    주차 범위의 문서를 배치 단위로 로딩 (Generator)

    Args:
        year: 연도 (예: 2025)
        start_week: 시작 주차 (예: 45)
        end_week: 종료 주차 (예: 49)
        batch_size: 배치 크기 (기본값: 100)

    Yields:
        List[Document]: 배치 단위의 Document 리스트

    용도:
        대량 문서 인덱싱 시 메모리 효율을 위해 배치 단위로 처리

    예시:
        >>> for batch in load_documents_batch(2025, 45, 49, batch_size=100):
        ...     print(f"[배치 처리] {len(batch)}개 문서")
        ...     # ChromaDB에 배치 인덱싱
    """
    current_batch = []

    for week in range(start_week, end_week + 1):
        try:
            docs = load_documents_by_week(year, week, validate=False)

            for doc in docs:
                current_batch.append(doc)

                if len(current_batch) >= batch_size:
                    logger.debug(f"[배치 생성] {len(current_batch)}개 문서 배치 반환")
                    yield current_batch
                    current_batch = []

        except Exception as e:
            logger.error(f"[로딩 에러] {year}-W{week:02d}: {e}")
            continue

    # 남은 문서 반환
    if current_batch:
        logger.debug(f"[최종 배치] {len(current_batch)}개 문서 배치 반환")
        yield current_batch


def load_all_documents(validate: bool = True, count: int = 6) -> List[Document]:
    """
    최근 count 주차의 전체 문서 로딩

    Args:
        validate: 검증 여부 (기본값: True, False 시 에러 무시)
        count: 로딩할 주차 수 (기본값: 6, 최근 6주차 로딩)

    Returns:
        List[Document]: 전체 Document 리스트 (506개 문서)

    동작:
        1. DOCUMENTS_DIR 스캔 (01_data/documents/)
        2. 연도별로 주차 디렉토리 정렬
        3. 최근 count 주차 문서 로딩 및 통합
    """
    logger.info(f"[로딩 시작] 최근 {count}주차 문서 로딩 중...")

    all_documents = []

    # Documents 디렉토리 존재 확인
    if not DOCUMENTS_DIR.exists():
        logger.error(f"[디렉토리 없음] Documents 디렉토리를 찾을 수 없습니다: {DOCUMENTS_DIR}")
        return []

    # Year 디렉토리 스캔
    cnt = 0; stop = False
    for year_dir in sorted(DOCUMENTS_DIR.iterdir(), reverse=True):
        if not year_dir.is_dir():
            continue

        try:
            year = int(year_dir.name)
        except ValueError:
            logger.warning(f"[스킵] 연도 형식이 아닌 디렉토리: {year_dir.name}")
            continue

        # Week 디렉토리 스캔 (연도 안에서는 최신 주차부터)
        for week_dir in sorted(year_dir.iterdir(), reverse=True):
            if not week_dir.is_dir():
                continue

            # 주차 번호 추출 (예: "2025-W49" → 49)
            match = re.match(WEEK_DIR_PATTERN, week_dir.name)
            if not match:
                 logger.warning(f"[스킵] 주차 형식이 아닌 디렉토리: {week_dir.name}")
                 continue

            week = int(match.group(1))

            try:
                docs = load_documents_by_week(year, week, validate=validate)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"[로딩 에러] {year}-W{week:02d}: {e}")
                if validate:
                    raise
                continue

            cnt += 1
            if cnt >= count :
                stop = True
                break

        if stop :
            break

    logger.info(f"[전체 로딩 완료] 총 {len(all_documents)}개 문서 로딩됨")
    return all_documents


# ==================== 통계 ====================

def get_document_statistics() -> Dict[str, Any]:
    """
    문서 통계 정보 조회

    Returns:
        Dict: 통계 정보 딕셔너리
            - total_documents: 총 문서 개수
            - weeks: 사용 가능한 주차 리스트 (예: ["2025-W45", ...])
            - by_week: 주차별 문서 개수 (예: {"2025-W49": 102, ...})
            - by_year: 연도별 문서 개수 (예: {2025: 506, ...})

    예시:
        >>> stats = get_document_statistics()
        >>> print(f"총 문서: {stats['total_documents']}")
        >>> print(f"주차 목록: {stats['weeks']}")
        >>> print(f"2025-W49: {stats['by_week']['2025-W49']}개")
    """
    logger.info("[통계 조회] 문서 통계 계산 중...")

    stats = {
        "total_documents": 0,
        "weeks": [],
        "by_week": {},
        "by_year": {}
    }

    if not DOCUMENTS_DIR.exists():
        logger.warning(f"[디렉토리 없음] {DOCUMENTS_DIR}")
        return stats

    # Year 디렉토리 스캔
    for year_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not year_dir.is_dir():
            continue

        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        year_count = 0

        # Week 디렉토리 스캔
        for week_dir in sorted(year_dir.iterdir()):
            if not week_dir.is_dir():
                continue

            match = re.match(WEEK_DIR_PATTERN, week_dir.name)
            if not match:
                continue

            week = int(match.group(1))
            week_str = f"{year}-W{week:02d}"

            # JSON 파일 개수 카운트
            json_files = list(week_dir.glob("*.json"))
            count = len(json_files)

            stats["weeks"].append(week_str)
            stats["by_week"][week_str] = count
            year_count += count

        if year_count > 0:
            stats["by_year"][year] = year_count
            stats["total_documents"] += year_count

    logger.info(
        f"[통계 조회 완료] 총 {stats['total_documents']}개 문서, "
        f"{len(stats['weeks'])}개 주차"
    )
    return stats


# ==================== 필터링 ====================

def load_documents_by_filter(
    year: Optional[int] = None,
    week: Optional[int] = None,
    min_upvote: Optional[int] = None,
    tags: Optional[List[str]] = None
) -> List[Document]:
    """
    필터 조건 기반 문서 조회

    Args:
        year: 연도 필터 (예: 2025)
        week: 주차 필터 (예: 49)
        min_upvote: 최소 upvote 임계값 (예: 100)
        tags: 태그 필터 (리스트 중 하나라도 매칭) (예: ["transformers", "vision"])

    Returns:
        List[Document]: 필터링된 Document 리스트

    필터링 로직:
        1. year, week로 문서 범위 선택
        2. min_upvote로 upvote 필터링
        3. tags로 태그 필터링 (OR 조건: 하나라도 매칭)

    예시:
        >>> # 2025년 W49, upvote 100 이상, "transformers" 태그
        >>> docs = load_documents_by_filter(
        ...     year=2025,
        ...     week=49,
        ...     min_upvote=100,
        ...     tags=["transformers"]
        ... )
        >>> print(f"[필터링 완료] {len(docs)}개 문서")
    """
    # 기본 문서 로딩
    if year and week:
        documents = load_documents_by_week(year, week, validate=False)
    elif year:
        documents = []
        for w in range(1, 54):
            try:
                docs = load_documents_by_week(year, w, validate=False)
                documents.extend(docs)
            except:
                continue
    else:
        documents = load_all_documents(validate=False)

    original_count = len(documents)

    # 필터 적용
    filtered_documents = []

    for doc in documents:
        # Upvote 필터
        if min_upvote is not None:
            if doc.metadata.get("upvote", 0) < min_upvote:
                continue

        # Tags 필터 (OR 조건)
        if tags:
            doc_tags = doc.metadata.get("tags", [])
            if not any(tag in doc_tags for tag in tags):
                continue

        filtered_documents.append(doc)

    logger.info(
        f"[필터링 완료] {len(filtered_documents)}개 문서 (원본: {original_count}개)"
    )
    return filtered_documents


# Alias for compatibility
get_statistics = get_document_statistics


if __name__ == "__main__":
    # 테스트용 메인 실행부
    stats = get_document_statistics()
    print(f"총 문서 수: {stats['total_documents']}")
    print(f"주차별 문서 수: {stats['by_week']}")

    # load_documents_batch 테스트
    for batch in load_documents_batch(2025, 45, 49, batch_size=100):
        print(f"배치 크기: {len(batch)}")

    # 데이터 정상적으로 불러오는지 확인
    all_docs = load_all_documents()
    print(f"전체 문서 로딩: {len(all_docs)}개")
    print(f"문서 head 초록: {[doc.page_content for doc in all_docs[:5]]}")
    print(f"문서 head metadata: {[doc.metadata for doc in all_docs[:5]]}") 