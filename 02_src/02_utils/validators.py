"""
RAG 기반 HuggingFace Papers 챗봇 데이터 검증 모듈

주요 기능:
1. Pydantic 모델 기반 타입 안전 데이터 검증
2. Upvote 필드 정규화 (string → int 자동 변환)
3. 논문 메타데이터 검증 (paper_name, urls, tag1)
4. 배치 검증 및 에러 리포트 생성
5. 디렉토리 전체 검증 및 통계 제공

검증 스키마:
- PaperMetadata: paper_name, github_url, huggingface_url, upvote, tag1, tag2, tag3
- PaperDocument: context(최소 50자), metadata
- DocIdInfo: doc_id(doc2549001), year, week, index

검증 규칙:
- context: 최소 50자 이상, 공백 제외
- upvote: 0 이상의 정수 (string "219" → int 219 자동 변환)
- tag1: string, 비어있지 않음
- tag2: string, 비어있지 않음
- tag3: string, 비어있지 않음
- huggingface_url: https://huggingface.co/papers/ 형식 필수
- github_url: https://github.com/ 형식 (선택사항, 빈 문자열 허용)

Version: 1.0
Author: SKN20-3rd-2TEAM
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator, ValidationError


# ==================== 전역 설정 ====================

logger = logging.getLogger(__name__)

# URL 패턴
HUGGINGFACE_URL_PATTERN = r"^https://huggingface\.co/papers/"
# GITHUB_URL_PATTERN = r"^https://github\.com/[\w\-\.]+/[\w\-\.]+"
DOC_ID_PATTERN = r"^doc\d{7}$"
DOC_FILENAME_PATTERN = r"doc(\d{2})(\d{2})(\d{3})\.json"


# ==================== Pydantic 검증 모델 ====================

class PaperMetadata(BaseModel):
    """
    논문 메타데이터 검증 모델

    필드:
        paper_name: 논문 제목 (최소 1자)
        github_url: GitHub 저장소 URL (선택사항, 빈 문자열 허용)
        huggingface_url: HuggingFace 논문 URL (필수)
        upvote: Upvote 개수 (0 이상의 정수, string → int 자동 변환)
        tag1: 키워드 문자열
        tag2: 키워드 문자열
        tag3: 키워드 문자열

    예시:
        >>> metadata = PaperMetadata(
        ...     paper_name="Attention Is All You Need",
        ...     github_url="https://github.com/tensorflow/tensor2tensor",
        ...     huggingface_url="https://huggingface.co/papers/1706.03762",
        ...     upvote=219,
        ...     tag1="transformers", 
        ...     tag2="attention",
        ...     tag3="nlp"
        ... )
    """

    paper_name: str = Field(
        ...,
        min_length=1,
        description="논문 제목"
    )

    # github_url: str = Field(
    #     default="",
    #     description="GitHub 저장소 URL (선택사항)"
    # )

    huggingface_url: str = Field(
        ...,
        pattern=HUGGINGFACE_URL_PATTERN,
        description="HuggingFace 논문 URL"
    )

    upvote: int = Field(
        ...,
        ge=0,
        description="Upvote 개수 (0 이상)"
    )

    tag1: str = Field(
        ...,
        description="키워드"
    )

    tag2: str = Field(
        ...,
        description="키워드"
    )

    tag3: str = Field(
        ...,
        description="키워드"
    )

    @validator("upvote", pre=True)
    def normalize_upvote(cls, v):
        """
        Upvote 필드 정규화 (string → int 자동 변환)

        CRITICAL: JSON 파일에 upvote가 string("219"), int(191), 또는 "-"로 혼재됨.
        Pydantic validator를 사용하여 자동 변환.

        Args:
            v: Upvote 값 (string 또는 int)

        Returns:
            int: 정규화된 upvote 값 (실패 시 0)
        """
        if isinstance(v, str):
            # '-' 문자는 0으로 처리
            if v.strip() == '-':
                return 0
            try:
                return int(v)
            except ValueError:
                logger.warning(f"[UPVOTE 변환 실패] 문자열을 int로 변환할 수 없습니다: {v}. 기본값 0 반환")
                return 0
        elif isinstance(v, int):
            return v
        else:
            logger.warning(f"[UPVOTE 타입 오류] 잘못된 타입: {type(v)}. 기본값 0 반환")
            return 0

    @validator("tag1", "tag2", "tag3")
    def validate_tags(cls, v):
        """
        Tag 검증

        Args:
            v: Tag값

        Returns:
            str: 검증된 tag

        Raises:
            ValueError: 태그가 비어있거나 문자열이 아닌 경우
        """
        if not v or not isinstance(v, str):
            raise ValueError(f"[태그 검증 실패] 잘못된 태그: {v}")
        return v
    class Config:
        extra = "allow"  # 추가 필드 허용 (유연성)


class PaperDocument(BaseModel):
    """
    완전한 논문 문서 검증 모델

    필드:
        context: 논문 요약 (Abstract, 최소 50자)
        metadata: 논문 메타데이터 (PaperMetadata)

    예시:
        >>> doc = PaperDocument(
        ...     context="This paper introduces the Transformer...",
        ...     metadata={
        ...         "paper_name": "Attention Is All You Need",
        ...         "huggingface_url": "https://huggingface.co/papers/1706.03762",
        ...         "upvote": 219,
        ...         "tag1": "transformers",
        ...         "tag2": "attention",
        ...         "tag3": "nlp"
        ...     }
        ... )
    """

    context: str = Field(
        ...,
        min_length=50,
        description="논문 요약 (Abstract, 최소 50자)"
    )

    metadata: PaperMetadata

    @validator("context")
    def validate_context(cls, v):
        """
        Context 검증 (공백 제외 최소 50자)

        Args:
            v: Context 문자열

        Returns:
            str: 검증된 context

        Raises:
            ValueError: 공백만 있거나 비어있는 경우
        """
        if not v.strip():
            raise ValueError("[검증 실패] Context가 비어있거나 공백만 포함됩니다")
        return v

    class Config:
        extra = "forbid"  # 추가 필드 금지 (엄격한 검증)


class DocIdInfo(BaseModel):
    """
    파싱된 doc_id 정보 검증 모델

    필드:
        doc_id: 문서 ID (doc2549001 형식)
        year: 연도 (2020~2100)
        week: 주차 번호 (1~53)
        index: 논문 인덱스 (1~999)

    예시:
        >>> doc_id_info = DocIdInfo(
        ...     doc_id="doc2549001",
        ...     year=2025,
        ...     week=49,
        ...     index=1
        ... )
    """

    doc_id: str = Field(
        ...,
        pattern=DOC_ID_PATTERN,
        description="문서 ID (doc2549001)"
    )

    year: int = Field(
        ...,
        ge=2020,
        le=2100,
        description="연도"
    )

    week: int = Field(
        ...,
        ge=1,
        le=53,
        description="주차 번호"
    )

    index: int = Field(
        ...,
        ge=1,
        le=999,
        description="논문 인덱스"
    )

    class Config:
        extra = "forbid"


class ValidationReport(BaseModel):
    """
    검증 결과 리포트 모델

    필드:
        total_documents: 총 문서 개수
        valid_documents: 유효한 문서 개수
        invalid_documents: 유효하지 않은 문서 개수
        errors: 에러 목록 (doc_id, file, error)

    메서드:
        success_rate(): 검증 성공률 계산 (0.0~1.0)

    예시:
        >>> report = ValidationReport(
        ...     total_documents=102,
        ...     valid_documents=100,
        ...     invalid_documents=2,
        ...     errors=[{"doc_id": "doc2549001", "error": "..."}]
        ... )
        >>> print(f"성공률: {report.success_rate():.1%}")  # 98.0%
    """

    total_documents: int = Field(..., ge=0)
    valid_documents: int = Field(..., ge=0)
    invalid_documents: int = Field(..., ge=0)
    errors: List[Dict[str, str]] = Field(default_factory=list)

    def success_rate(self) -> float:
        """
        검증 성공률 계산

        Returns:
            float: 성공률 (0.0~1.0)
        """
        if self.total_documents == 0:
            return 0.0
        return self.valid_documents / self.total_documents

    def __str__(self) -> str:
        """
        리포트 문자열 표현

        Returns:
            str: 포매팅된 리포트 문자열
        """
        return (
            f"ValidationReport(\n"
            f"  총 문서: {self.total_documents},\n"
            f"  유효: {self.valid_documents},\n"
            f"  무효: {self.invalid_documents},\n"
            f"  성공률: {self.success_rate():.1%}\n"
            f")"
        )


# ==================== 검증 함수 ====================

def validate_json_document(json_data: dict) -> PaperDocument:
    """
    JSON 문서 구조 검증 (Pydantic 사용)

    Args:
        json_data: 원본 JSON 딕셔너리

    Returns:
        PaperDocument: 검증된 PaperDocument 인스턴스

    Raises:
        ValidationError: 검증 실패 시

    예시:
        >>> data = {"context": "...", "metadata": {...}}
        >>> validated = validate_json_document(data)
        >>> print(validated.metadata.upvote)  # 정규화된 int
    """
    try:
        validated = PaperDocument(**json_data)
        return validated
    except ValidationError as e:
        logger.error(f"[검증 실패] {e}")
        raise


def validate_document_batch(documents: List[dict]) -> List[PaperDocument]:
    """
    배치 문서 검증

    Args:
        documents: 원본 JSON 딕셔너리 리스트

    Returns:
        List[PaperDocument]: 검증된 PaperDocument 리스트

    Note:
        유효하지 않은 문서는 경고 로그를 출력하지만 처리를 계속합니다.
        상세한 에러 추적은 ValidationReport를 사용하세요.

    예시:
        >>> docs = [{"context": "...", "metadata": {...}}, ...]
        >>> validated_docs = validate_document_batch(docs)
        >>> print(f"[검증 완료] {len(validated_docs)}개 유효")
    """
    validated_docs = []
    errors = []

    for idx, doc_data in enumerate(documents):
        try:
            validated = PaperDocument(**doc_data)
            validated_docs.append(validated)
        except ValidationError as e:
            errors.append({"index": idx, "error": str(e)})
            logger.warning(f"[검증 실패] 문서 {idx}: {e}")

    if errors:
        logger.warning(
            f"[배치 검증 완료] {len(errors)}개 에러 / {len(documents)}개 문서"
        )

    return validated_docs


def validate_doc_id(filename: str) -> DocIdInfo:
    """
    파일명에서 doc_id 검증 및 파싱

    Args:
        filename: 파일명 (예: "doc2549001.json")

    Returns:
        DocIdInfo: 검증된 DocIdInfo 인스턴스

    Raises:
        ValidationError: 파일명 형식이 잘못된 경우

    예시:
        >>> info = validate_doc_id("doc2549001.json")
        >>> print(info.year, info.week)  # 2025, 49
    """
    # 파일명 파싱
    match = re.match(DOC_FILENAME_PATTERN, filename)
    if not match:
        raise ValueError(f"[파일명 검증 실패] 잘못된 형식: {filename}")

    yy, ww, nnn = match.groups()

    doc_id_data = {
        "doc_id": filename.replace(".json", ""),
        "year": 2000 + int(yy),
        "week": int(ww),
        "index": int(nnn)
    }

    return DocIdInfo(**doc_id_data)


# ==================== 디렉토리 검증 ====================

def validate_directory(
    directory_path: str,
    stop_on_error: bool = False
) -> ValidationReport:
    """
    디렉토리 내 모든 JSON 파일 검증

    Args:
        directory_path: JSON 파일이 포함된 디렉토리 경로
        stop_on_error: 첫 에러 발생 시 중단 여부 (기본값: False)

    Returns:
        ValidationReport: 검증 결과 통계 및 에러 목록

    동작:
        1. 디렉토리 내 모든 *.json 파일 탐색
        2. 각 파일 로딩 및 검증
        3. 성공/실패 카운트 및 에러 수집
        4. ValidationReport 반환

    예시:
        >>> report = validate_directory("01_data/documents/2025/2025-W49")
        >>> print(report)
        >>> print(f"성공률: {report.success_rate():.1%}")
    """
    logger.info(f"[검증 시작] 디렉토리 검증 중: {directory_path}")

    path = Path(directory_path)
    if not path.exists():
        logger.error(f"[디렉토리 없음] {directory_path}")
        return ValidationReport(
            total_documents=0,
            valid_documents=0,
            invalid_documents=0,
            errors=[{
                "directory": str(directory_path),
                "error": "디렉토리를 찾을 수 없습니다"
            }]
        )

    json_files = list(path.glob("*.json"))
    total = len(json_files)
    valid = 0
    errors = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            validate_json_document(data)
            valid += 1

        except ValidationError as e:
            errors.append({
                "doc_id": json_file.stem,
                "file": str(json_file),
                "error": str(e)
            })

            if stop_on_error:
                logger.error(f"[검증 중단] {json_file.name}에서 에러 발생")
                break

        except json.JSONDecodeError as e:
            errors.append({
                "doc_id": json_file.stem,
                "file": str(json_file),
                "error": f"JSON 파싱 에러: {e}"
            })

            if stop_on_error:
                break

        except Exception as e:
            errors.append({
                "doc_id": json_file.stem,
                "file": str(json_file),
                "error": f"예상치 못한 에러: {e}"
            })

            if stop_on_error:
                break

    report = ValidationReport(
        total_documents=total,
        valid_documents=valid,
        invalid_documents=total - valid,
        errors=errors
    )

    logger.info(
        f"[검증 완료] 성공률: {report.success_rate():.1%} ({valid}/{total})"
    )

    return report


def validate_all_weeks(
    year: int,
    start_week: int,
    end_week: int
) -> Dict[str, ValidationReport]:
    """
    여러 주차에 걸친 문서 검증

    Args:
        year: 연도 (예: 2025)
        start_week: 시작 주차 번호
        end_week: 종료 주차 번호

    Returns:
        Dict[str, ValidationReport]: 주차 문자열 → ValidationReport 매핑

    동작:
        1. start_week부터 end_week까지 반복
        2. 각 주차 디렉토리 검증
        3. 주차별 ValidationReport 수집

    예시:
        >>> reports = validate_all_weeks(2025, 45, 49)
        >>> for week, report in reports.items():
        ...     print(f"{week}: {report.success_rate():.1%}")
        2025-W45: 98.5%
        2025-W46: 100.0%
        2025-W47: 99.2%
        2025-W48: 100.0%
        2025-W49: 98.0%
    """
    # Inline path calculation (no config.py)
    project_root = Path(__file__).parent.parent.parent
    documents_dir = project_root / "01_data" / "documents"

    reports = {}

    logger.info(f"[전체 검증 시작] {year}년 W{start_week}~W{end_week}")

    for week in range(start_week, end_week + 1):
        week_str = f"{year}-W{week:02d}"
        week_dir = documents_dir / str(year) / week_str

        if not week_dir.exists():
            logger.warning(f"[디렉토리 없음] {week_dir}")
            continue

        report = validate_directory(str(week_dir))
        reports[week_str] = report

    logger.info(f"[전체 검증 완료] {len(reports)}개 주차 검증됨")

    return reports
