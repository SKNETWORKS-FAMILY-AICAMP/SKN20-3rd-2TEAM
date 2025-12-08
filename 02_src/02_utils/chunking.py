import os
import json
import pickle
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 전역 경로 설정
# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "01_data"
CHUNKS_DIR = DATA_DIR / "chunks"

# 문서 분리 기준
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))
KEYWORD_METHOD = os.getenv("KEYWORD_EXTRACTION_METHOD").lower()


def load_json_files(use_weeks: int = 6, method: str = "keybert") -> List[Document]:
    """
    최근 use_weeks주차의 JSON 문서 파일을 로딩

    Args:
        use_weeks: 로딩할 최근 주차 수 (기본값: 6)
        method: "keybert" 또는 "tfidf" (문서 디렉토리 선택)

    Returns:
        LangChain Document 리스트

    동작 방식:
        1. 01_data/documents_K/ 또는 documents_T/ 폴더를 스캔
        2. 최신 주차부터 use_weeks 만큼 JSON 파일 로딩
        3. JSON의 context를 page_content로, metadata는 그대로 사용

    Directory Selection:
        - "keybert" → 01_data/documents_K/
        - "tfidf"   → 01_data/documents_T/
        - Falls back to 01_data/documents/ if method-specific dir not found
    """
    method_suffix = "K" if method == "keybert" else "T"
    documents_dir_method = DATA_DIR / f"documents_{method_suffix}"
    documents_dir_legacy = DATA_DIR / "documents"

    # Try method-specific directory first, fall back to legacy
    if documents_dir_method.exists():
        DOCUMENTS_DIR_ACTIVE = documents_dir_method
        print(f"\n[LOADING] 최근 {use_weeks}주차 JSON 문서 로딩 중... (Method: {method.upper()})")
    elif documents_dir_legacy.exists():
        DOCUMENTS_DIR_ACTIVE = documents_dir_legacy
        print(f"\n[LOADING] 최근 {use_weeks}주차 JSON 문서 로딩 중... (Legacy directory)")
        print(f"[WARNING] Method-specific directory not found, using legacy")
    else:
        print(f"[NOTFOUND] 문서 폴더를 찾을 수 없습니다 {documents_dir_method}")
        return []

    documents = []
    week_count = 0

    # 연도 폴더를 최신순으로 정렬
    year_dirs = sorted(DOCUMENTS_DIR_ACTIVE.iterdir(), reverse=True)

    for year_dir in year_dirs:
        if not year_dir.is_dir():
            continue

        # 주차 폴더를 최신순으로 정렬
        week_dirs = sorted(year_dir.iterdir(), reverse=True)

        for week_dir in week_dirs:
            if not week_dir.is_dir():
                continue

            # 주차별 JSON 파일 로딩
            json_files = sorted(week_dir.glob("*.json"))

            for json_file in json_files:
                try:
                    # JSON 파일 읽기
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 파일명에서 doc_id 추출 (doc2549001.json -> doc2549001)
                    doc_id = json_file.stem

                    # Document 생성
                    metadata = data.get("metadata", {}).copy()
                    metadata["doc_id"] = doc_id

                    # upvote 정규화 (문자열 -> 숫자)
                    if "upvote" in metadata:
                        upvote = metadata["upvote"]
                        if isinstance(upvote, str):
                            metadata["upvote"] = (
                                0 if upvote.strip() == "-" else int(upvote)
                            )

                    doc = Document(
                        page_content=data.get("context", ""), metadata=metadata
                    )

                    documents.append(doc)

                except Exception as e:
                    print(f"   [FAILED] 파일 로딩 실패 ({json_file.name}): {e}")
                    continue

            # 주차 카운트 증가
            week_count += 1
            if week_count >= use_weeks:
                print(f"[SUCCESS] {len(documents)}개 문서 로딩 완료 (최근 {use_weeks}주차)")
                return documents

    print(f"[SUCCESS] {len(documents)}개 문서 로딩 완료 (총 {week_count}주차)")
    return documents


def chunk_documents(
    documents: List[Document], chunk_size: int = 100, chunk_overlap: int = 10
) -> List[Document]:
    """
    문서 리스트를 작은 청크로 분할

    Args:
        documents: 원본 문서 리스트
        chunk_size: 청크 하나의 크기 (글자 수)
        chunk_overlap: 청크 간 중복되는 부분 (글자 수)

    Returns:
        청크로 분할된 문서 리스트

    동작 방식:
        1. RecursiveCharacterTextSplitter로 문서를 나눔
        2. 각 청크에 chunk_index와 total_chunks 정보 추가
        3. 원본 문서의 메타데이터는 모두 유지
    """
    print(f"\n[CHUNKING START] 청킹 시작: {len(documents)}개 문서")
    print(f"   - CHUNK_SIZE: {chunk_size}")
    print(f"   - CHUNK_OVERLAP: {chunk_overlap}")

    # 텍스트 분할기 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,
    )

    # 모든 청크를 저장할 리스트
    all_chunks = []

    # 각 문서를 청크로 분할
    for doc_idx, doc in enumerate(documents):
        # 문서를 청크로 나누기
        chunks = text_splitter.split_documents([doc])

        # 각 청크에 인덱스 정보 추가
        for chunk_idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_idx
            chunk.metadata["total_chunks"] = len(chunks)
            all_chunks.append(chunk)

        # 진행 상황 표시 (100개마다)
        if (doc_idx + 1) % 100 == 0:
            print(f"   진행: {doc_idx + 1}/{len(documents)} 문서 처리됨")

    print(f"\n[CHUNKING SUCCESS] 청킹 완료: {len(all_chunks)}개 청크 생성")
    print(f"   평균 {len(all_chunks)/len(documents):.1f}개 청크/문서")

    return all_chunks


def save_chunks_to_pkl(
    chunks: List[Document], chunk_size: int, chunk_overlap: int, method: str = "keybert"
) -> str:
    """
    청크 리스트를 .pkl 파일로 저장

    Args:
        chunks: 저장할 청크 리스트
        chunk_size: 청크 크기 (파일명에 사용)
        chunk_overlap: 청크 오버랩 (파일명에 사용)
        method: "keybert" 또는 "tfidf"

    Returns:
        저장된 파일 경로

    Filename Format:
        chunks_{chunk_size}_{chunk_overlap}_{K|T}.pkl
    """
    # 저장 폴더 생성 (없으면)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # 파일명 생성 (method suffix 포함)
    method_suffix = "K" if method == "keybert" else "T"
    filename = f"chunks_{chunk_size}_{chunk_overlap}_{method_suffix}.pkl"
    output_path = CHUNKS_DIR / filename

    print(f"\n[SAVE START] 저장 시작: {output_path}")

    # pickle로 저장
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 파일 크기 계산
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"[SAVE SUCCESS] 저장 완료: {file_size:.2f} MB")

    return str(output_path)


def chunk_and_save(
    use_weeks: int = 6, chunk_size: int = 100, chunk_overlap: int = 10, method: str = "keybert"
) -> str:
    """
    문서 로딩 → 청킹 → 저장을 한번에 실행

    Args:
        use_weeks: 사용할 최근 주차 수 (기본값: 6)
        chunk_size: 청크 크기 (기본값: 100)
        chunk_overlap: 청크 오버랩 (기본값: 10)
        method: "keybert" 또는 "tfidf"

    Returns:
        저장된 파일 경로

    사용 예시:
        # 기본 설정으로 실행
        >>> chunk_and_save()

        # 커스텀 파라미터로 실행
        >>> chunk_and_save(use_weeks=4, chunk_size=200, chunk_overlap=20, method="tfidf")
    """
    print("=" * 60)
    print(f"[START] 문서 청킹 시작 (Method: {method.upper()})")
    print("=" * 60)
    print(
        f"설정: use_weeks={use_weeks}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, method={method}"
    )

    # 1. 문서 로딩 (method 파라미터 전달)
    documents = load_json_files(use_weeks=use_weeks, method=method)

    if not documents:
        print("[FAILED LOADING] 로딩된 문서가 없습니다!")
        return ""

    # 2. 청킹
    chunks = chunk_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # 3. 저장 (method 파라미터 전달)
    filepath = save_chunks_to_pkl(chunks, chunk_size, chunk_overlap, method=method)

    print("\n" + "=" * 60)
    print("[SUCCESS] 모든 작업 완료!")
    print(f"[PATH] 저장 위치: {filepath}")
    print("=" * 60)

    return filepath


def load_chunks_from_pkl(
    chunk_size: int = 100, chunk_overlap: int = 10, method: str = "keybert"
) -> List[Document]:
    """
    .pkl 파일에서 청크 리스트 로딩

    Args:
        chunk_size: 청크 크기 (파일명 지정용)
        chunk_overlap: 청크 오버랩 (파일명 지정용)
        method: "keybert" 또는 "tfidf"

    Returns:
        로딩된 청크 리스트

    Fallback Strategy:
        1. Try method-specific file (chunks_{size}_{overlap}_{K|T}.pkl)
        2. Fall back to legacy file (chunks_{size}_{overlap}.pkl)

    사용 예시:
        >>> chunks = load_chunks_from_pkl(chunk_size=100, chunk_overlap=10, method="keybert")
        >>> print(f"로딩된 청크 개수: {len(chunks)}")
    """
    method_suffix = "K" if method == "keybert" else "T"

    # Try method-specific file first
    filename_method = f"chunks_{chunk_size}_{chunk_overlap}_{method_suffix}.pkl"
    pkl_path_method = CHUNKS_DIR / filename_method

    # Fallback to legacy filename
    filename_legacy = f"chunks_{chunk_size}_{chunk_overlap}.pkl"
    pkl_path_legacy = CHUNKS_DIR / filename_legacy

    if pkl_path_method.exists():
        pkl_path = pkl_path_method
        print(f"\n[LOADING] 청크 로딩 중: {pkl_path} (Method: {method.upper()})")
    elif pkl_path_legacy.exists():
        pkl_path = pkl_path_legacy
        print(f"\n[LOADING] 청크 로딩 중: {pkl_path} (Legacy file)")
        print(f"[WARNING] Method-specific file not found, using legacy")
    else:
        print(f"[NOT FOUND] 파일을 찾을 수 없습니다: {pkl_path_method}")
        raise FileNotFoundError(f"파일 없음: {pkl_path_method}")

    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"[SUCCESS] {len(chunks)}개 청크 로딩 완료")

    return chunks


if __name__ == "__main__":
    # 기본 설정으로 청킹 실행
    filepath = chunk_and_save(use_weeks=6, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, method=KEYWORD_METHOD)

    # 생성된 청크 확인
    print("\n" + "=" * 60)
    print("[SAMPLE] 청크 샘플 미리보기 (첫 3개)")
    print("=" * 60)

    chunks = load_chunks_from_pkl(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, method=KEYWORD_METHOD)

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- 청크 #{i+1} ---")
        print(f"문서 ID: {chunk.metadata.get('doc_id')}")
        print(
            f"청크 인덱스: {chunk.metadata.get('chunk_index')}/{chunk.metadata.get('total_chunks')}"
        )
        print(f"길이: {len(chunk.page_content)}자")
        print(f"내용 미리보기: {chunk.page_content[:100]}...")
        