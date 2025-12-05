from pathlib import Path

# 프로젝트 루트 자동 탐지
def get_project_root():
    """Git 루트 찾기"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent

# 경로 설정
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "01_data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHUNKS_DIR = DATA_DIR / "chunks"
DOCUMENTS_DIR = DATA_DIR / "documents"

# 디렉토리 생성
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 파일 경로
CHUNKS_FILE = CHUNKS_DIR / "chunks_all.pkl"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.pkl"