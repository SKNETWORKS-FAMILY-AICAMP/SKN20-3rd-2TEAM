import os
import json
import re
from typing import List
from dotenv import load_dotenv
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# NLTK 불용어 다운로드
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    print("[PREPARE] NLTK 데이터 다운로드 중...")
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

load_dotenv()
KEYWORD_METHOD = os.getenv("KEYWORD_EXTRACTION_METHOD", "keybert").lower()

# 불용어 로드
nltk_stopwords = set(stopwords.words('english'))
# 도메인 특화 불용어 추가 (선택적)
custom_stopwords = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "been", "be",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "we", "our", "their", "they", "it", "its", "which", "who", "when",
    "where", "why", "how", "what", "if", "than", "such", "into", "through",
    # 논문에서 너무 많이 나오는 단어들
    "paper", "propose", "present", "show", "demonstrate", "using", "used",
    "approach", "method", "model", "based", "results", "work",
    "task", "tasks", "result", "results", "data"
}

all_stopwords = nltk_stopwords.union(custom_stopwords)


# KeyBERT를 사용하여 keyword 추출
def extract_keywords_keybert(text: str, top_n: int = 3) -> List[str]:
    """
    KeyBERT를 사용한 키워드 추출

    Args:
        text: 논문 Abstract
        top_n: 추출할 키워드 개수

    Returns:
        키워드 리스트
    """
    if not text or len(text.split()) < 10:
        return [f"keyword{i+1}" for i in range(top_n)]

    try:
        # KeyBERT 모델 초기화
        kw_model = KeyBERT()

        # 키워드 추출 (불용어 제거)
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),   # 1~2단어 키워드
            stop_words=list(all_stopwords),
            top_n=top_n,
            use_maxsum=True     # 다양성 확보
        )

        # (keyword, score) 튜플에서 keyword만 추출
        result = [kw[0] for kw in keywords]

        # 부족한 키워드는 기본값으로 채우기
        while len(result) < top_n:
            result.append(f"keyword{len(result)+1}")

        return result[:top_n]

    except Exception as e:
        print(f"[WARNING]  키워드 추출 실패: {e}")
        return [f"keyword{i+1}" for i in range(top_n)]


# TF-IDF를 사용하여 keyword 추출
def extract_keywords_tfidf(text: str, top_n: int = 3) -> List[str]:
    """
    TF-IDF 기반 키워드 추출 (with lemmatization)

    Args:
        text: 논문 Abstract
        top_n: 추출할 키워드 개수

    Returns:
        키워드 리스트
    """
    # 1. Validate input
    if not text or len(text.split()) < 10:
        return [f"keyword{i+1}" for i in range(top_n)]

    try:
        # 2. Lemmatization preprocessing
        lemmatizer = WordNetLemmatizer()
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        preprocessed_text = ' '.join(lemmatized_tokens)

        # 3. TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),  # Match KeyBERT
            stop_words=list(all_stopwords),
            token_pattern=r'\b[a-z]{3,}\b',
            max_df=1,
            min_df=0.85
        )

        tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # 4. Extract top N*2 candidates
        top_indices = scores.argsort()[-top_n*2:][::-1]
        candidates = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]

        if not candidates:
            return [f"keyword{i+1}" for i in range(top_n)]

        # 5. Remove substring duplicates (e.g., "transformer" vs "transformer architecture")
        unique_keywords = []
        for keyword, score in candidates:
            if not any(keyword in existing for existing in unique_keywords):
                unique_keywords = [kw for kw in unique_keywords if kw not in keyword]
                unique_keywords.append(keyword)
            if len(unique_keywords) >= top_n:
                break

        # 6. Fill with defaults if needed
        while len(unique_keywords) < top_n:
            unique_keywords.append(f"keyword{len(unique_keywords)+1}")

        return unique_keywords[:top_n]

    except Exception as e:
        print(f"[WARNING] TF-IDF 키워드 추출 실패: {e}")
        return [f"keyword{i+1}" for i in range(top_n)]


# Unified keyword extraction function
def extract_keywords(text: str, title: str = "", top_n: int = 3, method: str = "keybert") -> List[str]:
    """
    키워드 추출 (KeyBERT, TF-IDF)

    Args:
        text: 논문 Abstract
        top_n: 추출할 키워드 개수
        method: "keybert", "tfidf"

    Returns:
        키워드 리스트

    Raises:
        ValueError: method가 유효하지 않은 경우
    """
    if method not in ["keybert", "tfidf"]:
        raise ValueError(f"Invalid method: {method}. Must be 'keybert', 'tfidf'")
    
    if method == "keybert":
        return extract_keywords_keybert(text, top_n)
    else:
        return extract_keywords_tfidf(text, top_n)


def generate_tags_for_documents(data_dir: str = "01_data", method: str = "keybert", top_n: int = 3):
    """
    저장된 JSON 문서들의 context를 읽어서 tags를 생성하고 업데이트

    Args:
        data_dir: 문서가 저장된 루트 디렉토리 (기본값: "01_data")
        method: "keybert" 또는 "tfidf"
        top_n: 추출할 키워드 개수
    """
    method_suffix = "K" if method == "keybert" else "T"
    documents_dir = os.path.join(data_dir, f"documents_{method_suffix}")

    if not os.path.exists(documents_dir):
        print(f"[ERROR] 디렉토리를 찾을 수 없습니다: {documents_dir}")
        return

    # 모든 JSON 파일 찾기
    json_files = list(Path(documents_dir).rglob("*.json"))

    if not json_files:
        print(f"[WARNING] {documents_dir}에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"[START] 태그 생성 시작 (Method: {method.upper()})")
    print(f"[INFO]  총 {len(json_files)}개 문서 발견")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for json_path in tqdm(json_files, desc="[PROCESSING] 태그 생성"):
        try:
            # JSON 파일 읽기
            with open(json_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            # context가 없으면 건너뛰기
            context = document.get("context", "")
            if not context:
                skip_count += 1
                continue

            # 이미 tags가 있고 비어있지 않으면 건너뛰기 (선택적)
            existing_tags = document.get("metadata", {}).get("tags", [])
            if existing_tags and len(existing_tags) > 0:
                skip_count += 1
                continue

            # 논문 제목 추출 (LLM 메서드에서 사용)
            title = document.get("metadata", {}).get("title", "")

            # 키워드 추출
            keywords = extract_keywords(context, title=title, top_n=top_n, method=method)

            # metadata에 tags 추가
            if "metadata" not in document:
                document["metadata"] = {}
            document["metadata"]["tags"] = keywords

            # JSON 파일 업데이트
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] {json_path} 처리 실패: {e}")
            fail_count += 1

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"[END] 태그 생성 완료")
    print(f"   성공: {success_count}개")
    print(f"   실패: {fail_count}개")
    print(f"   건너뜀: {skip_count}개")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=== 논문 태그 생성기 ===\n")

    # 태그 생성 실행
    generate_tags_for_documents(
        data_dir="01_data",
        method=KEYWORD_METHOD,
        top_n=3
    )

    print("\n[COMPLETE] 태그 생성 완료!")
