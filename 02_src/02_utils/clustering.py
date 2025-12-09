"""
Auto Clustering 모듈 (Utils 버전)

이 모듈은 문서를 K-Means 알고리즘으로 클러스터링하여 검색 정확도를 향상시킵니다.
vectordb.py와 통합하여 사용할 수 있는 간소화된 API를 제공합니다.
"""

import os
import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv

import numpy as np
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk

# NLTK 데이터 다운로드
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    print("[PREPARE] NLTK 데이터 다운로드 중...")
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

# 환경 변수 로드
load_dotenv()
EMBEDDING_MODEL = "OpenAI"  # 기본 임베딩 모델

# 전역 경로 설정 (02_src/02_utils에서 프로젝트 루트로)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "01_data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CLUSTERS_DIR = DATA_DIR / "clusters"

# 불용어 설정
nltk_stopwords = set(stopwords.words('english'))
custom_stopwords = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "been", "be",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "we", "our", "their", "they", "it", "its", "which", "who", "when",
    "where", "why", "how", "what", "if", "than", "such", "into", "through",
    "paper", "propose", "present", "show", "demonstrate", "using", "used",
    "approach", "method", "model", "based", "results", "work",
    "task", "tasks", "result", "results", "data"
}
all_stopwords = nltk_stopwords.union(custom_stopwords)


def load_documents_for_clustering(use_weeks: int = 10) -> Tuple[List[Document], List[str]]:
    """
    클러스터링을 위한 문서 로딩 (최근 N주차, chunk 없이 전체 abstract)

    Args:
        use_weeks: 사용할 최근 주차 수 (기본값: 10)

    Returns:
        (documents, doc_ids): Document 리스트와 doc_id 리스트

    Raises:
        FileNotFoundError: 문서 디렉토리가 없을 때
    """
    if not DOCUMENTS_DIR.exists():
        print(f"[NOTFOUND] 문서 폴더를 찾을 수 없습니다: {DOCUMENTS_DIR}")
        raise FileNotFoundError(f"문서 폴더 없음: {DOCUMENTS_DIR}")

    print(f"\n[CLUSTERING] 문서 로딩 중... (최근 {use_weeks}주차)")

    documents = []
    doc_ids = []
    week_count = 0

    # 연도 폴더를 최신순으로 정렬
    year_dirs = sorted(DOCUMENTS_DIR.iterdir(), reverse=True)

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

                    # 파일명에서 doc_id 추출
                    doc_id = json_file.stem

                    # Document 생성 (전체 context 사용, chunking 없음)
                    metadata = data.get("metadata", {}).copy()
                    metadata["doc_id"] = doc_id

                    # upvote 정규화
                    if "upvote" in metadata:
                        upvote = metadata["upvote"]
                        if isinstance(upvote, str):
                            metadata["upvote"] = 0 if upvote.strip() == "-" else int(upvote)

                    doc = Document(
                        page_content=data.get("context", ""),
                        metadata=metadata
                    )

                    documents.append(doc)
                    doc_ids.append(doc_id)

                except Exception as e:
                    print(f"   [FAILED] 파일 로딩 실패 ({json_file.name}): {e}")
                    continue

            # 주차 카운트 증가
            week_count += 1
            if week_count >= use_weeks:
                print(f"[SUCCESS] {len(documents)}개 문서 로딩 완료 (최근 {use_weeks}주차)")
                return documents, doc_ids

    print(f"[SUCCESS] {len(documents)}개 문서 로딩 완료 (총 {week_count}주차)")
    return documents, doc_ids


def generate_embeddings_with_cache(
    documents: List[Document],
    doc_ids: List[str],
    model_name: str = "OpenAI",
    cache_path: Path = None
) -> np.ndarray:
    """
    문서 임베딩 생성 (캐시 활용으로 재실행 시 시간 절약)

    Args:
        documents: 문서 리스트
        doc_ids: 문서 ID 리스트
        model_name: 임베딩 모델 이름 (기본값: "OpenAI")
        cache_path: 캐시 파일 경로 (None이면 기본 경로 사용)

    Returns:
        embeddings: numpy array (n_docs, embedding_dim)
    """
    if cache_path is None:
        cache_path = CLUSTERS_DIR / "cluster_embeddings.pkl"

    # 캐시 파일 확인
    cached_embeddings = {}
    if cache_path.exists():
        print(f"[CACHE] 기존 임베딩 캐시 발견: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached_embeddings = pickle.load(f)
            print(f"[CACHE] {len(cached_embeddings)}개 임베딩 로딩 완료")
        except Exception as e:
            print(f"[WARNING] 캐시 로딩 실패: {e}")
            cached_embeddings = {}

    # 임베딩 모델 초기화
    print(f"\n[EMBEDDING] 임베딩 생성 중... (Model: {model_name})")
    if model_name == "OpenAI":
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    # 신규 문서 확인
    new_docs = []
    new_doc_ids = []
    embeddings_list = []

    for doc, doc_id in zip(documents, doc_ids):
        if doc_id in cached_embeddings:
            # 캐시된 임베딩 사용
            embeddings_list.append(cached_embeddings[doc_id])
        else:
            # 신규 문서 추가
            new_docs.append(doc.page_content)
            new_doc_ids.append(doc_id)
            embeddings_list.append(None)

    # 신규 문서 임베딩 생성
    if new_docs:
        print(f"[EMBEDDING] 신규 문서 {len(new_docs)}개 임베딩 생성 중...")
        try:
            new_embeddings = embedding_model.embed_documents(new_docs)

            # 캐시 업데이트
            new_idx = 0
            for i, emb in enumerate(embeddings_list):
                if emb is None:
                    embeddings_list[i] = new_embeddings[new_idx]
                    cached_embeddings[new_doc_ids[new_idx]] = new_embeddings[new_idx]
                    new_idx += 1

            # 캐시 저장
            CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(cached_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[CACHE] 임베딩 캐시 저장 완료: {cache_path}")

        except Exception as e:
            print(f"[FAILED] 임베딩 생성 실패: {e}")
            raise
    else:
        print(f"[CACHE] 모든 문서가 캐시되어 있습니다 (신규 생성 없음)")

    embeddings_array = np.array(embeddings_list)
    print(f"[SUCCESS] 임베딩 생성 완료 (shape: {embeddings_array.shape})")

    return embeddings_array


def determine_optimal_clusters(
    embeddings: np.ndarray,
    min_k: int = 10,
    max_k: int = 30
) -> int:
    """
    엘보우 방법으로 최적 클러스터 개수 자동 결정

    Args:
        embeddings: 문서 임베딩 배열
        min_k: 최소 클러스터 개수 (기본값: 10)
        max_k: 최대 클러스터 개수 (기본값: 30)

    Returns:
        optimal_k: 최적 클러스터 개수
    """
    print(f"\n[OPTIMIZE] 최적 클러스터 개수 결정 중... (k={min_k}~{max_k})")

    inertias = []
    k_range = range(min_k, max_k + 1)

    for k in tqdm(k_range, desc="[OPTIMIZE] 클러스터 개수 최적화"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    # 엘보우 지점 찾기 (감소율이 급격히 줄어드는 지점)
    # 간단한 방법: 2차 미분이 최대인 지점
    if len(inertias) >= 3:
        diff1 = np.diff(inertias)
        diff2 = np.diff(diff1)
        # 감소율 변화가 가장 큰 지점 (절댓값 최대)
        elbow_idx = np.argmax(np.abs(diff2)) + 1
        optimal_k = min_k + elbow_idx
    else:
        # 중간값 선택
        optimal_k = (min_k + max_k) // 2

    # 범위 내로 제한
    optimal_k = max(min_k, min(optimal_k, max_k))

    print(f"[INFO] 최적 클러스터 개수: {optimal_k}")
    print(f"[INFO] Inertia 범위: {inertias[0]:.2f} → {inertias[-1]:.2f}")

    return optimal_k


def perform_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, KMeans]:
    """
    K-Means 클러스터링 수행

    Args:
        embeddings: 문서 임베딩 배열
        n_clusters: 클러스터 개수 (기본값: 20)
        random_state: 재현성을 위한 시드 (기본값: 42)

    Returns:
        (labels, centroids, model): 클러스터 레이블, 중심점, KMeans 모델
    """
    print(f"\n[CLUSTERING] K-Means 클러스터링 수행 중... (n_clusters={n_clusters})")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )

    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # 클러스터 품질 평가
    try:
        silhouette = silhouette_score(embeddings, labels)
        print(f"[INFO] Silhouette Score: {silhouette:.3f}")
    except Exception as e:
        print(f"[WARNING] Silhouette Score 계산 실패: {e}")

    # 클러스터 크기 분포
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[INFO] 클러스터 크기: 최소={counts.min()}, 평균={counts.mean():.1f}, 최대={counts.max()}")

    print(f"[SUCCESS] 클러스터링 완료 ({n_clusters}개 클러스터)")

    return labels, centroids, kmeans


def generate_cluster_metadata(
    documents: List[Document],
    doc_ids: List[str],
    labels: np.ndarray,
    centroids: np.ndarray,
    embeddings: np.ndarray
) -> Dict[int, Dict]:
    """
    각 클러스터의 통계 및 대표 키워드 생성

    Args:
        documents: 원본 문서 리스트
        doc_ids: 문서 ID 리스트
        labels: 클러스터 레이블 배열
        centroids: 클러스터 중심점
        embeddings: 문서 임베딩 배열

    Returns:
        cluster_metadata: 클러스터별 메타데이터
    """
    print(f"\n[METADATA] 클러스터 메타데이터 생성 중...")

    n_clusters = len(centroids)
    cluster_metadata = {}

    for cluster_id in tqdm(range(n_clusters), desc="[METADATA] 메타데이터 생성"):
        # 클러스터에 속한 문서 인덱스
        cluster_mask = (labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # 클러스터 내 문서 정보
        cluster_docs = [documents[i] for i in cluster_indices]
        cluster_doc_ids = [doc_ids[i] for i in cluster_indices]
        cluster_embeddings = embeddings[cluster_indices]

        # 중심점과의 거리 계산
        distances = np.linalg.norm(cluster_embeddings - centroids[cluster_id], axis=1)

        # 밀집도 및 반경 계산
        mean_distance = np.mean(distances)
        density = 1.0 / mean_distance if mean_distance > 0 else 0.0
        radius = np.max(distances)

        # 키워드 추출 (TF-IDF 기반)
        cluster_texts = [doc.page_content for doc in cluster_docs]
        keywords = extract_keywords_tfidf(cluster_texts, top_n=5)

        # 상위 논문 3개 (중심에 가까운 순)
        top_indices = np.argsort(distances)[:3]
        top_papers = []
        for idx in top_indices:
            doc_idx = cluster_indices[idx]
            top_papers.append({
                "doc_id": doc_ids[doc_idx],
                "title": documents[doc_idx].metadata.get("title", "Unknown"),
                "distance": float(distances[idx]),
                "upvote": documents[doc_idx].metadata.get("upvote", 0)
            })

        # 평균 upvote 계산
        upvotes = [doc.metadata.get("upvote", 0) for doc in cluster_docs]
        avg_upvote = np.mean(upvotes) if upvotes else 0.0

        # 연도별 분포
        years = [doc.metadata.get("publication_year", 0) for doc in cluster_docs]
        year_counts = {}
        for year in years:
            if year > 0:
                year_counts[year] = year_counts.get(year, 0) + 1

        # 메타데이터 저장
        cluster_metadata[cluster_id] = {
            "size": len(cluster_indices),
            "doc_ids": cluster_doc_ids,
            "density": float(density),
            "radius": float(radius),
            "keywords": keywords,
            "top_papers": top_papers,
            "avg_upvote": float(avg_upvote),
            "publication_years": year_counts
        }

    print(f"[SUCCESS] 메타데이터 생성 완료 ({len(cluster_metadata)}개 클러스터)")

    return cluster_metadata


def extract_keywords_tfidf(texts: List[str], top_n: int = 5) -> List[str]:
    """
    TF-IDF 기반 키워드 추출 (클러스터용)

    Args:
        texts: 텍스트 리스트 (클러스터 내 모든 문서)
        top_n: 추출할 키워드 개수

    Returns:
        keywords: 상위 키워드 리스트
    """
    if not texts:
        return [f"keyword{i+1}" for i in range(top_n)]

    try:
        # 전처리: lemmatization
        lemmatizer = WordNetLemmatizer()
        preprocessed_texts = []

        for text in texts:
            tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
            lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
            preprocessed_texts.append(' '.join(lemmatized))

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words=list(all_stopwords),
            token_pattern=r'\b[a-z]{3,}\b',
            max_df=0.8,
            min_df=2
        )

        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
        feature_names = vectorizer.get_feature_names_out()

        # 전체 문서에 대한 평균 TF-IDF 계산
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = avg_tfidf.argsort()[-top_n*2:][::-1]

        # 중복 제거
        candidates = [feature_names[i] for i in top_indices if avg_tfidf[i] > 0]
        unique_keywords = []
        for keyword in candidates:
            if not any(keyword in existing for existing in unique_keywords):
                unique_keywords = [kw for kw in unique_keywords if kw not in keyword]
                unique_keywords.append(keyword)
            if len(unique_keywords) >= top_n:
                break

        # 부족하면 기본값 채우기
        while len(unique_keywords) < top_n:
            unique_keywords.append(f"keyword{len(unique_keywords)+1}")

        return unique_keywords[:top_n]

    except Exception as e:
        print(f"[WARNING] 키워드 추출 실패: {e}")
        return [f"keyword{i+1}" for i in range(top_n)]


def save_cluster_results(
    doc_ids: List[str],
    labels: np.ndarray,
    cluster_metadata: Dict,
    output_dir: Path = None
) -> Tuple[str, str]:
    """
    클러스터링 결과를 JSON으로 저장

    Args:
        doc_ids: 문서 ID 리스트
        labels: 클러스터 레이블 배열
        cluster_metadata: 클러스터 메타데이터
        output_dir: 출력 디렉토리 (None이면 기본 경로)

    Returns:
        (assignments_path, metadata_path): 저장된 파일 경로
    """
    if output_dir is None:
        output_dir = CLUSTERS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[SAVE] 결과 저장 중... ({output_dir})")

    # 1. cluster_assignments.json
    assignments = {doc_id: int(label) for doc_id, label in zip(doc_ids, labels)}

    # 주차 정보 추출 (doc_id에서)
    weeks_used = sorted(set([doc_id[3:7] for doc_id in doc_ids if len(doc_id) >= 7]))
    weeks_formatted = [f"2025-W{week[2:]}" for week in weeks_used]

    assignments_data = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(),
            "n_documents": len(doc_ids),
            "n_clusters": len(cluster_metadata),
            "embedding_model": EMBEDDING_MODEL,
            "weeks_used": weeks_formatted
        },
        "assignments": assignments
    }

    assignments_path = output_dir / "cluster_assignments.json"
    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump(assignments_data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] cluster_assignments.json 저장 완료")

    # 2. cluster_metadata.json
    metadata_data = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(),
            "n_clusters": len(cluster_metadata),
            "total_documents": len(doc_ids)
        },
        "clusters": {str(k): v for k, v in cluster_metadata.items()}
    }

    metadata_path = output_dir / "cluster_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] cluster_metadata.json 저장 완료")

    return str(assignments_path), str(metadata_path)


def cluster_documents(use_weeks: int = 10) -> Dict[str, int]:
    """
    문서 클러스터링 실행 및 doc_id → cluster_id 매핑 반환 (공개 API)

    Args:
        use_weeks: 사용할 최근 주차 수 (기본값: 10)

    Returns:
        {doc_id: cluster_id} 딕셔너리

    사용 예시:
        >>> from clustering import cluster_documents
        >>> doc_cluster_map = cluster_documents(use_weeks=10)
        >>> print(doc_cluster_map["doc2549001"])  # 해당 문서의 cluster_id
    """
    print("=" * 60)
    print("[CLUSTERING API] 문서 클러스터링 시작")
    print("=" * 60)

    try:
        # 1. 문서 로딩
        documents, doc_ids = load_documents_for_clustering(use_weeks=use_weeks)

        if not documents:
            print("[FAILED] 로딩된 문서가 없습니다!")
            return {}

        # 2. 임베딩 생성
        embeddings = generate_embeddings_with_cache(
            documents=documents,
            doc_ids=doc_ids,
            model_name=EMBEDDING_MODEL
        )

        # 3. 최적 클러스터 개수 결정
        optimal_k = determine_optimal_clusters(
            embeddings=embeddings,
            min_k=10,
            max_k=30
        )

        # 4. K-Means 클러스터링
        labels, centroids, kmeans_model = perform_clustering(
            embeddings=embeddings,
            n_clusters=optimal_k
        )

        # 5. 메타데이터 생성
        cluster_metadata = generate_cluster_metadata(
            documents=documents,
            doc_ids=doc_ids,
            labels=labels,
            centroids=centroids,
            embeddings=embeddings
        )

        # 6. 결과 저장
        save_cluster_results(
            doc_ids=doc_ids,
            labels=labels,
            cluster_metadata=cluster_metadata
        )

        # 7. doc_id → cluster_id 매핑 반환
        doc_cluster_map = {doc_id: int(label) for doc_id, label in zip(doc_ids, labels)}

        print("\n" + "=" * 60)
        print(f"[COMPLETE] 클러스터링 완료! ({len(doc_cluster_map)}개 문서)")
        print("=" * 60)

        return doc_cluster_map

    except Exception as e:
        print(f"\n[FAILED] 클러스터링 실패: {e}")
        raise


if __name__ == "__main__":
    # 독립 실행 시 전체 클러스터링 수행
    result = cluster_documents(use_weeks=10)
    print(f"\n샘플 매핑: {dict(list(result.items())[:3])}")
