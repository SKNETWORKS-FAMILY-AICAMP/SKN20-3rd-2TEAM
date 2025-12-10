# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path
from collections import Counter

# Windows console UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent
CLUSTER_METADATA_PATH = PROJECT_ROOT / "01_data" / "clusters" / "cluster_metadata.json"


def load_cluster_metadata():
    """Load cluster metadata"""
    with open(CLUSTER_METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_cluster_info():
    """Print cluster information"""
    # Load metadata
    metadata = load_cluster_metadata()

    print("=" * 80)
    print("클러스터 메타데이터 정보")
    print("=" * 80)
    print(f"생성 일시: {metadata['_metadata']['generated_at']}")
    print(f"총 클러스터 수: {metadata['_metadata']['n_clusters']}")
    print(f"총 문서 수: {metadata['_metadata']['total_documents']}")
    print("=" * 80)
    print()

    # Print each cluster ID and keywords
    print("=" * 80)
    print("각 클러스터의 키워드")
    print("=" * 80)

    clusters = metadata['clusters']
    all_keywords = []

    for cluster_id in sorted(clusters.keys(), key=int):
        cluster = clusters[cluster_id]
        keywords = cluster['keywords']
        all_keywords.extend(keywords)

        print(f"\n클러스터 ID: {cluster_id} / 문서 수: {cluster['size']}")
        print(f"  키워드: [ {', '.join(keywords)} ]")

    print("\n" + "=" * 80)

    # Count and sort all keywords
    keyword_counter = Counter(all_keywords)
    sorted_keywords = keyword_counter.most_common()

    print("\n" + "=" * 80)
    print("전체 키워드 통계 (빈도순)")
    print("=" * 80)
    print(f"총 키워드 수: {len(all_keywords)}")
    print(f"고유 키워드 수: {len(keyword_counter)}")
    print("=" * 80)
    print()

    # Print 5 keywords per line
    for i, (keyword, count) in enumerate(sorted_keywords):
        print(f"{keyword:13s}: {count}", end="\t")
        if i % 5 == 4:
            print()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_cluster_info()
