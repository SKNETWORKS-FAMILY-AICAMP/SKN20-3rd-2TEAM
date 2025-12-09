"""
실제 클러스터 데이터 기반 개선된 TEST_QUERIES

클러스터 메타데이터 분석 결과를 바탕으로 생성된 테스트 쿼리입니다.
evaluate_embeddings.py의 TEST_QUERIES를 대체할 수 있습니다.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class TestQuery:
    """평가용 테스트 쿼리"""
    query: str
    relevant_keywords: List[str]
    relevant_doc_ids: List[str] = None

    def __repr__(self):
        return f"TestQuery(query='{self.query[:50]}...', keywords={self.relevant_keywords})"


# ==================== 개선된 TEST_QUERIES ====================

IMPROVED_TEST_QUERIES = [
    # Cluster 2, 15: video generation (58+53=111 papers, avg_upvote: 33.1)
    TestQuery(
        query="video generation and motion synthesis models",
        relevant_keywords=["video", "generation", "motion"],
    ),

    # Cluster 1, 16: multimodal understanding (83+53=136 papers, avg_upvote: 31.9)
    TestQuery(
        query="multimodal visual understanding and image token processing",
        relevant_keywords=["multimodal", "visual", "image"],
    ),

    # Cluster 8: code generation (65 papers, avg_upvote: 26.7)
    TestQuery(
        query="code generation and large language models for programming",
        relevant_keywords=["code", "language", "llm"],
    ),

    # Cluster 5, 9: diffusion models (53+60=113 papers, avg_upvote: 37.8)
    TestQuery(
        query="diffusion models for high quality image generation",
        relevant_keywords=["diffusion", "generation", "quality"],
    ),

    # Cluster 14: reinforcement learning with rewards (28 papers, avg_upvote: 36.4)
    TestQuery(
        query="group relative policy optimization and reward models",
        relevant_keywords=["grpo", "reward", "agent"],
    ),

    # Cluster 3, 4, 6: reasoning and LLMs (85+26+76=187 papers, avg_upvote: 33.6)
    TestQuery(
        query="reasoning capabilities and training methods for large language models",
        relevant_keywords=["reasoning", "llm", "training"],
    ),

    # Cluster 12: vision-language-action robotics (44 papers, avg_upvote: 23.6)
    TestQuery(
        query="vision language action models for robotic manipulation",
        relevant_keywords=["vla", "robot", "reasoning"],
    ),

    # Cluster 7: image editing with instructions (37 papers, avg_upvote: 23.6)
    TestQuery(
        query="instruction-guided image editing and controllable generation",
        relevant_keywords=["instruction", "image editing", "control"],
    ),

    # Cluster 11: attention and context (81 papers, avg_upvote: 42.9 - HIGHEST!)
    TestQuery(
        query="attention mechanisms and long context processing in transformers",
        relevant_keywords=["attention", "token", "context"],
    ),

    # Cluster 10: spatial reasoning (31 papers, avg_upvote: 21.7)
    TestQuery(
        query="spatial reasoning and scene understanding in multimodal models",
        relevant_keywords=["spatial", "reasoning", "scene"],
    ),

    # Cluster 13: scientific benchmarks (60 papers, avg_upvote: 28.7)
    TestQuery(
        query="scientific research benchmarks and evaluation frameworks",
        relevant_keywords=["scientific", "benchmark", "system"],
    ),

    # Cluster 17: LLM training and performance (81 papers, avg_upvote: 44.3 - HIGHEST!)
    TestQuery(
        query="training techniques and performance optimization for large language models",
        relevant_keywords=["training", "llm", "performance"],
    ),

    # Cluster 0: feature extraction (57 papers, avg_upvote: 25.3)
    TestQuery(
        query="high quality feature extraction from images",
        relevant_keywords=["image", "feature", "high"],
    ),

    # Cluster 5: long context (53 papers, avg_upvote: 38.7)
    TestQuery(
        query="long context processing and quality improvements in generation models",
        relevant_keywords=["long", "quality", "training"],
    ),
]


# ==================== 클러스터 커버리지 분석 ====================

def analyze_cluster_coverage():
    """
    개선된 TEST_QUERIES가 몇 개의 클러스터를 커버하는지 분석

    클러스터 매핑:
    - Cluster 0: Feature extraction (IMPROVED_TEST_QUERIES[12])
    - Cluster 1: Multimodal visual (IMPROVED_TEST_QUERIES[1])
    - Cluster 2: Video generation (IMPROVED_TEST_QUERIES[0])
    - Cluster 3, 4, 6: LLM reasoning (IMPROVED_TEST_QUERIES[5])
    - Cluster 5: Long context (IMPROVED_TEST_QUERIES[13])
    - Cluster 7: Image editing (IMPROVED_TEST_QUERIES[7])
    - Cluster 8: Code generation (IMPROVED_TEST_QUERIES[2])
    - Cluster 9: Image generation (IMPROVED_TEST_QUERIES[3])
    - Cluster 10: Spatial reasoning (IMPROVED_TEST_QUERIES[9])
    - Cluster 11: Attention mechanisms (IMPROVED_TEST_QUERIES[8])
    - Cluster 12: VLA robotics (IMPROVED_TEST_QUERIES[6])
    - Cluster 13: Scientific benchmarks (IMPROVED_TEST_QUERIES[10])
    - Cluster 14: Reinforcement learning (IMPROVED_TEST_QUERIES[4])
    - Cluster 15: Video/multimodal (IMPROVED_TEST_QUERIES[0])
    - Cluster 16: Multimodal understanding (IMPROVED_TEST_QUERIES[1])
    - Cluster 17: LLM training (IMPROVED_TEST_QUERIES[11])

    커버리지: 18개 클러스터 중 18개 커버 (100%)
    """
    cluster_mapping = {
        0: [12],
        1: [1],
        2: [0],
        3: [5],
        4: [5],
        5: [3, 13],
        6: [5],
        7: [7],
        8: [2],
        9: [3],
        10: [9],
        11: [8],
        12: [6],
        13: [10],
        14: [4],
        15: [0],
        16: [1],
        17: [11],
    }

    print("=== 클러스터 커버리지 ===")
    print(f"총 클러스터 수: 18")
    print(f"총 TEST_QUERIES 수: {len(IMPROVED_TEST_QUERIES)}")
    print(f"커버된 클러스터: {len(cluster_mapping)}/18 (100%)")
    print("\n주요 클러스터 (avg_upvote >= 35):")
    print("  - Cluster 17 (LLM training, 44.3): IMPROVED_TEST_QUERIES[11]")
    print("  - Cluster 11 (Attention, 42.9): IMPROVED_TEST_QUERIES[8]")
    print("  - Cluster 5 (Long context, 38.7): IMPROVED_TEST_QUERIES[13]")
    print("  - Cluster 15 (Video/multimodal, 39.6): IMPROVED_TEST_QUERIES[0]")
    print("  - Cluster 3 (LLM reasoning, 38.4): IMPROVED_TEST_QUERIES[5]")
    print("  - Cluster 9 (Image generation, 36.9): IMPROVED_TEST_QUERIES[3]")
    print("  - Cluster 14 (GRPO, 36.4): IMPROVED_TEST_QUERIES[4]")
    print("  - Cluster 6 (LLM reasoning, 36.3): IMPROVED_TEST_QUERIES[5]")
    print("  - Cluster 16 (Multimodal, 35.5): IMPROVED_TEST_QUERIES[1]")


# ==================== 사용 방법 ====================

if __name__ == "__main__":
    print("=" * 80)
    print("실제 클러스터 기반 개선된 TEST_QUERIES")
    print("=" * 80)
    print()

    for i, query in enumerate(IMPROVED_TEST_QUERIES, 1):
        print(f"{i}. {query}")
        print()

    print()
    analyze_cluster_coverage()

    print("\n" + "=" * 80)
    print("사용 방법:")
    print("=" * 80)
    print("""
evaluate_embeddings.py에서 TEST_QUERIES를 교체하려면:

1. 84~125번째 줄의 TEST_QUERIES를 삭제
2. 다음 import 추가:
   from improved_test_queries import IMPROVED_TEST_QUERIES as TEST_QUERIES

또는 직접 TEST_QUERIES 변수를 IMPROVED_TEST_QUERIES로 교체하세요.
""")
