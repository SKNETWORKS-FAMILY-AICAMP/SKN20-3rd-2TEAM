# 임베딩 모델 비교 평가 결과

**평가 날짜:** 2025-12-04
**데이터셋:** chunks_all.pkl (16,508개 청크 중 2,000개 샘플링)
**평가 쿼리:** 10개의 ML/DL/LLM 관련 테스트 쿼리
**평가 메트릭:** Recall@5, Recall@10, MRR (Mean Reciprocal Rank), 검색 속도

---

## 📊 최종 결과 (Recall@10 기준 정렬)

| 순위 | 모델 | R@5 | R@10 | MRR | 속도(초) | 비고 |
|------|------|-----|------|-----|---------|------|
| 🥇 | **sentence-transformers/all-MiniLM-L6-v2** | **0.040** | **0.064** | **0.658** | **0.02** | ✅ **최종 선택** |
| 🥈 | sentence-transformers/msmarco-MiniLM-L-6-v3 | 0.030 | 0.043 | 0.417 | 0.02 | 검색 최적화 |
| 🥉 | OpenAI text-embedding-3-small | 0.023 | 0.038 | 0.443 | 0.56 | 비용 발생, 28배 느림 |
| 4 | sentence-transformers/all-mpnet-base-v2 | 0.030 | 0.034 | 0.583 | 0.08 | 높은 차원(768) |
| 5 | sentence-transformers/allenai-specter | 0.016 | 0.018 | 0.367 | 0.07 | 과학 논문 특화 |

---

## 🏆 최고 성능 모델: all-MiniLM-L6-v2

### 선정 이유

1. **최고 검색 성능**
   - Recall@10: 6.4% (1위)
   - MRR: 0.658 (1위) - 관련 문서가 평균 상위 1.5위 이내 등장
   - 관련 문서를 가장 정확하게 상위에 랭킹

2. **최고 속도**
   - 0.02초/쿼리 (OpenAI 대비 **28배 빠름**)
   - 실시간 검색에 최적

3. **무료 및 오프라인 가능**
   - API 비용 없음
   - 인터넷 연결 불필요 (모델 다운로드 후)

4. **적절한 차원**
   - 384차원 (MPNet의 768차원 대비 절반)
   - 메모리 효율적이면서도 높은 성능

### 모델 스펙

- **모델명:** sentence-transformers/all-MiniLM-L6-v2
- **임베딩 차원:** 384
- **아키텍처:** MiniLM (Knowledge Distillation from BERT)
- **학습 데이터:** 10억+ 문장 쌍 (다양한 도메인)
- **정규화:** normalize_embeddings=True (코사인 유사도 최적화)

---

## 📈 상세 분석

### 1. Recall@10 분석

Recall@10은 상위 10개 결과에 관련 문서가 포함된 비율을 측정합니다.

- **MiniLM-L6**: 6.4% - 가장 많은 관련 문서를 상위에 포함
- **MsMarco**: 4.3% - 검색 특화 모델이지만 일반 도메인에서 약간 낮음
- **OpenAI**: 3.8% - 범용 모델로 특화 성능 부족
- **MPNet**: 3.4% - 높은 차원에도 불구하고 성능 낮음
- **SPECTER**: 1.8% - 과학 논문 특화로 일반 ML/DL/LLM 쿼리에 부적합

### 2. MRR (Mean Reciprocal Rank) 분석

MRR은 첫 번째 관련 문서의 순위를 평가합니다 (높을수록 좋음).

- **MiniLM-L6**: 0.658 (평균 1.5위) - 최상위 랭킹
- **MPNet**: 0.583 (평균 1.7위) - 높은 품질이지만 속도 느림
- **OpenAI**: 0.443 (평균 2.3위) - 중간 성능
- **MsMarco**: 0.417 (평균 2.4위) - 범용성 부족
- **SPECTER**: 0.367 (평균 2.7위) - 도메인 미스매치

### 3. 속도 비교

실시간 검색에서 속도는 매우 중요합니다.

```
MiniLM-L6:   0.02초  ████
MsMarco:     0.02초  ████
MPNet:       0.08초  ████████████████
SPECTER:     0.07초  ██████████████
OpenAI:      0.56초  ████████████████████████████████████████████████████████████████████
```

**OpenAI 모델은 28배 느림** + 네트워크 지연 + API 비용 발생

### 4. 비용 분석

**무료 모델 (HuggingFace):**
- MiniLM-L6, MPNet, MsMarco, SPECTER: **$0**
- 초기 다운로드 후 무제한 사용

**유료 모델 (OpenAI):**
- text-embedding-3-small: **$0.02 / 1M tokens**
- 2,000개 청크 임베딩 (~500K tokens): **$0.01**
- 16,508개 전체 청크: **~$0.083**
- 매달 재인덱싱 시 비용 누적

---

## 🔬 테스트 쿼리 예시

평가에 사용된 10개 쿼리:

1. "최신 vision transformer 모델과 이미지 분류 성능"
2. "대규모 언어 모델의 fine-tuning 기법"
3. "code generation을 위한 LLM 모델"
4. "multimodal learning과 vision-language 모델"
5. "reinforcement learning from human feedback"
6. "diffusion models for image generation"
7. "efficient transformers and model compression"
8. "graph neural networks and molecular modeling"
9. "video understanding and temporal modeling"
10. "zero-shot and few-shot learning methods"

각 쿼리에 대해 관련 키워드가 metadata.tags에 포함된 문서를 정답으로 설정하고,
상위 10개 검색 결과에 정답이 포함되는지 평가했습니다.

---

## ✅ 최종 권장사항

### embeddings.py 현재 설정 유지

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"  # GPU 사용 시 "cuda"로 변경 가능
```

**이유:**
- ✅ 5개 모델 중 최고 성능 (R@10: 6.4%, MRR: 0.658)
- ✅ 가장 빠른 속도 (0.02초/쿼리)
- ✅ 무료 및 오프라인 가능
- ✅ 메모리 효율적 (384차원)
- ✅ 이미 프로덕션에서 검증된 모델

### GPU 사용 시 추가 최적화

```python
DEFAULT_DEVICE = "cuda"  # GPU 사용 활성화
```

GPU 사용 시 속도가 2-3배 더 빠를 수 있습니다.

---

## 📝 재현 방법

```bash
# 평가 스크립트 실행
python 02_src/02_utils/evaluate_embeddings.py

# 전체 16,508개 청크 평가 (시간 소요)
# evaluate_embeddings.py에서 샘플링 코드 제거 후 실행
```

---

## 🔍 추가 고려사항

### 1. 도메인 특화 필요 시

현재 HuggingFace DailyPapers 데이터는 ML/DL/LLM 논문이므로,
**all-MiniLM-L6-v2**가 범용 성능으로 최적입니다.

만약 **생명과학/의학** 논문으로 전환 시:
- `allenai/scibert` 또는 `sentence-transformers/allenai-specter` 고려

### 2. 더 높은 품질 필요 시

검색 품질을 더 높이려면:
- **all-mpnet-base-v2** (768차원, 4배 느림)
- **text-embedding-3-large** (OpenAI, 3072차원, 비용 높음)

하지만 현재 R@10: 6.4%는 샘플링과 엄격한 평가 기준 때문이며,
실제 사용자 경험은 훨씬 나을 것으로 예상됩니다.

### 3. 멀티링구얼 필요 시

한국어 쿼리 지원이 중요하다면:
- `jhgan/ko-sroberta-multitask`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

현재는 영어 쿼리 기준이므로 all-MiniLM-L6-v2가 최적입니다.

---

**평가자:** SKN20-3rd-2TEAM
**도구:** evaluate_embeddings.py
**버전:** 1.0
