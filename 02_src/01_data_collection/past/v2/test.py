import json
import glob
from pathlib import Path
from collections import Counter

# documents_K 폴더의 모든 JSON 파일 경로 가져오기
base_path = Path(__file__).parent.parent.parent / "01_data" / "documents_T"
json_files = glob.glob(str(base_path / "**" / "*.json"), recursive=True)

print(f"총 JSON 파일 개수: {len(json_files)}\n")

# 모든 태그를 수집
all_tags = []

for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tags = data.get('metadata', {}).get('tags', [])
            all_tags.extend(tags)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")

# 태그 분석
tag_counts = Counter(all_tags)

# 결과 출력
print(f"총 태그 개수: {len(all_tags)}")
print(f"고유 태그 개수: {len(tag_counts)}\n")

print("=" * 60)
print("각 태그별 빈도:")
print("=" * 60)

# 빈도순으로 정렬하여 출력 (한 줄에 5개씩)
tags_list = tag_counts.most_common()
for i in range(0, len(tags_list), 5):
    row = tags_list[i:i+5]
    print(" | ".join([f"{tag}: {count}" for tag, count in row]))
