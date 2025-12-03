import os
import json

# 확인할 폴더들 (예: 2025-W45, 2025-W46 등)
folders = [f for f in os.listdir() if os.path.isdir(f) and f.startswith("2025-W")]

for folder in folders:
    print(f"\n🔹 Checking folder: {folder}")
    empty_content_count = 0
    incomplete_metadata_count = 0

    # 폴더 안의 JSON 파일 확인
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "sr", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # content 확인
                    page_content = data.get("content", "").strip()
                    if not page_content:
                        print(f"⚠️ Empty content: {file_path}")
                        empty_content_count += 1

                    # metadata 확인
                    metadata = data.get("metadata", {})
                    missing_fields = []
                    for key in ["paper_name", "github_url", "huggingface_url", "upvote", "tags"]:
                        if key not in metadata or metadata[key] in [None, "", []]:
                            missing_fields.append(key)
                    if missing_fields:
                        print(f"⚠️ Incomplete metadata ({', '.join(missing_fields)}): {file_path}")
                        incomplete_metadata_count += 1

                except Exception as e:
                    print(f"❌ Failed to load JSON: {file_path} ({e})")
                    empty_content_count += 1
                    incomplete_metadata_count += 1

    print(f"✅ Total empty content in {folder}: {empty_content_count}")
    print(f"✅ Total incomplete metadata in {folder}: {incomplete_metadata_count}")
