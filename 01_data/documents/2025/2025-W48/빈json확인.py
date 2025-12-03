import os
import json
import pandas as pd

# 검사할 주차 폴더 지정
WEEK_DIR = "./01_data/documents/2025/2025-W48"   # 필요하면 수정!

def check_empty_context(week_dir=WEEK_DIR):
    problem_files = []

    for filename in os.listdir(week_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(week_dir, filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            context = data.get("context", "")

            if context is None or len(context.strip()) < 10:  # 10글자 미만이면 거의 비어있는 것
                problem_files.append({
                    "file": filename,
                    "context_length": len(context.strip()),
                    "paper_name": data.get("metadata", {}).get("paper_name", ""),
                    "huggingface_url": data.get("metadata", {}).get("huggingface_url", "")
                })
        except Exception as e:
            print(f"[ERROR] {filename} 읽기 실패: {e}")
            continue

    # 결과 출력
    print("\n===== 빈 context 의심 파일 =====")
    if not problem_files:
        print("🎉 모든 JSON이 정상입니다!")
    else:
        for p in problem_files:
            print(f"- {p['file']} (len={p['context_length']}): {p['paper_name']}")

        # CSV 저장
        df = pd.DataFrame(problem_files)
        csv_path = os.path.join(week_dir, "empty_context_check.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n📄 CSV로 저장됨: {csv_path}")

    return problem_files


if __name__ == "__main__":
    check_empty_context()
