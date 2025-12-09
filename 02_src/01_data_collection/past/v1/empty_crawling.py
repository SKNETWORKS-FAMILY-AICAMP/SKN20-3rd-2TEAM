"""
비어있는 데이터를 검증하는 모듈
주차 범위 내의 JSON 파일들을 검사하여 필수 데이터가 누락된 논문의 URL을 수집합니다.
"""

import os
import json
import logging
import datetime
from typing import List, Dict

TAG_COUNT = 3

def empty_crawling(year: int, start: int, end: int) -> List[Dict[str, str]]:
    """
    주차 범위 내의 논문 데이터를 검증하고, 필수 데이터가 누락된 논문의 URL을 반환

    Args:
        year: 연도 (예: 2023)
        start: 시작 주차 (0~52)
        end: 끝나는 주차 (0~52)

    Returns:
        List[Dict[str, str]]: 필수 데이터가 누락된 논문 정보 리스트 (filename, paper_url 포함)

    Raises:
        ValueError: 파라미터가 유효하지 않을 때
        FileNotFoundError: 기본 데이터 디렉토리가 존재하지 않을 때

    검증 항목:
        - context: 비어있거나 50자 미만이면 안 됨
        - metadata.paper_name: 비어있으면 안 됨
        - metadata.huggingface_url: 비어있으면 안 됨
        - metadata.tags: 비어있거나 3개가 아니면 안 됨
    """
    # ===== 파라미터 유효성 검증 =====

    # 1. year 유효성 검증 (1900~2100 범위)
    if not isinstance(year, int) or year < 1900 or year > datetime.datetime.now().year + 1:
        raise ValueError(f"[ERROR] 유효하지 않은 연도: {year} (1900~2100 범위여야 합니다)")

    # 2. start, end 유효성 검증 (0~52 범위)
    if not isinstance(start, int) or start < 0 or start > 52:
        raise ValueError(f"[ERROR] 유효하지 않은 시작 주차: {start} (0~52 범위여야 합니다)")

    if not isinstance(end, int) or end < 0 or end > 52:
        raise ValueError(f"[ERROR] 유효하지 않은 종료 주차: {end} (0~52 범위여야 합니다)")

    # 3. start <= end 검증
    if start > end:
        raise ValueError(
            f"[ERROR] 시작 주차({start})가 종료 주차({end})보다 클 수 없습니다. "
            f"start <= end 조건을 만족해야 합니다."
        )

    # 4. 기본 데이터 디렉토리 존재 검증
    base_data_dir = f"././01_data/documents/{year}"
    if not os.path.exists(base_data_dir):
        raise FileNotFoundError(
            f"[ERROR] 연도 디렉토리가 존재하지 않습니다: {base_data_dir}\n"
            f"해당 연도({year})의 데이터를 먼저 수집해주세요."
        )

    # 검증 통과 로그
    logging.info(f"[SUCCESS] 파라미터 검증 완료: year={year}, start=W{start:02d}, end=W{end:02d}")

    # 문제가 있는 논문의 URL을 담을 리스트
    invalid_urls = []

    # 통계 정보
    total_files = 0
    invalid_files = 0

    # 주차 범위 순회
    for week in range(start, end + 1):
        week_str = f"{year}-W{week:02d}"
        docs_dir = f"././01_data/documents/{year}/{week_str}"

        # 디렉토리가 존재하지 않으면 스킵
        if not os.path.exists(docs_dir):
            logging.warning(f"[WARNING] 디렉토리가 존재하지 않음: {docs_dir}")
            continue

        print(f"\n[STATUS] 검증 중: {week_str}")
        week_invalid_count = 0

        # 해당 주차의 JSON 파일들 검사
        for filename in os.listdir(docs_dir):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(docs_dir, filename)
            total_files += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 데이터 검증
                is_invalid = False
                issues = []

                # 1. context 검증 (비어있거나 너무 짧으면 안 됨)
                context = data.get('context', '').strip()
                if not context or len(context) < 50:
                    is_invalid = True
                    issues.append(f"context 부족 (길이: {len(context)})")

                # 2. metadata 검증
                metadata = data.get('metadata', {})

                # 2-1. paper_name 검증
                paper_name = metadata.get('paper_name', '').strip()
                if not paper_name:
                    is_invalid = True
                    issues.append("paper_name 누락")

                # 2-2. huggingface_url 검증
                huggingface_url = metadata.get('huggingface_url', '').strip()
                if not huggingface_url:
                    is_invalid = True
                    issues.append("huggingface_url 누락")

                # 2-3. tags 검증 (정확히 3개여야 함)
                tags = metadata.get('tags', [])
                if not isinstance(tags, list) or len(tags) != TAG_COUNT:
                    is_invalid = True
                    issues.append(f"tags 오류 (개수: {len(tags) if isinstance(tags, list) else 'N/A'})")

                # 문제가 있으면 dict를 리스트에 추가
                if is_invalid:
                    invalid_urls.append({
                        "filename": filename,
                        "paper_url": huggingface_url if huggingface_url else ""
                    })
                    invalid_files += 1
                    week_invalid_count += 1

                    print(f"  [EMPTY] {filename}: {', '.join(issues)}")
                    if huggingface_url:
                        print(f"     URL: {huggingface_url}")

            except json.JSONDecodeError as e:
                # JSON 파싱 오류
                invalid_urls.append(f"[파일: {filename}] - JSON 파싱 오류")
                invalid_files += 1
                week_invalid_count += 1
                print(f"  [FAILED] {filename}: JSON 파싱 실패 ({e})")

            except Exception as e:
                # 기타 오류
                invalid_urls.append(f"[파일: {filename}] - 기타 오류")
                invalid_files += 1
                week_invalid_count += 1
                print(f"  [FAILED] {filename}: 오류 발생 ({e})")

        # 주차별 통계 출력
        if week_invalid_count == 0:
            print(f"  [STATE] 모든 파일 정상")
        else:
            print(f"  [STATE] 문제 파일 수: {week_invalid_count}")

    # 전체 통계 출력
    print("\n" + "=" * 60)
    print("[STATUS] 검증 완료 통계")
    print(f"  - 검증 대상 주차: W{start:02d} ~ W{end:02d}")
    print(f"  - 총 파일 수: {total_files}")
    print(f"  - 정상 파일: {total_files - invalid_files}")
    print(f"  - 문제 파일: {invalid_files}")
    print("=" * 60)

    return invalid_urls


def convert_tags_to_individual_fields(year: int, start: int, end: int, backup: bool = True) -> Dict[str, int]:
    """
    JSON 파일의 metadata.tags 배열을 tag1, tag2, tag3 개별 필드로 변환

    Args:
        year: 연도 (예: 2025)
        start: 시작 주차 (0~52)
        end: 끝나는 주차 (0~52)
        backup: 백업 생성 여부 (기본값: True)

    Returns:
        Dict[str, int]: 처리 통계 {'total': 전체, 'converted': 변환 성공, 'skipped': 스킵, 'failed': 실패}

    Raises:
        ValueError: 파라미터가 유효하지 않을 때
        FileNotFoundError: 기본 데이터 디렉토리가 존재하지 않을 때
    """
    import shutil

    # 파라미터 유효성 검증
    if not isinstance(year, int) or year < 1900 or year > datetime.datetime.now().year + 1:
        raise ValueError(f"[ERROR] 유효하지 않은 연도: {year}")

    if start > end:
        raise ValueError(
            f"[ERROR] 시작 주차({start})가 종료 주차({end})보다 클 수 없습니다."
        )

    # 기본 데이터 디렉토리 존재 검증
    base_data_dir = f"././01_data/documents/{year}"
    if not os.path.exists(base_data_dir):
        raise FileNotFoundError(
            f"[ERROR] 연도 디렉토리가 존재하지 않습니다: {base_data_dir}"
        )

    logging.info(f"[START] tags 배열을 tag1, tag2, tag3 필드로 변환 시작: {year}-W{start:02d} ~ W{end:02d}")

    # 통계 정보
    stats = {
        'total': 0,
        'converted': 0,
        'skipped': 0,
        'failed': 0
    }

    # 주차 범위 순회
    for week in range(start, end + 1):
        week_str = f"{year}-W{week:02d}"
        docs_dir = f"././01_data/documents/{year}/{week_str}"

        if not os.path.exists(docs_dir):
            logging.warning(f"[WARNING] 디렉토리가 존재하지 않음: {docs_dir}")
            continue

        print(f"\n[STATUS] 변환 중: {week_str}")

        # 해당 주차의 JSON 파일들 처리
        for filename in os.listdir(docs_dir):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(docs_dir, filename)
            stats['total'] += 1

            try:
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # metadata 확인
                metadata = data.get('metadata', {})
                tags = metadata.get('tags', [])

                # tags가 이미 없고 tag1, tag2, tag3가 있으면 스킵
                if 'tags' not in metadata and all(f'tag{i}' in metadata for i in range(1, 4)):
                    stats['skipped'] += 1
                    print(f"  [SKIP] {filename}: 이미 변환됨")
                    continue

                # tags가 없거나 리스트가 아니면 스킵
                if not isinstance(tags, list):
                    stats['skipped'] += 1
                    print(f"  [SKIP] {filename}: tags가 리스트 형태가 아님")
                    continue

                # 백업 생성
                if backup:
                    backup_path = file_path + ".backup"
                    if not os.path.exists(backup_path):
                        shutil.copy2(file_path, backup_path)

                # tags 배열을 tag1, tag2, tag3로 변환
                metadata['tag1'] = tags[0] if len(tags) > 0 else ""
                metadata['tag2'] = tags[1] if len(tags) > 1 else ""
                metadata['tag3'] = tags[2] if len(tags) > 2 else ""

                # tags 필드 제거
                del metadata['tags']

                # 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                stats['converted'] += 1
                print(f"  [CONVERTED] {filename}: tags -> tag1, tag2, tag3")

            except json.JSONDecodeError as e:
                stats['failed'] += 1
                print(f"  [FAILED] {filename}: JSON 파싱 실패 ({e})")

            except Exception as e:
                stats['failed'] += 1
                print(f"  [FAILED] {filename}: 변환 오류 ({e})")

    # 전체 통계 출력
    print("\n" + "=" * 60)
    print("[STATUS] 변환 완료 통계")
    print(f"  - 변환 대상 주차: W{start:02d} ~ W{end:02d}")
    print(f"  - 총 파일 수: {stats['total']}")
    print(f"  - 변환 성공: {stats['converted']}")
    print(f"  - 스킵: {stats['skipped']}")
    print(f"  - 실패: {stats['failed']}")
    print("=" * 60)

    logging.info(f"[DONE] 변환 완료: 성공 {stats['converted']}, 스킵 {stats['skipped']}, 실패 {stats['failed']}")

    return stats


# 실행 예시
if __name__ == "__main__":
    # 로깅 설정 (선택적)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # ===== 사용 예시 1: 데이터 검증 =====
    # 예시: 45주차부터 49주차까지 검증
    # invalid_paper_urls = empty_crawling(year = 2025, start = 45, end = 49)

    # # 결과 출력
    # if invalid_paper_urls:
    #     print(f"\n[WARNING] 다시 크롤링이 필요한 논문 URL 목록 ({len(invalid_paper_urls)}개):")
    #     for idx, dict in enumerate(invalid_paper_urls, 1):
    #         print(f"{idx}. {dict['filename']} - {dict['paper_url']}")
    # else:
    #     print("\n[SUCCESS] 모든 논문 데이터가 정상입니다!")

    # ===== 사용 예시 2: tags 배열을 tag1, tag2, tag3 필드로 변환 =====
    # 예시: 45주차 데이터의 tags 배열을 개별 필드로 변환 (백업 생성)
    stats = convert_tags_to_individual_fields(year=2025, start=46, end=49, backup=True)

    # 변환 후 검증
    print("\n[INFO] 변환 완료. 샘플 파일을 확인하여 결과를 검증하세요.")
    print("      백업 파일은 원본 파일명에 '.backup' 확장자가 붙어 있습니다.")