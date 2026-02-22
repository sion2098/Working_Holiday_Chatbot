import os
import re

# =========================
# 0) 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")           # 원본 txt들이 있는 곳
CLEAN_DIR = os.path.join(DATA_DIR, "_clean")        # 정제 txt 저장할 곳(자동 생성)

# =========================
# 1) "노이즈" 판별 규칙
# =========================
PAGE_MARKER_RE = re.compile(r"^=+\s*PAGE\s*\d+\s*=+$")  # ===== PAGE 1 =====
HEADER_RE = re.compile(r".*워킹홀리데이인포센터\s*\|\s*재외동포청.*")  # 날짜+출처 반복
NAV_RE = re.compile(r".*워홀비자.*국가/지역소개.*안전정보.*초기정착.*취업정보.*어학연수.*여행정보.*귀국준비.*")

# "URL + 페이지표시(예: 3/5)" 형태만 제거
URL_WITH_PAGENO_RE = re.compile(r"^https?://\S+\s+\d+/\d+$")

def is_noise_line(line: str) -> bool:
    """한 줄이 노이즈인지 판별"""
    if not line:
        return True  # 빈 줄은 일단 제거 (필요하면 유지하도록 바꿔도 됨)

    if PAGE_MARKER_RE.match(line):
        return True

    if HEADER_RE.match(line):
        return True

    if NAV_RE.match(line):
        return True

    if URL_WITH_PAGENO_RE.match(line):
        return True

    return False


def clean_text(text: str) -> str:
    """
    원본 텍스트에서 노이즈 라인만 제거하고,
    나머지는 순서 그대로 유지
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if is_noise_line(line):
            continue

        cleaned_lines.append(line)

    # 너무 딱딱하면 줄 간격 조금 살리기 (원하면 아래 2줄 제거 가능)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned


def clean_all_txt_under_data():
    """
    data/ 아래의 모든 .txt를 찾아서
    data/_clean/ 아래에 동일한 폴더 구조로 저장
    """
    os.makedirs(CLEAN_DIR, exist_ok=True)

    processed = 0
    skipped = 0

    for root, _, files in os.walk(DATA_DIR):
        # _clean 폴더는 다시 처리하지 않도록 제외
        if os.path.abspath(root).startswith(os.path.abspath(CLEAN_DIR)):
            continue

        for fname in files:
            if not fname.endswith(".txt"):
                continue

            src_path = os.path.join(root, fname)

            # data/ 기준 상대경로 유지 (예: australia/australia_visa.txt)
            rel_path = os.path.relpath(src_path, DATA_DIR)
            dst_path = os.path.join(CLEAN_DIR, rel_path)

            # 출력 폴더 자동 생성
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    raw = f.read()

                cleaned = clean_text(raw)

                with open(dst_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                processed += 1

            except Exception as e:
                print(f"❌ 실패: {src_path} -> {e}")
                skipped += 1

    print(f"\n✅ 완료: {processed}개 정제, ❌ 실패/스킵: {skipped}개")
    print(f"📁 정제 결과 폴더: {CLEAN_DIR}")


if __name__ == "__main__":
    clean_all_txt_under_data()
