import pdfplumber
import os

# =========================
# 1. PDF 파일 목록 (여기에 6개 넣기)
# =========================
pdf_files = [
    "뉴질랜드_워홀비자_재외동포청.pdf",
    "뉴질랜드_초기정착_재외동포청.pdf",
    "뉴질랜드_취업정보_재외동포청.pdf",
    "뉴질랜드_귀국준비_재외동포청.pdf",
    "뉴질랜드_국가지역_재외동포청.pdf",
    "뉴질랜드_안전정보_재외동포청.pdf",
]

# =========================
# 2. 출력 폴더 (자동 생성)
# =========================
output_dir = "txt_output"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 3. 변환 로직
# =========================
for pdf_path in pdf_files:
    if not os.path.exists(pdf_path):
        print(f"❌ 파일 없음: {pdf_path}")
        continue

    txt_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_full.txt"
    txt_path = os.path.join(output_dir, txt_name)

    print(f"📄 변환 중: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        all_text = []
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            all_text.append(f"\n===== PAGE {page_num} =====\n")
            if text:
                all_text.append(text)
            else:
                all_text.append("[⚠️ 이 페이지에서 추출된 텍스트 없음]")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    print(f"✅ 완료: {txt_path}")

print("\n🎉 모든 PDF 변환 완료")
