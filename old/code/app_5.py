# ============================================================
# 0. 기본 설정 & 라이브러리
#    - Streamlit UI
#    - LangChain (RAG)
#    - 문서 로딩 / 벡터 검색
# ============================================================
import os
import glob
import re
import streamlit as st
from dotenv import load_dotenv
from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(
    page_title="워홀 RAG 챗봇",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 워홀 RAG 챗봇")
st.caption("공식 문서 기반 워킹홀리데이 상담 챗봇")

# ============================================================
# 1. 국가 설정
#    - UI 표시용 이름 ↔ 내부 키 매핑
# ============================================================
BASE_DATA_DIR = "data"

COUNTRY_MAP = {
    "🇦🇺 호주": "australia",
    "🇯🇵 일본": "japan",
    "🇨🇦 캐나다": "canada",
    "🇳🇿 뉴질랜드": "newzealand",
    "🇩🇪 독일": "germany",
}

# 질문 문장에서 국가 키워드 감지용
COUNTRY_KEYWORDS = {
    "호주": "australia",
    "일본": "japan",
    "캐나다": "canada",
    "뉴질랜드": "newzealand",
    "독일": "germany",
}

# 내부 키 → UI 라벨
REV_COUNTRY = {v: k for k, v in COUNTRY_MAP.items()}

# ============================================================
# 1-1. 국가별 추천 질문 (온보딩 UX)
# ============================================================
SUGGESTED_QUESTIONS = {
    "australia": [
        "호주 워홀 처음인데, 준비는 어디서부터 해야 해?",
        "호주 워홀 비자 신청할 때 잔고 증명은 어느 정도 필요해?",
        "호주랑 캐나다 워홀 조건을 비교해줘",
    ],
    "japan": [
        "일본 워홀은 나이 제한이 어떻게 돼?",
        "일본 워홀 비자 신청 절차를 순서대로 알려줘",
        "일본이랑 독일 워홀을 비교해줘",
    ],
    "canada": [
        "캐나다 워홀은 아무나 신청할 수 있어?",
        "캐나다 워홀은 경쟁이 치열하다고 하던데 왜 그래?",
        "호주랑 캐나다 워홀 조건을 비교해줘",
    ],
    "newzealand": [
        "뉴질랜드 워홀은 다른 나라보다 어떤 점이 달라?",
        "뉴질랜드 워홀 비자 신청 전에 준비해야 할 서류가 뭐야?",
        "뉴질랜드랑 호주 워홀 중에 어디가 더 나을까?",
    ],
    "germany": [
        "독일 워홀은 영어만으로도 가능한가요?",
        "독일 워홀 비자 신청 조건을 정리해줘",
        "독일이랑 일본 워홀을 비교해줘",
    ],
    None: [
        "워홀 국가를 아직 못 정했는데, 어떤 기준으로 선택하면 좋아?",
        "호주, 일본, 캐나다 워홀을 한 번에 비교해줘",
        "워홀 처음인데, 나라 고르기 전에 뭘 알아야 해?",
    ]
}

# ============================================================
# 2. 출처 관련 유틸
#    - 파일명 기반 섹션 추론
#    - (공식 사이트 구조: 국가별 단일 URL + 탭)
# ============================================================
def infer_section_from_filename(fp: str) -> str:
    """
    txt 파일명에 포함된 키워드를 기준으로
    '비자 / 취업 / 정착' 등의 섹션을 추론
    """
    name = os.path.basename(fp).lower()
    if "visa" in name:
        return "워홀비자 관련 정보"
    if "job" in name or "work" in name:
        return "취업 및 구직 정보"
    if "settle" in name or "life" in name:
        return "초기 정착 정보"
    if "safety" in name or "law" in name:
        return "안전 정보"
    return "기타 공식 정보"


def country_page_url(country: str) -> str:
    """
    워킹홀리데이 인포센터는
    국가별 단일 페이지 + 내부 탭 구조를 사용하므로
    출처 URL은 국가별 페이지 단위로 제공한다.
    """

    COUNTRY_URL_MAP = {
        "australia": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100002",
        "japan": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100012",
        "canada": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100013",
        "newzealand": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100003",
        "germany": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100010",
    }

    # 혹시 모를 예외 대비 (국가 미매핑 시 메인 페이지)
    return COUNTRY_URL_MAP.get(
        country,
        "https://whic.mofa.go.kr/whic/main/"
    )


# ============================================================
# 3. 문서 로딩 & 벡터스토어 구축
#    - data/국가/*.txt 로딩
#    - 출처 메타데이터 포함
# ============================================================
def load_documents() -> List[Document]:
    docs = []

    for country in COUNTRY_MAP.values():
        path = os.path.join(BASE_DATA_DIR, country)
        if not os.path.isdir(path):
            continue

        for fp in glob.glob(os.path.join(path, "**", "*.txt"), recursive=True):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "country": country,
                        "site": "워킹홀리데이 인포센터 (외교부)",
                        # "section": infer_section_from_filename(fp),
                        "url": country_page_url(country)
                    }
                )
            )

    return docs


@st.cache_resource
def build_vectorstore():
    """
    문서를 chunk 단위로 분리한 뒤
    OpenAI Embedding → FAISS 벡터스토어 생성
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(load_documents())

    return FAISS.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small")
    )


vectorstore = build_vectorstore()

# ============================================================
# 4. 검색 & 질문 유형 판단
# ============================================================
def retrieve_by_countries(query: str, countries: List[str], k=6):
    """
    비교 질문에서 숫자/조건 정보가 포함된 chunk를
    더 잘 가져오기 위해 검색 쿼리를 확장한다.
    """

    # 🔑 검색용 쿼리 확장 (사용자 질문은 변경하지 않음)
    search_query = f"""
    {query}
    워킹홀리데이 비자
    모집 인원 신청 기간 신청 자격 조건
    체류 기간 연령 제한
    """

    results = vectorstore.similarity_search(search_query, k=40)

    buckets = {c: [] for c in countries}
    for d in results:
        c = d.metadata.get("country")
        if c in buckets and len(buckets[c]) < k:
            buckets[c].append(d)

    return buckets

def format_context(docs: List[Document], max_len=2000) -> str:
    """
    LLM에 전달할 컨텍스트 문자열 구성
    """
    text = ""
    for d in docs:
        if len(text) > max_len:
            break
        text += d.page_content + "\n\n"
    return text.strip()


def extract_countries(q: str) -> List[str]:
    """
    질문 문장에서 언급된 국가 추출
    """
    return list({v for k, v in COUNTRY_KEYWORDS.items() if k in q})


def is_comparison(q: str, mentioned: List[str], base: Optional[str]) -> bool:
    """
    비교 질문 여부 판단
    """
    return (
        len(mentioned) >= 2
        or any(t in q for t in ["비교", "vs", "차이", "어디"])
    )

# ============================================================
# 5. 출처 표시 포맷
#    - 국가 페이지 URL + 섹션 설명
# ============================================================
def format_sources(docs: List[Document], max_sources=2) -> str:
    seen = set()
    blocks = []

    for d in docs:
        site = d.metadata.get("site")
        section = d.metadata.get("section")
        url = d.metadata.get("url")

        key = (site, section)
        if key in seen:
            continue
        seen.add(key)

        blocks.append(
            f"- **{site}**\n"
            f"  · {url}"
        )

        if len(blocks) >= max_sources:
            break

    if not blocks:
        return ""

    return "\n\n---\n📄 **참고 출처**\n" + "\n".join(blocks)

# ============================================================
# 6. LLM 설정
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ============================================================
# 7. 단일 국가 답변 생성
# ============================================================
def answer_single(question: str, country: str) -> str:
    docs = retrieve_by_countries(question, [country])[country]

    answer = llm.invoke(f"""
[기준 국가]
{REV_COUNTRY[country]}

[공식 문서]
{format_context(docs)}

[질문]
{question}

중요 지침:
- 이 질문은 '국가 비교'가 아닙니다.
- 표를 만들지 마세요.
- 수도, 언어, 관광 정보, 체험 프로그램(WWOOF/HelpX 등) 설명은 하지 마세요.
- 워킹홀리데이 비자 기준으로, **처음 준비할 때 필요한 절차와 조건**만 설명하세요.
- 설명은 단계별(1→2→3)로 정리하세요.
- 질문이 단일 사실(숫자, 인원, 기간 등)을 묻는 경우, 핵심 답변만 한 문장으로 제공하고 추가 설명은 포함하지 마세요.

추가 지침:
- 답변이 끝난 뒤,
  "다음으로 도움이 될 수 있는 내용" 제목으로
  다음 단계에서 준비하면 좋은 내용이나
  이어서 많이 묻는 질문을
  최대 3개 bullet point로 정리하세요.
- 각 bullet point는 한 줄로 간결하게 작성하세요.
- 추천이나 판단은 하지 마세요.
""").content.strip()

    answer += format_sources(docs)
    return answer

# ============================================================
# 8. 국가 비교 답변 생성
# ============================================================
def answer_compare(question: str, countries: List[str]) -> str:
    buckets = retrieve_by_countries(question, countries, k=3)
    blocks = []

    for c in countries:
        blocks.append(
            f"### {REV_COUNTRY[c]}\n{format_context(buckets[c], 1200)}"
        )

    answer = llm.invoke(f"""
아래 제공된 공식 문서를 근거로,
**사용자가 명시한 국가만** 워킹홀리데이 제도를 비교하세요.

중요 지침:
- 조건, 제도, 비자 관련 내용만 비교하세요.
- 수도, 언어, 관광, 체험 프로그램(WWOOF/HelpX 등)은 제외하세요.
- 개인적인 추천이나 판단은 하지 마세요.
- 반드시 표 형태로 작성하세요.
- 문서에 없는 정보는 '추가 확인 필요'로 표시하되,
    '검색된 문서 범위 내에서 확인되지 않음'이라고 설명하세요.
- 질문이 단일 사실(숫자, 인원, 기간 등)을 묻는 경우, 핵심 답변만 한 문장으로 제공하고 추가 설명은 포함하지 마세요.


{chr(10).join(blocks)}

질문: {question}
""").content.strip()

    all_docs = []
    for c in countries:
        all_docs.extend(buckets[c])

    answer += format_sources(all_docs)
    return answer

# ============================================================
# 9. 세션 상태 초기화
# ============================================================
for k, v in {
    "onboarded": False,
    "base_country": None,
    "messages": []
}.items():
    st.session_state.setdefault(k, v)

# ============================================================
# 10. 온보딩 (기준 국가 선택)
# ============================================================
if not st.session_state.onboarded:
    choice = st.radio(
        "기준 국가 선택",
        list(COUNTRY_MAP.keys()) + ["➕ 아직 정하지 않았어요"]
    )

    if st.button("시작하기"):
        st.session_state.base_country = (
            None if choice.endswith("어요") else COUNTRY_MAP[choice]
        )
        st.session_state.onboarded = True
        st.session_state.messages = [
            {"role": "assistant", "content": "궁금한 걸 자유롭게 물어봐 😊"}
        ]
        st.rerun()

    st.stop()

# ============================================================
# 11. 사이드바 (설정)
# ============================================================
with st.sidebar:
    st.subheader("⚙️ 설정")

    options = list(COUNTRY_MAP.keys()) + ["➕ 아직 정하지 않았어요"]
    current = st.session_state.base_country
    idx = options.index(
        "➕ 아직 정하지 않았어요" if current is None else REV_COUNTRY[current]
    )

    new = st.selectbox("기준 국가 변경", options, index=idx)
    if st.button("기준 국가 적용"):
        st.session_state.base_country = None if new.endswith("어요") else COUNTRY_MAP[new]
        st.rerun()

    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = [
            {"role": "assistant", "content": "대화를 초기화했어! 다시 질문해줘 😊"}
        ]
        st.rerun()

# ============================================================
# 12. 추천 질문 UI
# ============================================================
if len(st.session_state.messages) == 1:
    st.markdown(
        "💡 **아래는 사람들이 자주 묻는 질문이에요. "
        "버튼을 눌러 바로 질문해도 되고, 직접 입력해도 괜찮아요.**"
    )

    questions = SUGGESTED_QUESTIONS.get(
        st.session_state.base_country,
        SUGGESTED_QUESTIONS[None]
    )

    cols = st.columns(3)
    for col, q in zip(cols, questions):
        with col:
            if st.button(q):
                st.session_state.pending_question = q
                st.rerun()

# ============================================================
# 13. 채팅 UI
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("질문을 입력하세요")

if "pending_question" in st.session_state:
    user_q = st.session_state.pop("pending_question")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    mentioned = extract_countries(user_q)
    compare = is_comparison(user_q, mentioned, st.session_state.base_country)

    if compare:
        targets = mentioned or list(COUNTRY_MAP.values())
        answer = answer_compare(user_q, targets)
    else:
        country = mentioned[0] if mentioned else st.session_state.base_country
        if country is None:
            answer = answer_compare(user_q, list(COUNTRY_MAP.values()))
        else:
            answer = answer_single(user_q, country)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
