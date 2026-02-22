# ============================================================
# 1. 환경 설정 & 라이브러리
# ============================================================
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# 2. Streamlit 기본 UI 설정
# ============================================================
st.set_page_config(page_title="워홀 RAG 챗봇", page_icon="🌍", layout="wide")
st.title("🌍 워홀 챗봇")
st.caption("공식 문서 기반으로 워킹홀리데이 준비 단계를 따라가며 답변합니다.")


# ============================================================
# 3. 데이터 & 국가 설정
# ============================================================
BASE_DATA_DIR = "data"

COUNTRY_MAP = {
    "🇦🇺 호주": "australia",
    "🇨🇦 캐나다": "canada",
    "🇯🇵 일본": "japan",
    "🇳🇿 뉴질랜드": "newzealand",
    "🇩🇪 독일": "germany",
}

COUNTRY_KEYWORDS = {
    "호주": "australia",
    "캐나다": "canada",
    "일본": "japan",
    "뉴질랜드": "newzealand",
    "독일": "germany",
}


# ============================================================
# 4. 사용자 상태 정의
# ============================================================
STATE_LEVEL = {
    "BEGINNER": 1,
    "PREPARING": 2,
    "ON_SITE": 3,
    "COMPARISON": 0
}

PREPARING_KEYWORDS = ["비자", "잔고증명", "서류", "신청", "비용"]
ON_SITE_KEYWORDS = ["TFN", "유심", "은행", "집", "세금", "일"]


# ============================================================
# 5. 문서 로딩 & 벡터스토어
# ============================================================
def load_all_documents(base_dir):
    docs = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".txt"):
                continue

            fp = os.path.join(root, fname)
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read().strip()

            country = os.path.normpath(fp).split(os.sep)[1]

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "country": country,
                        "source": fname
                    }
                )
            )
    return docs


@st.cache_resource
def build_vectorstore():
    docs = load_all_documents(BASE_DATA_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# ============================================================
# 6. 상태 분류 & 승격 로직
# ============================================================
def classify_state(question: str) -> str:
    if any(k in question for k in ["비교", "추천", "어디"]):
        return "COMPARISON"
    if any(k in question for k in ON_SITE_KEYWORDS):
        return "ON_SITE"
    if any(k in question for k in PREPARING_KEYWORDS):
        return "PREPARING"
    return "BEGINNER"


def update_state(current_state, question, counters):
    if any(k in question for k in PREPARING_KEYWORDS):
        counters["PREPARING"] += 1
    if any(k in question for k in ON_SITE_KEYWORDS):
        counters["ON_SITE"] += 1

    if counters["ON_SITE"] >= 2:
        return "ON_SITE", counters
    if counters["PREPARING"] >= 2 and current_state == "BEGINNER":
        return "PREPARING", counters

    return current_state, counters


# ============================================================
# 7. 국가 자동 감지
# ============================================================
def detect_country(question):
    for k, v in COUNTRY_KEYWORDS.items():
        if k in question:
            return v
    return None


# ============================================================
# 8. 상태별 프롬프트
# ============================================================
STATE_PROMPTS = {
    "BEGINNER": "당신은 워홀 초보자에게 방향만 제시하는 상담원입니다.",
    "PREPARING": "당신은 워홀 준비 단계에서 순서와 주의사항을 알려주는 상담원입니다.",
    "ON_SITE": "당신은 현지 정착을 돕는 실무 상담원입니다.",
    "COMPARISON": "당신은 워홀 국가 선택을 돕는 상담원입니다.",
}


# ============================================================
# 9. Streamlit 세션 초기화
# ============================================================
if "state" not in st.session_state:
    st.session_state.state = "BEGINNER"
    st.session_state.counters = {"PREPARING": 0, "ON_SITE": 0}
    st.session_state.messages = [
        {"role": "assistant", "content": "워홀 관련해서 궁금한 걸 물어봐!"}
    ]


# ============================================================
# 10. 사이드바 (고급 국가 선택)
# ============================================================
with st.sidebar:
    st.subheader("🔧 고급 설정")
    use_manual_country = st.checkbox("특정 국가 기준으로 보기")
    selected_country = None

    if use_manual_country:
        label = st.selectbox("국가 선택", list(COUNTRY_MAP.keys()))
        selected_country = COUNTRY_MAP[label]


# ============================================================
# 11. 채팅 UI 출력
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# 12. 메인 채팅 로직
# ============================================================
user_q = st.chat_input("질문을 입력하세요")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    detected_state = classify_state(user_q)
    new_state, counters = update_state(
        st.session_state.state,
        user_q,
        st.session_state.counters
    )

    st.session_state.state = new_state
    st.session_state.counters = counters

    auto_country = detect_country(user_q)
    country_key = auto_country or selected_country

    docs = retriever.invoke(
        user_q,
        filter={"country": country_key} if country_key else None
    )

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
{STATE_PROMPTS[new_state]}

[공식 문서 정보]
{context}

[질문]
{user_q}
"""

    answer = llm.invoke(prompt).content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
