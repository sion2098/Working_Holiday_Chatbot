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
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

COMPARE_FIELDS = {
    "모집 인원": "모집 인원 연간 인원 정원",
    "신청 기간": "신청 기간 연중 분기별 접수",
    "신청 자격 요건": "연령 나이 자격 조건 초기 자금",
    "비자 주요 특징": "체류 기간 취업 제한 학업 가능"
}


COUNTRY_MAP = {
    "🇦🇺 호주": "australia",
    "🇯🇵 일본": "japan",
    "🇨🇦 캐나다": "canada",
    "🇳🇿 뉴질랜드": "newzealand",
    "🇩🇪 독일": "germany",
}

COUNTRY_KEYWORDS = {
    "호주": "australia",
    "일본": "japan",
    "캐나다": "canada",
    "뉴질랜드": "newzealand",
    "독일": "germany",
}


ALL_COUNTRIES = ["australia", "japan", "canada", "newzealand", "germany"]
REV_COUNTRY = {v: k for k, v in COUNTRY_MAP.items()}

COMPARE_COUNTRY_PROMPT = """
당신은 워킹홀리데이 공식 안내 문서(txt 파일)에 기반하여
여러 국가의 제도를 **비교 정리하는 안내 도우미**입니다.
아래 지침을 반드시 모두 준수하세요.

이 챗봇은 제공된 문서에 기반한 정보만 사용합니다.

[답변 범위]
각 파일의 역할은 다음과 같다:
- safety.txt : 사고, 범죄, 여권 분실·도난, 응급상황, 긴급 연락처, 재외공관, 위기 대응 절차
- visa.txt : 비자 종류, 신청 절차, 자격 요건, 모집 인원, 체류 기간, 취업·학업 제한, 비자 연장 조건
- jobs.txt : 취업 준비, TFN, 이력서·면접, 임금, 근무 조건, 노동법, 불법 고용 주의
- settlement.txt : 주거 형태, 생활비, 은행 계좌, 휴대폰·인터넷, 초기 정착, 일상생활 정보
- return.txt : 세금 환급, 연금 환급, 귀국 준비, 계좌·계약 해지, 이삿짐 정리

[파일별 정보 활용 지침]
- 모집 인원, 신청 기간, 신청 자격 요건, 연령 요건, 체류 가능 기간은 visa.txt 파일을 우선적으로 참고하세요.
- 비자 주요 특징(취업·학업 제한, 체류 조건)은 visa.txt와 jobs.txt의 내용을 함께 참고하세요.
- 특정 항목에 대한 정보가 한 파일에 없을 경우, 위 파일 범위 내에서 관련 내용을 종합하여 정리하세요.
- 위 파일들에 근거가 전혀 없는 경우에만 "검색된 문서 범위 내에서 확인되지 않음"으로 표기하세요.


[비교 대상 국가]
- 비교 대상 국가는 다음과 같습니다: {countries}
- 위에 명시된 국가 외의 국가는 포함하지 마세요.

[비교 기준]
- 기본 비교 항목은 아래 5개입니다.

모집 인원  
신청 기간  
신청 자격 요건  
연령 요건  
체류 가능 기간  

- 사용자가 비교 항목을 명시하지 않은 경우,
  위 5개 항목을 기준으로 **전체 비교 표**를 작성하세요.
- 사용자가 특정 비교 항목을 명시한 경우,
  해당 항목에 대해서만 **국가별로 더 구체적으로 비교**하세요.

[출력 형식 규칙]
- 반드시 **표(table)** 형태로 작성하세요.
- 행(row): 국가
- 열(column): 비교 항목
- 모든 셀을 채우세요.
- 문서 근거가 전혀 없는 경우에만
  "검색된 문서 범위 내에서 확인되지 않음"으로 표기하세요.
- 임의 요약, 추측, 일반 상식 사용은 금지됩니다.

[금지 사항]
- 개인적인 추천, 판단, 우열 비교 표현 금지
- 서술형 설명 금지 (표 외 형식 금지)
- 수도, 언어, 관광, 문화, 체험 프로그램 설명 금지

[마무리]
- 표 아래에 반드시 아래 섹션을 포함하세요.

### 다음으로 도움이 될 수 있는 내용
- 최대 3개 bullet point
- 한 줄씩 간결하게
- 추천·판단·질문 유도 금지

====================
[국가별 공식 문서]
====================
{contexts}

질문:
{question}
"""
def format_sources_by_country(docs: List[Document]) -> str:
    seen = set()
    blocks = []

    for d in docs:
        country = d.metadata.get("country")
        site = d.metadata.get("site")
        url = d.metadata.get("url")

        if not country or country in seen:
            continue

        seen.add(country)

        blocks.append(
            f"- **{site} ({REV_COUNTRY.get(country, country)})**\n"
            f"  · {url}"
        )

    if not blocks:
        return ""

    return "\n\n---\n📄 **참고 출처**\n" + "\n".join(blocks)
VECTORSTORE_DIR = "vectorstore/faiss"

BASE_DATA_DIR = "data"

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
                        "url": country_page_url(country)
                    }
                )
            )

    return docs

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ✅ 이미 저장된 벡터스토어가 있으면 로드
    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # ❌ 없으면 새로 생성
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(load_documents())

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # ✅ 파일로 저장
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    return vectorstore

vectorstore = get_vectorstore()


def retrieve_by_field(country: str, field_query: str, k=3):
    """
    항목별 retriever:
    특정 국가 + 특정 항목(모집 인원, 기간 등)에 대한 문서만 검색
    """
    query = f"{field_query} 워킹홀리데이"
    results = vectorstore.similarity_search(query, k=k)
    return [d for d in results if d.metadata.get("country") == country]


def country_page_url(country: str) -> str:
    COUNTRY_URL_MAP = {
        "australia": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100002",
        "japan": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100012",
        "canada": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100013",
        "newzealand": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100003",
        "germany": "https://whic.mofa.go.kr/whic/nation/info.jsp?boardNo=100010",
    }
    return COUNTRY_URL_MAP.get(country, "https://whic.mofa.go.kr/whic/main/")
def format_context(docs: List[Document], max_len=2000) -> str:
    text = ""
    for d in docs:
        if len(text) > max_len:
            break
        text += d.page_content + "\n\n"
    return text.strip()


def build_compare_context(country: str) -> str:
    """
    비교 질문 전용 컨텍스트 생성
    항목별로 문서를 나눠서 LLM에 전달
    """
    context = f"### {REV_COUNTRY[country]}\n"

    for field, query in COMPARE_FIELDS.items():
        docs = retrieve_by_field(country, query, k=3)
        snippet = format_context(docs, max_len=400)

        context += f"\n[{field}]\n"
        context += snippet if snippet else "검색된 문서 범위 내에서 확인되지 않음"
        context += "\n"

    return context

def answer_compare(question: str, countries: List[str]) -> str:
    # 1️⃣ 국가 미지정 → 전체 국가 비교
    if not countries:
        countries = ALL_COUNTRIES

    # 2️⃣ 국가별 비교 컨텍스트 생성
    context_blocks = []
    for c in countries:
        ctx = build_compare_context(c)
        context_blocks.append(f"[{c.upper()} 공식 문서]\n{ctx}")

    prompt = COMPARE_COUNTRY_PROMPT.format(
        countries=", ".join(countries),
        contexts="\n\n".join(context_blocks),
        question=question
    )

    answer = llm.invoke(prompt).content.strip()

    # 3️⃣ 출처 정리
    source_docs = [
        Document(
            page_content="",
            metadata={
                "country": c,
                "site": "워킹홀리데이 인포센터 (외교부)",
                "url": country_page_url(c)
            }
        )
        for c in countries
    ]

    answer += format_sources_by_country(source_docs)
    return answer

if __name__ == '__main__':
    question = "호주랑 캐나다 워홀 어떤지 비교해줘"
    target = ['australia']

    answer_compare(question, target)
