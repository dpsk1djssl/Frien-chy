# -*- coding: utf-8 -*-
# main_inference_final.py
# LangGraph + PostgreSQL Memory 개선 버전 - 동적 브랜드 인식

import os
import uuid
import re
from typing import List, Dict, Any, TypedDict, Set, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient

# PostgreSQL 관련 import
import psycopg
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --------------------
# Config
# --------------------
CFG_NAME = "LangGraph_With_Dynamic_Brand_Recognition_v5.0"
EMBED_MODEL = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RETRIEVE = 20  # 검색량 더 증가
FINAL_TOPK = 8   # 최종 선택량 증가
MIN_SCORE = float(os.getenv("QDRANT_MIN_SCORE", "0.40"))  # 임계값 더 하향 조정

# --------------------
# Builders
# --------------------
def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_qdrant_vectorstore(embeddings):
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST"),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True,
        https=False
    )
    collection = os.getenv("QDRANT_COLLECTION", "franchise-db-1")
    vs = LCQdrant(
        client=client,
        collection_name=collection,
        embeddings=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )
    return vs.as_retriever(search_kwargs={"k": K_RETRIEVE}), collection

def build_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

# --------------------
# Custom PostgreSQL Chat Message History
# --------------------
class CustomPostgresChatMessageHistory(BaseChatMessageHistory):
    """PostgreSQL 기반의 커스텀 채팅 메시지 히스토리"""
    
    def __init__(self, session_id: str, connection_string: str, table_name: str = "message_store"):
        self.session_id = session_id
        self.connection_string = connection_string
        self.table_name = table_name
        self._create_table_if_not_exists()
        self._messages_cache = None

    def _create_table_if_not_exists(self):
        """메시지 저장 테이블이 없으면 생성"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_session_id 
                    ON {self.table_name} (session_id);
                """)
                conn.commit()

    def _refresh_cache(self):
        """메시지 캐시 새로고침"""
        messages = []
        try:
            with psycopg.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT message_type, content FROM {self.table_name} "
                        f"WHERE session_id = %s ORDER BY created_at ASC",
                        (self.session_id,)
                    )
                    for message_type, content in cur.fetchall():
                        if message_type == "human":
                            messages.append(HumanMessage(content=content))
                        elif message_type == "ai":
                            messages.append(AIMessage(content=content))
        except Exception as e:
            print(f"메시지 조회 오류: {e}")
        self._messages_cache = messages

    @property
    def messages(self) -> List[BaseMessage]:
        """세션의 모든 메시지 조회 (캐시 사용)"""
        if self._messages_cache is None:
            self._refresh_cache()
        return self._messages_cache or []

    def add_message(self, message: BaseMessage) -> None:
        """새 메시지 추가"""
        try:
            message_type = "human" if isinstance(message, HumanMessage) else "ai"
            with psycopg.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {self.table_name} (session_id, message_type, content) "
                        f"VALUES (%s, %s, %s)",
                        (self.session_id, message_type, message.content)
                    )
                    conn.commit()
            # 캐시 업데이트
            if self._messages_cache is not None:
                self._messages_cache.append(message)
        except Exception as e:
            print(f"메시지 추가 오류: {e}")

    def clear(self) -> None:
        """세션의 모든 메시지 삭제"""
        try:
            with psycopg.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE session_id = %s",
                        (self.session_id,)
                    )
                    conn.commit()
            self._messages_cache = []
        except Exception as e:
            print(f"메시지 삭제 오류: {e}")

# --------------------
# 동적 브랜드/카테고리 인식기 (완전히 새로 작성)
# --------------------
class DynamicBrandExtractor:
    """동적 브랜드 및 카테고리 추출기"""
    
    def __init__(self):
        # 브랜드명 패턴 (동적)
        self.brand_patterns = [
            # 한국어 브랜드명 (XX치킨, XX피자 등)
            r'[가-힣]{2,10}(?:치킨|chicken)',
            r'[가-힣]{2,10}(?:피자|pizza)',
            r'[가-힣]{2,10}(?:버거|burger)',
            r'[가-힣]{2,10}(?:카페|cafe|커피)',
            r'[가-힣]{2,10}(?:족발|보쌈)',
            r'[가-힣]{2,10}(?:떡볶이|분식)',
            r'[가-힣]{2,10}(?:마라탕|훠궈)',
            r'[가-힣]{2,10}(?:곱창|막창)',
            r'[가-힣]{2,10}(?:삼겹살|고기)',
            r'[가-힣]{2,10}(?:학원|교육)',
            r'[가-힣]{2,10}(?:네일|미용)',
            r'[가-힣]{2,10}(?:마사지|힐링)',
            r'[가-힣]{2,10}(?:편의점|마트)',
            
            # 영문/숫자 브랜드명
            r'[A-Za-z0-9]{2,15}(?:\s|$|치킨|피자|burger|chicken|pizza|cafe)',
            
            # 특별한 패턴들
            r'맥도날드|KFC|버거킹|롯데리아|맘스터치',
            r'스타벅스|이디야|투썸플레이스|엔젤리너스',
            r'파리바게뜨|뚜레쥬르|던킨도너츠',
            r'GS25|CU|세븐일레븐|이마트24',
            r'BBQ|치킨플러스|굽네치킨|교촌치킨|네네치킨|호식이두마리치킨',
            
            # 한국어 고유명사 패턴 (일반적)
            r'[가-힣]{2,6}(?=\s|은|는|의|에서|을|를|이|가|도|와|과|에|로|으로)',
            
            # 숫자+한글 조합
            r'[0-9]+[가-힣]{1,5}',
            r'[가-힣]{1,5}[0-9]+',
        ]
        
        # 업종/카테고리 패턴 (포괄적)
        self.category_patterns = [
            # 외식업
            r'(?:치킨|chicken|닭|튀김)',
            r'(?:피자|pizza)',
            r'(?:버거|햄버거|burger)',
            r'(?:카페|cafe|커피|coffee|베이커리)',
            r'(?:한식|중식|일식|양식|분식)',
            r'(?:족발|보쌈|돼지고기)',
            r'(?:곱창|막창|내장)',
            r'(?:마라탕|훠궈|중국요리)',
            r'(?:떡볶이|순대|튀김|분식)',
            r'(?:삼겹살|고기|구이|bbq)',
            r'(?:도시락|김밥|간편식)',
            r'(?:아이스크림|디저트|빙수)',
            
            # 서비스업
            r'(?:미용|헤어|hair|beauty)',
            r'(?:네일|nail|메이크업)',
            r'(?:마사지|스파|힐링|웰빙)',
            r'(?:치료|의료|health)',
            r'(?:펜션|숙박|모텔)',
            r'(?:헬스|피트니스|운동)',
            r'(?:학원|교육|키즈)',
            r'(?:세탁|클리닝)',
            r'(?:주유|자동차|정비)',
            
            # 소매업
            r'(?:편의점|마트|리테일)',
            r'(?:의류|패션|옷)',
            r'(?:휴대폰|통신|핸드폰)',
            r'(?:안경|렌즈)',
            r'(?:문구|사무용품)',
            r'(?:꽃|화훼)',
            
            # 기타
            r'(?:부동산|중개)',
            r'(?:여행|관광)',
            r'(?:펫|애완동물)',
        ]
        
        # 업종 키워드 캐시
        self._category_cache = set()
        self._brand_cache = set()
    
    def extract_from_text(self, text: str) -> Tuple[Set[str], Set[str]]:
        """텍스트에서 브랜드명과 카테고리 추출"""
        if not text:
            return set(), set()
        
        text_lower = text.lower().strip()
        brands = set()
        categories = set()
        
        # 브랜드명 추출
        for pattern in self.brand_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and len(match.strip()) >= 2:  # 최소 2글자
                    brand = match.strip()
                    if not self._is_common_word(brand):
                        brands.add(brand)
                        self._brand_cache.add(brand)
        
        # 카테고리 추출
        for pattern in self.category_patterns:
            if re.search(pattern, text_lower):
                # 패턴에서 실제 키워드 추출
                match = re.search(pattern, text_lower)
                if match:
                    category = match.group().replace('(?:', '').replace(')', '')
                    categories.add(category)
                    self._category_cache.add(category)
        
        return brands, categories
    
    def extract_from_documents(self, docs: List[Document]) -> Dict[str, Set[str]]:
        """문서들의 메타데이터에서 실제 브랜드/카테고리 정보 추출"""
        doc_brands = set()
        doc_categories = set()
        doc_headquarters = set()
        
        for doc in docs:
            metadata = doc.metadata or {}
            
            # 메타데이터에서 브랜드 정보 추출
            brand_name = metadata.get("brand_name", "")
            headquarters_name = metadata.get("headquarters_name", "")
            industry_large = metadata.get("industry_large", "")
            industry_medium = metadata.get("industry_medium", "")
            industry_small = metadata.get("industry_small", "")
            
            # 브랜드명 추가
            if brand_name:
                doc_brands.add(brand_name.strip())
            if headquarters_name:
                doc_headquarters.add(headquarters_name.strip())
            
            # 업종 정보 추가
            for industry in [industry_large, industry_medium, industry_small]:
                if industry:
                    doc_categories.add(industry.strip())
            
            # 문서 내용에서도 추출
            content_brands, content_categories = self.extract_from_text(doc.page_content)
            doc_brands.update(content_brands)
            doc_categories.update(content_categories)
        
        return {
            "brands": doc_brands,
            "categories": doc_categories,
            "headquarters": doc_headquarters
        }
    
    def _is_common_word(self, word: str) -> bool:
        """일반적인 단어인지 확인 (브랜드명이 아닌)"""
        common_words = {
            "그것", "이것", "저것", "그거", "이거", "저거",
            "치킨", "피자", "버거", "카페", "커피", "음식", "요리",
            "창업", "가맹", "비용", "정보", "문의", "상담",
            "프랜차이즈", "브랜드", "업체", "회사",
            "어디", "언제", "무엇", "어떻게", "왜", "누구",
            "안녕", "감사", "죄송", "미안", "괜찮"
        }
        return word.lower() in common_words

# --------------------
# 개선된 컨텍스트 추출기
# --------------------
class EnhancedContextExtractor:
    """대화에서 브랜드/업종 컨텍스트 추출 (동적)"""
    
    def __init__(self):
        self.brand_extractor = DynamicBrandExtractor()
    
    def extract_brand_context(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """최근 대화에서 브랜드/업종 정보 동적 추출"""
        context = {
            "mentioned_brands": set(),
            "mentioned_categories": set(),
            "mentioned_headquarters": set(),
            "last_brand": None,
            "last_category": None,
            "last_headquarters": None,
            "conversation_focus": None  # 대화의 주요 초점
        }
        
        if not messages:
            return context
        
        # 최근 8개 메시지 분석 (더 많은 컨텍스트)
        recent_messages = messages[-8:] if len(messages) > 8 else messages
        
        for msg in reversed(recent_messages):  # 최신부터 확인
            brands, categories = self.brand_extractor.extract_from_text(msg.content)
            
            # 브랜드 정보 업데이트
            context["mentioned_brands"].update(brands)
            if brands and not context["last_brand"]:
                context["last_brand"] = list(brands)[0]
            
            # 카테고리 정보 업데이트
            context["mentioned_categories"].update(categories)
            if categories and not context["last_category"]:
                context["last_category"] = list(categories)[0]
        
        # 대화 초점 결정
        if context["last_brand"]:
            context["conversation_focus"] = f"brand:{context['last_brand']}"
        elif context["last_category"]:
            context["conversation_focus"] = f"category:{context['last_category']}"
        
        # Set을 list로 변환 (JSON 직렬화를 위해)
        context["mentioned_brands"] = list(context["mentioned_brands"])
        context["mentioned_categories"] = list(context["mentioned_categories"])
        context["mentioned_headquarters"] = list(context["mentioned_headquarters"])
        
        return context

# --------------------
# 지능적 문서 필터링 (완전히 개선됨)
# --------------------
def intelligent_document_filter(docs: List[Document], context: Dict[str, Any], 
                               question: str, brand_extractor: DynamicBrandExtractor) -> List[Document]:
    """컨텍스트와 메타데이터를 활용한 지능적 문서 필터링"""
    
    if not docs:
        return docs
    
    print(f"=== 지능적 문서 필터링 시작 ===")
    print(f"입력 문서 수: {len(docs)}")
    print(f"대화 컨텍스트: {context}")
    
    # 1. 문서에서 실제 브랜드/카테고리 정보 추출
    doc_info = brand_extractor.extract_from_documents(docs)
    print(f"문서에서 추출된 브랜드: {list(doc_info['brands'])[:10]}")  # 처음 10개만
    print(f"문서에서 추출된 카테고리: {list(doc_info['categories'])[:10]}")
    
    # 2. 질문에서 브랜드/카테고리 추출
    question_brands, question_categories = brand_extractor.extract_from_text(question)
    print(f"질문에서 추출된 브랜드: {list(question_brands)}")
    print(f"질문에서 추출된 카테고리: {list(question_categories)}")
    
    # 3. 점수 기반 문서 평가
    scored_docs = []
    
    for doc in docs:
        base_score = _get_qdrant_score(doc)
        final_score = base_score
        score_details = {"base": base_score}
        
        metadata = doc.metadata or {}
        content = doc.page_content.lower() if doc.page_content else ""
        
        doc_brand = str(metadata.get("brand_name", "")).lower()
        doc_hq = str(metadata.get("headquarters_name", "")).lower()
        doc_industry_medium = str(metadata.get("industry_medium", "")).lower()
        doc_industry_small = str(metadata.get("industry_small", "")).lower()
        
        # 4. 다양한 매칭 보너스 적용
        
        # 4-1. 정확한 브랜드 매칭 (최고 우선순위)
        last_brand = context.get("last_brand", "")
        if last_brand:
            last_brand_lower = last_brand.lower()
            if (last_brand_lower == doc_brand or 
                last_brand_lower in doc_brand or 
                last_brand_lower in doc_hq or
                any(last_brand_lower in brand.lower() for brand in question_brands)):
                final_score += 0.25
                score_details["exact_brand_match"] = 0.25
        
        # 4-2. 질문 내 브랜드 매칭
        for q_brand in question_brands:
            q_brand_lower = q_brand.lower()
            if (q_brand_lower == doc_brand or 
                q_brand_lower in doc_brand or 
                q_brand_lower in doc_hq):
                final_score += 0.20
                score_details["question_brand_match"] = 0.20
                break
        
        # 4-3. 카테고리 매칭
        last_category = context.get("last_category", "")
        if last_category:
            last_cat_lower = last_category.lower()
            if (last_cat_lower in doc_industry_medium or 
                last_cat_lower in doc_industry_small or
                last_cat_lower in content):
                final_score += 0.15
                score_details["category_match"] = 0.15
        
        # 4-4. 질문 카테고리 매칭
        for q_category in question_categories:
            q_cat_lower = q_category.lower()
            if (q_cat_lower in doc_industry_medium or 
                q_cat_lower in doc_industry_small or
                q_cat_lower in content):
                final_score += 0.12
                score_details["question_category_match"] = 0.12
                break
        
        # 4-5. 키워드 매칭 (창업, 비용 관련)
        business_keywords = ["창업비용", "가맹비", "초기비용", "투자금", "보증금", 
                           "유의사항", "조건", "제한", "자격", "요구사항"]
        question_lower = question.lower()
        
        for keyword in business_keywords:
            if keyword in question_lower and keyword in content:
                final_score += 0.08
                score_details["keyword_match"] = score_details.get("keyword_match", 0) + 0.08
        
        # 4-6. 문서 품질 보너스 (긴 컨텐츠, 메타데이터 완성도)
        if len(content) > 200:  # 충분한 내용
            final_score += 0.05
            score_details["content_quality"] = 0.05
        
        if all(metadata.get(key) for key in ["brand_name", "industry_medium"]):
            final_score += 0.03
            score_details["metadata_quality"] = 0.03
        
        scored_docs.append((doc, final_score, score_details))
    
    # 5. 점수순으로 정렬
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 6. 임계값 적용 (동적 조정)
    min_threshold = MIN_SCORE
    
    # 컨텍스트가 있으면 임계값 완화
    if context.get("last_brand") or context.get("last_category"):
        min_threshold -= 0.08
        print(f"컨텍스트 존재로 임계값 완화: {min_threshold}")
    
    # 질문에 브랜드/카테고리가 있으면 추가 완화
    if question_brands or question_categories:
        min_threshold -= 0.05
        print(f"질문 내 브랜드/카테고리 존재로 추가 완화: {min_threshold}")
    
    filtered_docs = [(doc, score, details) for doc, score, details in scored_docs if score >= min_threshold]
    
    # 7. 최소 개수 보장
    if len(filtered_docs) < 4 and len(scored_docs) >= 4:
        filtered_docs = scored_docs[:4]
        print("최소 문서 개수 보장을 위해 상위 4개 선택")
    
    final_docs = [doc for doc, _, _ in filtered_docs[:FINAL_TOPK]]
    
    # 8. 필터링 결과 로깅
    print(f"=== 필터링 결과 ===")
    print(f"임계값: {min_threshold}")
    print(f"필터링 후 문서 수: {len(final_docs)}")
    
    for i, (doc, score, details) in enumerate(filtered_docs[:FINAL_TOPK]):
        brand = doc.metadata.get("brand_name", "N/A")
        category = doc.metadata.get("industry_medium", "N/A")
        print(f"문서 {i+1}: {brand} ({category}) | 최종점수: {score:.3f} | 상세: {details}")
    
    return final_docs

# --------------------
# Utilities
# --------------------
def concat_context(docs: List[Document], max_chars: int = 6000, sep: str = "\n---\n"):
    """문서들을 컨텍스트로 결합 (길이 제한 증가)"""
    out, n = [], 0
    for d in docs:
        t = (d.page_content or "").strip()
        if n + len(t) > max_chars:
            t = t[: max(0, max_chars - n)]
        out.append(t)
        n += len(t)
        if n >= max_chars:
            break
    return sep.join(out)

def serialize_doc(d: Document) -> dict:
    return {"page_content": d.page_content or "", "metadata": d.metadata or {}}

def _get_qdrant_score(d: Document) -> float:
    md = d.metadata or {}
    for k in ("_qdrant_score", "score", "similarity"):
        if k in md:
            try: 
                return float(md[k])
            except: 
                pass
    return float(getattr(d, "score", 0.0))

# --------------------
# 개선된 Prompts
# --------------------
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
당신은 프랜차이즈 전문 질의응답 보조자입니다. 아래 컨텍스트만을 사용하여 질문에 정확하게 답하세요.

**답변 규칙:**
1. 컨텍스트에 정보가 있으면 그 정보만을 사용하여 답변
2. 정보가 없거나 불충분하면 "해당 정보를 찾을 수 없습니다"라고 답변
3. 한국어 존댓말로 간결하고 정확하게 답변
4. 구체적인 수치나 조건이 있으면 명시
5. 답변 형식: "[주체]는/은 [구체적 답변]입니다."
6. 브랜드명이 명확하면 답변에 포함

# 질문: {question}

# 컨텍스트:
{context}

# 답변:
""".strip())

CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """주어진 대화 기록과 최신 질문을 바탕으로 완전한 독립형 질문으로 재구성하세요.

**재구성 규칙:**
1. 대화에서 언급된 특정 브랜드나 업종을 질문에 포함
2. "그것", "그거", "이것" 같은 대명사를 구체적인 명사로 교체
3. 이전 맥락의 핵심 정보를 질문에 통합
4. 질문 외에 다른 설명은 추가하지 않음
5. 브랜드명은 정확하게 유지

예시:
- 이전 대화: "써브웨이에 대해 알고 싶어요"
- 현재 질문: "창업비용은?"
- 재구성된 질문: "써브웨이 창업비용은?"

- 이전 대화: "치킨 프랜차이즈 알아봐요"
- 현재 질문: "어떤 브랜드가 좋을까요?"
- 재구성된 질문: "치킨 프랜차이즈 어떤 브랜드가 좋을까요?"
"""),
        MessagesPlaceholder("chat_history"),
        ("user", "{question}"),
    ]
)

# --------------------
# LangGraph State & Definition
# --------------------
class GraphState(TypedDict):
    question: str
    answer: str
    docs: List[Document]
    used_docs: List[Dict]
    chat_history: List[BaseMessage]

# --------------------
# 개선된 Memory-Aware RAG Chain (동적 브랜드 인식)
# --------------------
class EnhancedMemoryRAGChain:
    def __init__(self, retriever, llm, get_session_history):
        self.retriever = retriever
        self.llm = llm
        self.get_session_history = get_session_history
        self.contextualize_q_chain = CONTEXTUALIZE_QUESTION_PROMPT | llm | StrOutputParser()
        self.context_extractor = EnhancedContextExtractor()
        self.brand_extractor = DynamicBrandExtractor()

    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """동적 브랜드 인식을 포함한 개선된 RAG 처리"""
        session_id = config.get("configurable", {}).get("session_id")
        if not session_id:
            raise ValueError("session_id가 config에 없습니다.")
        
        # 세션 히스토리 가져오기
        history = self.get_session_history(session_id)
        chat_history = history.messages
        
        question = input_data["question"]
        print(f"=== 동적 브랜드 인식 RAG 처리 시작 ===")
        print(f"원본 질문: {question}")
        print(f"기존 대화 기록: {len(chat_history)}개 메시지")
        
        # 1. 질문에서 브랜드/카테고리 추출
        question_brands, question_categories = self.brand_extractor.extract_from_text(question)
        print(f"질문에서 추출된 브랜드: {list(question_brands)}")
        print(f"질문에서 추출된 카테고리: {list(question_categories)}")
        
        # 2. 대화 기록에서 컨텍스트 추출
        context_info = self.context_extractor.extract_brand_context(chat_history)
        print(f"대화 컨텍스트: {context_info}")
        
        # 3. 질문 맥락화
        if chat_history:
            try:
                contextualized_question = self.contextualize_q_chain.invoke(
                    {"question": question, "chat_history": chat_history}
                )
                print(f"맥락화된 질문: {contextualized_question}")
            except Exception as e:
                print(f"질문 맥락화 실패: {e}, 원본 질문 사용")
                contextualized_question = question
        else:
            contextualized_question = question
            print("대화 기록 없음, 원본 질문 사용")
        
        # 4. 다중 검색 전략
        search_queries = self._generate_search_queries(
            question, contextualized_question, context_info, question_brands, question_categories
        )
        
        all_docs = []
        for search_query in search_queries:
            try:
                docs = self.retriever.invoke(search_query)
                all_docs.extend(docs)
                print(f"검색어 '{search_query}': {len(docs)}개 문서")
            except Exception as e:
                print(f"검색 실패 '{search_query}': {e}")
        
        # 중복 제거 (content 기준)
        seen_contents = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100] if doc.page_content else "")
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        print(f"중복 제거 후 문서 수: {len(unique_docs)}")
        
        # 5. 지능적 문서 필터링
        final_docs = intelligent_document_filter(
            unique_docs, context_info, question, self.brand_extractor
        )
        
        # 6. 답변 생성
        if not final_docs:
            answer = "요청하신 정보를 찾을 수 없습니다. 더 구체적인 브랜드명이나 업종을 포함하여 질문해주시면 도움을 드릴 수 있습니다."
            used_docs = []
        else:
            # 컨텍스트 생성
            ctx = concat_context(final_docs)
            
            # RAG 체인 실행
            rag_chain = (
                RunnablePassthrough.assign(context=lambda x: ctx)
                | ANSWER_PROMPT
                | self.llm
                | StrOutputParser()
            )
            
            try:
                answer = rag_chain.invoke({"question": question})
                used_docs = [serialize_doc(d) for d in final_docs]
            except Exception as e:
                print(f"답변 생성 실패: {e}")
                answer = "답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요."
                used_docs = []
        
        print(f"생성된 답변: {answer}")
        
        # 7. 대화 기록에 저장
        try:
            history.add_message(HumanMessage(content=question))
            history.add_message(AIMessage(content=answer))
        except Exception as e:
            print(f"대화 기록 저장 실패: {e}")
        
        return {
            "question": question,
            "answer": answer,
            "used_docs": used_docs,
            "context_info": context_info,
            "search_queries": search_queries
        }
    
    def _generate_search_queries(self, original_question: str, contextualized_question: str, 
                               context_info: Dict[str, Any], question_brands: Set[str], 
                               question_categories: Set[str]) -> List[str]:
        """다양한 검색 쿼리 생성"""
        queries = []
        
        # 1. 맥락화된 질문 (가장 중요)
        if contextualized_question and contextualized_question != original_question:
            queries.append(contextualized_question)
        
        # 2. 원본 질문
        queries.append(original_question)
        
        # 3. 질문의 브랜드 + 대화 컨텍스트 조합
        last_brand = context_info.get("last_brand")
        if last_brand and not any(last_brand.lower() in q.lower() for q in queries):
            queries.append(f"{last_brand} {original_question}")
        
        # 4. 질문에서 추출된 브랜드로 확장
        for brand in question_brands:
            brand_query = f"{brand} {original_question}"
            if brand_query not in queries:
                queries.append(brand_query)
        
        # 5. 카테고리 기반 검색
        last_category = context_info.get("last_category")
        if last_category and not any(last_category.lower() in q.lower() for q in queries):
            queries.append(f"{last_category} {original_question}")
        
        # 6. 핵심 키워드 추출하여 검색
        keywords = self._extract_key_terms(original_question)
        if keywords:
            keyword_query = " ".join(keywords)
            if keyword_query not in queries and len(keyword_query.strip()) > 3:
                queries.append(keyword_query)
        
        # 최대 5개로 제한 (성능 고려)
        return queries[:5]
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """질문에서 핵심 키워드 추출"""
        # 중요한 키워드들
        important_terms = [
            "창업비용", "가맹비", "초기비용", "투자금", "보증금",
            "유의사항", "조건", "제한", "자격", "요구사항",
            "수익", "매출", "순익", "로열티", "수수료",
            "교육", "지원", "혜택", "서비스",
            "위치", "입지", "면적", "크기", "규모",
            "경쟁", "차별화", "특징", "장단점"
        ]
        
        found_terms = []
        question_lower = question.lower()
        
        for term in important_terms:
            if term in question_lower:
                found_terms.append(term)
        
        return found_terms

# --------------------
# FastAPI App (동적 브랜드 인식 적용)
# --------------------
class QuestionRequest(BaseModel):
    question: str
    session_id: str = Field(..., description="각 사용자의 대화를 구분하는 고유 ID")

class AnswerResponse(BaseModel):
    question: str
    answer: str
    used_docs: List[Dict[str, Any]]
    context_info: Dict[str, Any] = Field(default_factory=dict)
    search_queries: List[str] = Field(default_factory=list)
    status: str
    cfg: str

app = FastAPI(title="프랜차이즈 QA API (동적 브랜드 인식)", version="5.0.0")

# PostgreSQL 연결 정보
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """PostgreSQL 기반 세션 히스토리 반환"""
    return CustomPostgresChatMessageHistory(
        session_id=session_id,
        connection_string=DATABASE_URL,
        table_name="message_store"
    )

@app.on_event("startup")
def on_startup():
    # DB 연결 테스트
    try:
        test_history = get_session_history("test_connection")
        print("PostgreSQL 연결 성공!")
    except Exception as e:
        print(f"PostgreSQL 연결 실패: {e}")
        raise
    
    embeddings = build_embeddings()
    retriever, collection = build_qdrant_vectorstore(embeddings)
    llm = build_llm()
    
    # 동적 브랜드 인식 RAG 체인 생성
    app.state.enhanced_rag_chain = EnhancedMemoryRAGChain(retriever, llm, get_session_history)
    app.state.collection = collection
    
    print(f"=== {CFG_NAME} 서버 시작 완료 ===")
    print(f"동적 브랜드 인식 활성화")
    print(f"검색량: {K_RETRIEVE}, 최종 선택: {FINAL_TOPK}")
    print(f"최소 점수 임계값: {MIN_SCORE}")

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id가 비어 있습니다.")
    
    try:
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.session_id.strip()))
        config = {"configurable": {"session_id": session_uuid}}
        
        print(f"=== 새로운 질문 처리 ===")
        print(f"세션 ID: {session_uuid}")
        print(f"질문: {req.question}")
        
        result = app.state.enhanced_rag_chain.invoke(
            {"question": req.question.strip()},
            config=config
        )
        
        return AnswerResponse(
            question=req.question.strip(),
            answer=result["answer"],
            used_docs=result["used_docs"],
            context_info=result.get("context_info", {}),
            search_queries=result.get("search_queries", []),
            status="success",
            cfg=CFG_NAME
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 건강 체크 및 DB 상태 확인 엔드포인트
@app.get("/health")
def health_check():
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy", 
        "config": CFG_NAME,
        "database": db_status,
        "features": {
            "dynamic_brand_recognition": True,
            "intelligent_filtering": True,
            "multi_query_search": True,
            "context_extraction": True
        }
    }

# 브랜드 추출 테스트 엔드포인트
@app.post("/debug/extract-brands")
def debug_extract_brands(request: dict):
    """텍스트에서 브랜드/카테고리 추출 테스트"""
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text 필드가 필요합니다.")
    
    extractor = DynamicBrandExtractor()
    brands, categories = extractor.extract_from_text(text)
    
    return {
        "input_text": text,
        "extracted_brands": list(brands),
        "extracted_categories": list(categories),
        "brand_count": len(brands),
        "category_count": len(categories)
    }

# 세션 기록 조회 엔드포인트 (강화됨)
@app.get("/debug/history/{session_id}")
def get_debug_history(session_id: str):
    try:
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id.strip()))
        history = get_session_history(session_uuid)
        messages = history.messages
        
        # 컨텍스트 추출
        context_extractor = EnhancedContextExtractor()
        context_info = context_extractor.extract_brand_context(messages)
        
        # 최근 메시지들에서 브랜드 추출
        brand_extractor = DynamicBrandExtractor()
        recent_brands = set()
        recent_categories = set()
        
        for msg in messages[-5:]:  # 최근 5개 메시지
            brands, categories = brand_extractor.extract_from_text(msg.content)
            recent_brands.update(brands)
            recent_categories.update(categories)
        
        return {
            "session_id": session_id,
            "session_uuid": session_uuid,
            "message_count": len(messages),
            "context_info": context_info,
            "recent_extracted": {
                "brands": list(recent_brands),
                "categories": list(recent_categories)
            },
            "messages": [
                {
                    "type": type(msg).__name__, 
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', None)
                } 
                for msg in messages[-10:]  # 최근 10개만
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 세션 기록 삭제 엔드포인트
@app.delete("/debug/history/{session_id}")
def clear_debug_history(session_id: str):
    try:
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id.strip()))
        history = get_session_history(session_uuid)
        history.clear()
        return {
            "message": "세션 기록이 삭제되었습니다.", 
            "session_id": session_id,
            "session_uuid": session_uuid
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 실시간 브랜드 학습 엔드포인트
@app.post("/debug/learn-from-docs")
def learn_brands_from_docs():
    """현재 문서들에서 브랜드/카테고리 학습"""
    try:
        # 샘플 검색으로 문서들 가져오기
        embeddings = build_embeddings()
        retriever, _ = build_qdrant_vectorstore(embeddings)
        
        sample_docs = retriever.invoke("프랜차이즈")  # 샘플 검색
        
        extractor = DynamicBrandExtractor()
        learned_info = extractor.extract_from_documents(sample_docs)
        
        return {
            "status": "success",
            "sample_doc_count": len(sample_docs),
            "learned_brands": list(learned_info["brands"])[:20],  # 처음 20개만
            "learned_categories": list(learned_info["categories"])[:20],
            "learned_headquarters": list(learned_info["headquarters"])[:20],
            "total_brands": len(learned_info["brands"]),
            "total_categories": len(learned_info["categories"]),
            "total_headquarters": len(learned_info["headquarters"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)