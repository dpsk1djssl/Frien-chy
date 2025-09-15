# -*- coding: utf-8 -*-
# main_inference_final.py
# LangGraph + PostgreSQL Memory 개선

import os
import uuid
from typing import List, Dict, Any, TypedDict, Optional, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import json
import langchain
from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.cache import SQLiteCache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient

# PostgreSQL 관련 import
import psycopg
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Qdrant 관련 import
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType

# --------------------
# Config
# --------------------
CFG_NAME = "LangGraph_With_PostgreSQL_v4.2_Improved"
EMBED_MODEL = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RETRIEVE = 15  # 검색량 증가
FINAL_TOPK = 7   # 최종 선택량 증가
MIN_SCORE = float(os.getenv("QDRANT_MIN_SCORE", "0.45"))  # 임계값 조정

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
        timeout=60.0,
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
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

def build_llm_light():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

def build_llm_report():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
    )

# --------------------
# Custom PostgreSQL Chat Message History (개선됨)
# --------------------
class CustomPostgresChatMessageHistory(BaseChatMessageHistory):
    """PostgreSQL 기반의 커스텀 채팅 메시지 히스토리 (개선됨)"""
    
    def __init__(self, session_id: str, connection_string: str, table_name: str = "message_store"):
        self.session_id = session_id
        self.connection_string = connection_string
        self.table_name = table_name
        self._create_table_if_not_exists()
        self._messages_cache = None  # 캐싱 추가

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
# Qdrant Payload Index 설정
# --------------------
def setup_payload_indexes(client: QdrantClient, collection_name: str):
    """
    자주 사용하는 메타데이터 필드에 대한 payload index 생성
    """
    try:
        # 브랜드명 인덱스 (정확히 매칭되는 필터링에 사용)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.brand_name",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("brand_name 인덱스 생성 완료")
        
        # 업종(중분류) 인덱스 (카테고리 매칭에 사용)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.industry_medium",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("industry_medium 인덱스 생성 완료")
        
        # 섹션명 인덱스 (섹션별 정리에 사용)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.section_name",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("section_name 인덱스 생성 완료")
        
        # 업종(대분류) 인덱스 (추가적인 카테고리 필터링에 사용 가능)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.industry_large",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("industry_large 인덱스 생성 완료")
        
        # 연도 인덱스 (데이터 기준연도 필터링에 사용 가능)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.year",
            field_schema=PayloadSchemaType.INTEGER
        )
        print("year 인덱스 생성 완료")
        
    except Exception as e:
        # 이미 인덱스가 존재하거나 기타 오류 시 계속 진행
        print(f"⚠️ Payload 인덱스 설정 중 일부 오류 발생 (무시하고 계속): {e}")

# --------------------
# 컨텍스트 추출기 (새로 추가)
# --------------------
class ContextExtractor:
    """대화에서 브랜드/업종 컨텍스트 추출"""
    
    @staticmethod
    def extract_brand_context(messages: List[BaseMessage]) -> Dict[str, Any]:
        """최근 대화에서 브랜드/업종 정보 추출"""
        context = {
            "mentioned_brands": set(),
            "mentioned_categories": set(),
            "last_brand": None,
            "last_category": None
        }
        
        # 최근 5개 메시지만 확인 (너무 오래된 것은 제외)
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        brand_keywords = ["치킨", "피자", "버거", "카페", "네네", "교촌", "굽네", "BBQ"]
        category_keywords = ["외식", "치킨", "피자", "햄버거", "카페", "베이커리"]
        
        for msg in reversed(recent_messages):  # 최신부터 확인
            content = msg.content.lower()
            
            # 브랜드 키워드 찾기
            for brand in brand_keywords:
                if brand.lower() in content:
                    context["mentioned_brands"].add(brand)
                    if not context["last_brand"]:
                        context["last_brand"] = brand
            
            # 카테고리 키워드 찾기
            for category in category_keywords:
                if category.lower() in content:
                    context["mentioned_categories"].add(category)
                    if not context["last_category"]:
                        context["last_category"] = category
        
        return context



# --------------------
# 개선된 문서 필터링
# --------------------
def smart_document_filter(docs: List[Document], context: Dict[str, Any], 
                         question: str) -> List[Document]:
    """컨텍스트를 고려한 스마트 문서 필터링"""
    
    if not docs:
        return docs
    
    # 1. 점수 기반 기본 필터링
    scored_docs = [(d, _get_qdrant_score(d)) for d in docs]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 2. 컨텍스트 기반 가중치 적용
    weighted_docs = []
    question_lower = question.lower()
    
    for doc, score in scored_docs:
        content = doc.page_content.lower()
        metadata = doc.metadata or {}
        
        # 기본 점수
        final_score = score
        
        # 브랜드 매칭 보너스
        if context.get("last_brand"):
            brand = context["last_brand"].lower()
            if brand in content or brand in str(metadata.get("brand_name", "")).lower():
                final_score += 0.15
        
        # 카테고리 매칭 보너스
        if context.get("last_category"):
            category = context["last_category"].lower()
            if category in content or category in str(metadata.get("industry_medium", "")).lower():
                final_score += 0.1
        
        # 질문 키워드 매칭 보너스
        question_keywords = ["창업비용", "가격", "유의할점", "조건", "제한", "부담"]
        for keyword in question_keywords:
            if keyword in question_lower and keyword in content:
                final_score += 0.05
        
        weighted_docs.append((doc, final_score))
    
    # 3. 최종 점수로 재정렬 및 필터링
    weighted_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 임계값 적용 (동적 조정)
    min_threshold = MIN_SCORE
    if context.get("last_brand") or context.get("last_category"):
        min_threshold -= 0.05  # 컨텍스트가 있으면 임계값 완화
    
    filtered_docs = [doc for doc, score in weighted_docs if score >= min_threshold]
    
    # 최소 3개는 보장
    if len(filtered_docs) < 3 and len(weighted_docs) >= 3:
        filtered_docs = [doc for doc, _ in weighted_docs[:3]]
    
    return filtered_docs[:FINAL_TOPK]

# --------------------
# Utilities (개선됨)
# --------------------
def concat_context(docs: List[Document], max_chars: int = 5000, sep: str = "\n---\n"):
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
당신은 프랜차이즈 창업의 든든한 동반자, '프랜치'입니다.
당신의 임무는 아래 # 찾아온 정보를 바탕으로, 사용자의 궁금증을 시원하게 해결해주는 것입니다.

예비 창업자와 기존 점주님 모두가 쉽게 이해할 수 있도록, 핵심만 쏙쏙 뽑아 명확하게 설명해주세요.

**답변 작성 가이드:**
- **정확한 정보만!:** # 찾아온 정보에 있는 내용만을 근거로 답변해야 합니다. 정보에 없는 내용은 '아직 확인되지 않은 정보'라고 솔직하게 말해주세요.
- **맞춤형 답변!:**
    - 정보가 충분하다면, 질문 전체에 대해 명쾌한 해결책을 제시해주세요.
    - 정보가 부족하다면, 아는 부분까지만이라도 "우선 ~에 대해 먼저 설명해 드릴게요."라며 친절하게 설명해주세요.
    - 정보가 전혀 관련 없다면, "죄송하지만, 문의하신 내용과 관련된 정보는 찾지 못했습니다."라고 정중하게 말씀해주세요.
    - 찾아온 정보의 기준연도 명시해주세요.
    - 별첨 자료가 말로만 있고 실제로 없다면 무시해주세요.
- **친절한 조언!:** 단순히 정보를 나열하지 말고, 전문가로서 "이런 점을 특히 유의하시면 좋습니다." 와 같이 실질적인 조언을 덧붙여주세요.

# 질문: {question}

# 찾아온 정보:
{context}

# 프랜치의 답변:
""".strip())

CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """주어진 대화 기록과 최신 질문을 바탕으로 완전한 독립형 질문으로 재구성하세요.

**재구성 규칙:**
1. 대화에서 언급된 특정 브랜드나 업종을 질문에 포함
2. "그것", "그거", "이것" 같은 대명사를 구체적인 명사로 교체
3. 이전 맥락의 핵심 정보를 질문에 통합
4. 질문 외에 다른 설명은 추가하지 않음

예시:
- 이전 대화: "네네치킨에 대해 알고 싶어요"
- 현재 질문: "창업비용은?"
- 재구성된 질문: "네네치킨 창업비용은?"
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
# 개선된 Memory-Aware RAG Chain
# --------------------
class MemoryAwareRAGChain:
    def __init__(self, retriever, llm_light, llm, get_session_history):
        self.retriever = retriever
        self.llm = llm
        self.get_session_history = get_session_history
        self.contextualize_q_chain = CONTEXTUALIZE_QUESTION_PROMPT | llm_light | StrOutputParser()
        self.context_extractor = ContextExtractor()
        # self.synonym_normalizer = SynonymNormalizer()  # 🚀 성능 최적화: 완전 제거

    def _get_recent_user_questions(self, chat_history: List[BaseMessage], limit: int = 5) -> List[BaseMessage]:
        """최근 사용자 질문만 추출 (AI 답변 제외)"""
        user_questions = [msg for msg in chat_history if isinstance(msg, HumanMessage)]
        return user_questions[-limit:] if len(user_questions) > limit else user_questions

    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """메모리 기능을 포함한 개선된 RAG 처리"""
        session_id = config.get("configurable", {}).get("session_id")
        if not session_id:
            raise ValueError("session_id가 config에 없습니다.")
        
        # 세션 히스토리 가져오기
        history = self.get_session_history(session_id)
        chat_history = history.messages
        
        question = input_data["question"]
        print(f"=== RAG 처리 시작 ===")
        print(f"원본 질문: {question}")
        print(f"기존 대화 기록: {len(chat_history)}개 메시지")
        
        # 컨텍스트 추출
        context_info = self.context_extractor.extract_brand_context(chat_history)
        print(f"추출된 컨텍스트: {context_info}")
        
        # 대화 기록이 있으면 질문을 맥락화 (최근 사용자 질문 5개만 사용)
        if chat_history:
            recent_user_questions = self._get_recent_user_questions(chat_history, 5)
            contextualized_question = self.contextualize_q_chain.invoke(
                {"question": question, "chat_history": recent_user_questions}
            )
            print(f"맥락화된 질문: {contextualized_question}")
            print(f"사용된 사용자 질문 수: {len(recent_user_questions)}")
        else:
            contextualized_question = question
            print("대화 기록 없음, 원본 질문 사용")
        
        # 유사어 기반 재작성(표준화/확장) - 🚀 성능 최적화를 위해 완전 비활성화
        # rewritten_question = self.synonym_normalizer.rewrite(contextualized_question)
        rewritten_question = contextualized_question  # 성능 최적화: 유사어 재작성 생략
        print(f"⚡ 성능 최적화: 유사어 재작성 완전 생략")
        print(f"최종 검색 질문: {rewritten_question}")

        # 문서 검색 (재작성된 질문으로)
        raw_docs = self.retriever.invoke(rewritten_question)
        print(f"검색된 원본 문서 수: {len(raw_docs)}")
        
        # 스마트 필터링 적용
        final_docs = smart_document_filter(raw_docs, context_info, question)
        print(f"필터링 후 최종 문서 수: {len(final_docs)}")
        
        # 사용된 문서의 브랜드 정보 출력 (더 상세하게)
        for i, doc in enumerate(final_docs):
            brand = doc.metadata.get("brand_name", "N/A")
            category = doc.metadata.get("industry_medium", "N/A")
            section = doc.metadata.get("section_name", "N/A")[:50]  # 50자까지만
            score = _get_qdrant_score(doc)
            print(f"문서 {i+1}: {brand} ({category}) | {section}... | 점수: {score:.3f}")
        
        # 답변 생성 (원본 질문으로)
        if not final_docs:
            answer = "요청하신 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주시면 도움을 드릴 수 있습니다."
        else:
            ctx = concat_context(final_docs)
            rag_chain = (
                RunnablePassthrough.assign(context=lambda x: ctx)
                | ANSWER_PROMPT
                | self.llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke({"question": question})
        
        print(f"생성된 답변: {answer}")
        
        # 대화 기록에 저장
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=answer))
        
        return {
            "question": question,
            "answer": answer,
            "used_docs": [serialize_doc(d) for d in final_docs]
        }
    
    # NEW: 스트리밍을 위한 비동기 제너레이터 메소드 추가
    async def astream_invoke(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """메모리 기능을 포함한 개선된 RAG 처리 (비동기 스트리밍)"""
        session_id = config.get("configurable", {}).get("session_id")
        if not session_id:
            raise ValueError("session_id가 config에 없습니다.")

        question = input_data["question"]
        
        print("=== RAG 스트리밍 처리 시작 ===")
        print(f"원본 질문: {question}")

        # 🚀 즉시 스트리밍 시작: 처리 시작 알림
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'step': 'started'}, ensure_ascii=False)}\n\n"

        # 1. 대화 기록 조회
        history = self.get_session_history(session_id)
        chat_history = history.messages
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'step': 'context_loading'}, ensure_ascii=False)}\n\n"

        # 2. 컨텍스트 추출
        context_info = self.context_extractor.extract_brand_context(chat_history)
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'step': 'context_extracted'}, ensure_ascii=False)}\n\n"
        
        # 3. 질문 맥락화
        if chat_history:
            yield f"event: status\ndata: {json.dumps({'status': 'processing', 'step': 'contextualizing_question'}, ensure_ascii=False)}\n\n"
            recent_user_questions = self._get_recent_user_questions(chat_history, 5)
            contextualized_question = self.contextualize_q_chain.invoke(
                {"question": question, "chat_history": recent_user_questions}
            )
        else:
            contextualized_question = question
        
        # 🚀 성능 최적화: 유사어 재작성 완전 생략
        rewritten_question = contextualized_question
        print(f"⚡ 성능 최적화: 유사어 재작성 완전 생략")
        print(f"최종 검색 질문: {rewritten_question}")

        # 4. 문서 검색 시작 알림
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'step': 'searching_documents'}, ensure_ascii=False)}\n\n"
        
        raw_docs = self.retriever.invoke(rewritten_question)
        final_docs = smart_document_filter(raw_docs, context_info, question)
        print(f"최종 문서 수: {len(final_docs)}")
        
        # 5. 검색 완료 및 참고 문서 정보 전송
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'step': 'documents_found', 'count': len(final_docs)}, ensure_ascii=False)}\n\n"
        
        used_docs_json = json.dumps([serialize_doc(d) for d in final_docs], ensure_ascii=False)
        yield f"event: sources\ndata: {used_docs_json}\n\n"

        # 6. 답변 생성 시작 알림
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'step': 'generating_answer'}, ensure_ascii=False)}\n\n"

        # 7. 답변 생성 스트리밍
        if not final_docs:
            answer_chunk = "요청하신 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주시면 도움을 드릴 수 있습니다."
            yield f"data: {json.dumps({'token': answer_chunk}, ensure_ascii=False)}\n\n"
            full_answer = answer_chunk
        else:
            ctx = concat_context(final_docs)
            rag_chain = (
                RunnablePassthrough.assign(context=lambda x: ctx)
                | ANSWER_PROMPT
                | self.llm
                | StrOutputParser()
            )
            
            full_answer = ""
            # .astream()을 사용하여 비동기적으로 토큰을 받아옵니다.
            async for chunk in rag_chain.astream({"question": question}):
                full_answer += chunk
                # 각 토큰(chunk)을 SSE 형식으로 클라이언트에 yield
                yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"

        # 4. 스트리밍 종료 후 대화 기록 저장
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=full_answer))
        print(f"전체 답변 저장 완료: {full_answer}")

        # 5. 스트림의 끝을 알리는 특별 이벤트 전송
        yield "event: end\ndata: Stream ended\n\n"

# --------------------
# FastAPI App (기존과 동일하지만 개선된 체인 사용)
# --------------------
class QuestionRequest(BaseModel):
    question: str
    session_id: str = Field(..., description="각 사용자의 대화를 구분하는 고유 ID")

class AnswerResponse(BaseModel):
    question: str
    answer: str
    used_docs: List[Dict[str, Any]]
    status: str
    cfg: str

app = FastAPI(title="프랜차이즈 QA API (개선된 메모리)", version="4.2.0")

# PostgreSQL 연결 정보
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

# PostgreSQL 연결 정보 (동의어 DB는 제거됨 - 성능 최적화)
# DATABASE_URL_SYN = os.getenv("DATABASE_SYN_URL")  # 🚀 성능 최적화: 제거

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """PostgreSQL 기반 세션 히스토리 반환"""
    return CustomPostgresChatMessageHistory(
        session_id=session_id,
        connection_string=DATABASE_URL,
        table_name="message_store"
    )

# --------------------
# 브랜드 리포트용 Request/Response Models
# --------------------
class BrandReportRequest(BaseModel):
    brand_name: str = Field(..., description="리포트를 생성할 브랜드명")
    session_id: Optional[str] = Field(None, description="세션 ID (선택사항)")

class BrandReportResponse(BaseModel):
    brand_name: str
    report: str
    total_docs: int
    sections_covered: List[str]
    status: str
    cfg: str

# --------------------
# 브랜드 리포트 생성용 Prompt
# --------------------
BRAND_REPORT_PROMPT = ChatPromptTemplate.from_template("""
당신은 프랜차이즈 브랜드 전문 분석가입니다. 
주어진 {brand_name} 브랜드의 모든 정보를 바탕으로 종합적인 브랜드 리포트를 작성하세요.

**리포트 작성 지침:**
1. 체계적이고 구조화된 형식으로 작성
2. 핵심 정보를 섹션별로 명확히 구분
3. 구체적인 수치와 조건을 포함
4. 장단점을 객관적으로 분석
5. 예비 창업자에게 유용한 인사이트 제공

**필수 포함 섹션:**
- 브랜드 개요
- 창업 비용 및 조건
- 운영 시스템 및 지원
- 수익성 분석
- 경쟁력 및 차별화 요소
- 주의사항 및 제약조건
- 종합 평가 및 추천 대상

# 브랜드명: {brand_name}

# 수집된 정보:
{context}

# 종합 리포트:
""".strip())

# --------------------
# Qdrant Scroll을 사용한 문서 수집 함수
# --------------------
def collect_brand_documents(client: QdrantClient, collection_name: str, 
                           brand_name: str, batch_size: int = 100) -> List[Dict]:
    """
    Qdrant scroll을 사용하여 특정 브랜드의 모든 문서를 수집
    """
    all_documents = []
    offset = None
    
    # 브랜드명 필터 생성
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="metadata.brand_name",
                match=MatchValue(value=brand_name)
            )
        ]
    )
    
    while True:
        # Scroll 요청
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False  # 벡터는 필요없음
        )
        
        points, next_offset = result
        
        if not points:
            break
            
        # 문서 정보 추출
        for point in points:
            payload = point.payload
            doc_info = {
                "content": payload.get("page_content", ""),
                "metadata": payload.get("metadata", {}),
                "section": payload.get("metadata", {}).get("section_name", "기타")
            }
            all_documents.append(doc_info)
        
        offset = next_offset
        if offset is None:
            break
    
    return all_documents

# --------------------
# 문서 정리 및 구조화 함수
# --------------------
def organize_documents_by_section(documents: List[Dict]) -> Dict[str, List[str]]:
    """
    문서를 섹션별로 정리
    """
    organized = {}
    
    for doc in documents:
        section = doc["section"]
        content = doc["content"]
        
        if section not in organized:
            organized[section] = []
        
        # 중복 제거를 위한 간단한 체크
        if content not in organized[section]:
            organized[section].append(content)
    
    return organized


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Setting up SQLite cache for LangChain...")
    langchain.llm_cache = SQLiteCache(database_path="langchain.db")
    print("Cache setup complete.")
    
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
    llm_light = build_llm_light()
    llm_report = build_llm_report()
    
    # Qdrant Payload Index 설정
    print("Qdrant Payload Index 설정 중...")
    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60.0,
            prefer_grpc=True,
            https=False
        )
        setup_payload_indexes(client, collection)
        print("Payload Index 설정 완료!")
    except Exception as e:
        print(f"⚠️ Payload Index 설정 실패 (앱은 계속 실행됩니다): {e}")
    
    # 개선된 메모리 인식 RAG 체인 생성
    app.state.memory_rag_chain = MemoryAwareRAGChain(retriever, llm_light, llm, get_session_history)
    app.state.collection = collection
    app.state.llm_report = llm_report
    app.state.qdrant_client = client  # ✨ Qdrant 클라이언트를 app.state에 저장
    
    yield
    
    # Shutdown (필요시 정리 작업)
    print("Application shutdown...")


app = FastAPI(title="프랜차이즈 QA API (개선된 메모리)", version="4.2.0", lifespan=lifespan)

# ✨ CHANGED: /ask 엔드포인트를 비동기 스트리밍 방식으로 변경
@app.post("/ask") # ⛔️ response_model=AnswerResponse 제거
async def ask(req: QuestionRequest): # ✨ async def로 변경
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id가 비어 있습니다.")

    try:
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.session_id.strip()))
        config = {"configurable": {"session_id": session_uuid}}

        # StreamingResponse를 반환합니다.
        # content는 비동기 제너레이터 함수 호출 그 자체입니다.
        return StreamingResponse(
            app.state.memory_rag_chain.astream_invoke(
                {"question": req.question.strip()},
                config=config
            ),
            media_type="text/event-stream" # ✨ SSE를 위한 media type
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 스트리밍 오류는 다른 방식으로 처리해야 할 수 있습니다.
        # 여기서는 간단히 500 에러를 발생시킵니다.
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
        "database": db_status
    }

# 세션 기록 조회 엔드포인트 (디버깅용)
@app.get("/debug/history/{session_id}")
def get_debug_history(session_id: str):
    try:
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id.strip()))
        history = get_session_history(session_uuid)
        messages = history.messages
        context_info = ContextExtractor.extract_brand_context(messages)
        
        return {
            "session_id": session_id,
            "session_uuid": session_uuid,
            "message_count": len(messages),
            "context_info": {
                "mentioned_brands": list(context_info["mentioned_brands"]),
                "mentioned_categories": list(context_info["mentioned_categories"]),
                "last_brand": context_info["last_brand"],
                "last_category": context_info["last_category"]
            },
            "messages": [{"type": type(msg).__name__, "content": msg.content} for msg in messages[-10:]]  # 최근 10개만
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 세션 기록 삭제 엔드포인트 (디버깅용)
@app.delete("/debug/history/{session_id}")
def clear_debug_history(session_id: str):
    try:
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id.strip()))
        history = get_session_history(session_uuid)
        history.clear()
        return {"message": "세션 기록이 삭제되었습니다.", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Qdrant 인덱스 상태 확인 엔드포인트 (디버깅용)
@app.get("/debug/qdrant/indexes")
def check_qdrant_indexes():
    """Qdrant 컬렉션의 payload index 상태 확인"""
    try:
        client = app.state.qdrant_client  # ✨ 재사용된 클라이언트 사용
        collection_name = app.state.collection
        collection_info = client.get_collection(collection_name)
        
        # payload schema 정보에서 인덱스 확인
        payload_schema = collection_info.config.params.vectors.get("default", {}).get("size", 0)
        
        # 컬렉션 상세 정보 가져오기
        return {
            "collection_name": collection_name,
            "collection_status": collection_info.status,
            "points_count": collection_info.points_count,
            "payload_schema": collection_info.config.payload_index if hasattr(collection_info.config, 'payload_index') else "정보 없음",
            "message": "인덱스 정보를 확인하세요. 자주 사용하는 필드에 인덱스가 있는지 확인하시기 바랍니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant 상태 확인 실패: {str(e)}")

# --------------------
# 브랜드 리포트 생성 엔드포인트
# --------------------
@app.post("/brand-report", response_model=BrandReportResponse)
async def generate_brand_report(req: BrandReportRequest):
    """
    특정 브랜드의 모든 정보를 수집하여 종합 리포트 생성
    """
    if not req.brand_name.strip():
        raise HTTPException(status_code=400, detail="브랜드명이 비어 있습니다.")
    
    try:
        brand_name = req.brand_name.strip()
        print(f"=== 브랜드 리포트 생성 시작: {brand_name} ===")
        
        # ✨ app.state에서 재사용된 Qdrant 클라이언트 사용
        client = app.state.qdrant_client
        collection_name = app.state.collection
        
        # 1. 브랜드 관련 모든 문서 수집
        print(f"Qdrant에서 {brand_name} 문서 수집 중...")
        all_docs = collect_brand_documents(client, collection_name, brand_name)
        print(f"수집된 문서 수: {len(all_docs)}")
        
        if not all_docs:
            return BrandReportResponse(
                brand_name=brand_name,
                report=f"{brand_name}에 대한 정보를 찾을 수 없습니다. 브랜드명을 정확히 입력했는지 확인해주세요.",
                total_docs=0,
                sections_covered=[],
                status="no_data",
                cfg=CFG_NAME
            )
        
        # 2. 문서를 섹션별로 정리
        organized_docs = organize_documents_by_section(all_docs)
        sections_covered = list(organized_docs.keys())
        print(f"커버된 섹션: {sections_covered}")
        
        # 3. 컨텍스트 생성 (섹션별로 구조화)
        context_parts = []
        for section, contents in organized_docs.items():
            context_parts.append(f"\n## {section}")
            # 각 섹션당 최대 5개 문서만 포함 (너무 길어지는 것 방지)
            for content in contents[:5]:
                # 각 문서를 500자로 제한
                truncated = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"- {truncated}")
        
        context = "\n".join(context_parts)
        
        # 컨텍스트 길이 제한 (토큰 제한 고려)
        max_context_length = 15000  # 약 3750 토큰
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... (추가 정보 생략)"
        
        # 4. LLM을 사용하여 종합 리포트 생성
        llm_report = app.state.llm_report
        
        report_chain = (
            BRAND_REPORT_PROMPT
            | llm_report
            | StrOutputParser()
        )
        
        report = report_chain.invoke({
            "brand_name": brand_name,
            "context": context
        })
        
        print(f"리포트 생성 완료. 길이: {len(report)}자")
        
        # 5. 세션 히스토리에 저장 (선택사항)
        if req.session_id:
            try:
                session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.session_id.strip()))
                history = get_session_history(session_uuid)
                history.add_message(HumanMessage(content=f"{brand_name} 브랜드 리포트 요청"))
                history.add_message(AIMessage(content=f"[브랜드 리포트 생성 완료 - {len(all_docs)}개 문서 분석]"))
            except Exception as e:
                print(f"세션 저장 실패: {e}")
        
        return BrandReportResponse(
            brand_name=brand_name,
            report=report,
            total_docs=len(all_docs),
            sections_covered=sections_covered,
            status="success",
            cfg=CFG_NAME
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"리포트 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
