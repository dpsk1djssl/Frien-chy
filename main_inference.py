# -*- coding: utf-8 -*-
# main_inference_final.py
# LangGraph + PostgreSQL Memory 개선 버전

import os
import uuid
from typing import List, Dict, Any, TypedDict

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
CFG_NAME = "LangGraph_With_PostgreSQL_v4.2_Improved"
EMBED_MODEL = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RETRIEVE = 15  # 검색량 증가
FINAL_TOPK = 7   # 최종 선택량 증가
MIN_SCORE = float(os.getenv("QDRANT_MIN_SCORE", "0.45"))  # 임계값 하향 조정

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
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
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
당신은 프랜차이즈 전문 질의응답 보조자입니다. 아래 컨텍스트만을 사용하여 질문에 정확하게 답하세요.

**답변 규칙:**
1. 컨텍스트에 정보가 있으면 그 정보만을 사용하여 답변
2. 정보가 없거나 불충분하면 "해당 정보를 찾을 수 없습니다"라고 답변
3. 한국어 존댓말로 간결하고 정확하게 답변
4. 구체적인 수치나 조건이 있으면 명시
5. 답변 형식: "[주체]는/은 [구체적 답변]입니다."

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
    def __init__(self, retriever, llm, get_session_history):
        self.retriever = retriever
        self.llm = llm
        self.get_session_history = get_session_history
        self.contextualize_q_chain = CONTEXTUALIZE_QUESTION_PROMPT | llm | StrOutputParser()
        self.context_extractor = ContextExtractor()

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
        
        # 대화 기록이 있으면 질문을 맥락화
        if chat_history:
            contextualized_question = self.contextualize_q_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )
            print(f"맥락화된 질문: {contextualized_question}")
        else:
            contextualized_question = question
            print("대화 기록 없음, 원본 질문 사용")
        
        # 문서 검색 (맥락화된 질문으로)
        raw_docs = self.retriever.invoke(contextualized_question)
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
    
    # 개선된 메모리 인식 RAG 체인 생성
    app.state.memory_rag_chain = MemoryAwareRAGChain(retriever, llm, get_session_history)
    app.state.collection = collection

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id가 비어 있습니다.")
    
    try:
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.session_id.strip()))
        config = {"configurable": {"session_id": session_uuid}}
        
        print(f"세션 ID: {session_uuid}")
        print(f"질문: {req.question}")
        
        result = app.state.memory_rag_chain.invoke(
            {"question": req.question.strip()},
            config=config
        )
        
        return AnswerResponse(
            question=req.question.strip(),
            answer=result["answer"],
            used_docs=result["used_docs"],
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
