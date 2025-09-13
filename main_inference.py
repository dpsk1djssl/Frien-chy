# -*- coding: utf-8 -*-
# main_inference_final.py
# LangGraph + PostgreSQL Memory 완전 수정 버전

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
CFG_NAME = "LangGraph_With_PostgreSQL_v4.1_Fixed"
EMBED_MODEL = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RETRIEVE = 10
FINAL_TOPK = 5
MIN_SCORE = float(os.getenv("QDRANT_MIN_SCORE", "0.5"))

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

    @property
    def messages(self) -> List[BaseMessage]:
        """세션의 모든 메시지 조회"""
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
        return messages

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
        except Exception as e:
            print(f"메시지 삭제 오류: {e}")

# --------------------
# Utilities
# --------------------
def concat_context(docs: List[Document], max_chars: int = 4000, sep: str = "\n---\n"):
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
# Prompts
# --------------------
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
당신은 질의응답 보조자입니다. 아래 컨텍스트만을 사용하여 질문에 답하세요.
모르면 모른다고 하세요. 한국어(존댓말)로 간결히 답변합니다.
항상 "[질문에서 언급된 주체]는/은 [답변]입니다." 형태로 시작하세요.

# 질문: {question}

# 컨텍스트:
{context}

# 답변:
""".strip())

CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "주어진 대화 기록과 마지막 질문을 사용하여, 대화의 맥락을 모르는 사람도 이해할 수 있는 완전한 독립형 질문으로 재구성하세요. 질문 외에 다른 말은 덧붙이지 마세요."),
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
# Memory-Aware RAG Chain
# --------------------
class MemoryAwareRAGChain:
    def __init__(self, retriever, llm, get_session_history):
        self.retriever = retriever
        self.llm = llm
        self.get_session_history = get_session_history
        self.contextualize_q_chain = CONTEXTUALIZE_QUESTION_PROMPT | llm | StrOutputParser()

    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """메모리 기능을 포함한 RAG 처리"""
        session_id = config.get("configurable", {}).get("session_id")
        if not session_id:
            raise ValueError("session_id가 config에 없습니다.")
        
        # 세션 히스토리 가져오기
        history = self.get_session_history(session_id)
        chat_history = history.messages
        
        question = input_data["question"]
        
        # 사용자 메시지 저장
        history.add_message(HumanMessage(content=question))
        
        # 대화 기록이 있으면 질문을 맥락화
        if chat_history:
            print(f"대화 기록 발견: {len(chat_history)}개 메시지")
            contextualized_question = self.contextualize_q_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )
            print(f"원본 질문: {question}")
            print(f"맥락화된 질문: {contextualized_question}")
        else:
            contextualized_question = question
            print("대화 기록 없음, 원본 질문 사용")

        # 문서 검색
        docs = self.retriever.invoke(contextualized_question)
        print(f"검색된 문서 수: {len(docs)}")
        
        # 점수 필터링 및 상위 K개 선택
        filtered_docs = [d for d in docs if _get_qdrant_score(d) >= MIN_SCORE]
        if not filtered_docs:
            filtered_docs = docs
        final_docs = filtered_docs[:FINAL_TOPK]
        print(f"최종 사용 문서 수: {len(final_docs)}")
        
        # 답변 생성
        ctx = concat_context(final_docs)
        rag_chain = (
            RunnablePassthrough.assign(context=lambda x: ctx)
            | ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke({"question": question})
        print(f"생성된 답변: {answer}")
        
        # AI 응답 저장
        history.add_message(AIMessage(content=answer))
        
        return {
            "question": question,
            "answer": answer,
            "used_docs": [serialize_doc(d) for d in final_docs]
        }

# --------------------
# FastAPI App
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

app = FastAPI(title="프랜차이즈 QA API (LangGraph + PostgreSQL Memory)", version="4.1.0")

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
    
    # 메모리 인식 RAG 체인 생성
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
        return {
            "session_id": session_id,
            "session_uuid": session_uuid,
            "message_count": len(messages),
            "messages": [{"type": type(msg).__name__, "content": msg.content} for msg in messages]
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