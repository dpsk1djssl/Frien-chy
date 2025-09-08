# -*- coding: utf-8 -*-
# main_inference.py  (LangChain-first)
# FastAPI entrypoint for the Franchise RAG system with the fixed config:
#   GS_k10_fscore_pc5_rrbge_lite2
#     - Retrieval: k=10 from Qdrant via LangChain VectorStore retriever
#     - Filter: score-only with Qdrant score threshold (QDRANT_MIN_SCORE)
#     - Pre-cut: 5 (slice BEFORE rerank)
#     - Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 via LangChain CrossEncoderReranker
#     - Final top-k for LLM context: 5
#     - LLM: Gemini 2.5 Flash
#
# Environment variables:
#   GEMINI_API_KEY  (required)
#   QDRANT_MODE: REST or GRPC (default: GRPC)
#   # Remote: QDRANT_URL (or QDRANT_HOST/QDRANT_GRPC_PORT), QDRANT_API_KEY, QDRANT_COLLECTION (default: franchise-db-1)
#   QDRANT_MIN_SCORE (default: 0.5)

import os
from typing import List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# -------- CPU/Thread & Runtime Tuning (configurable) --------
CPU_THREADS = int(os.getenv("TORCH_NUM_THREADS", os.getenv("CPU_THREADS", "2")))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
try:
    torch.set_num_threads(CPU_THREADS)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass
PORT = int(os.getenv("PORT", "8000"))
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))
# ------------------------------------------------------------

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# --------------------
# Fixed config
# --------------------
CFG_NAME = "GS_k10_fscore_pc5_rrbge_lite2_history_aware"

EMBED_MODEL = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RETRIEVE = 10
PRE_CUT_TOPN = 5
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINAL_TOPK = 5
MIN_SCORE = float(os.getenv("QDRANT_MIN_SCORE", "0.5"))

# --------------------
# Builders
# --------------------
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_qdrant_vectorstore(embeddings):
    mode = os.getenv("QDRANT_MODE", "GRPC").lower()
    collection = os.getenv("QDRANT_COLLECTION")
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url and not os.getenv("QDRANT_HOST"):
        raise RuntimeError("Qdrant: provide QDRANT_URL or QDRANT_HOST.")

    if mode == "rest":
        client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
    else:
        host = os.getenv("QDRANT_HOST")
        grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
        client = QdrantClient(host=host, grpc_port=grpc_port, api_key=api_key, prefer_grpc=True, https=False)

    if not collection:
        collection = "franchise-db-1"

    vs = LCQdrant(
        client=client, collection_name=collection, embeddings=embeddings,
        content_payload_key="page_content", metadata_payload_key="metadata",
    )
    return vs, collection

def build_llm():
    api_key = _require_env("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=0.1,
    )

def build_reranker_model():
    return CrossEncoder(RERANK_MODEL, device=DEVICE)

# --------------------
# Utilities
# --------------------
def concat_context(docs: List[Document], max_chars: int = 4000, sep: str = "\n---\n") -> str:
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
            except Exception:
                pass
    return float(getattr(d, "score", 0.0))

# --------------------
# Prompts
# --------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """당신은 질의응답 보조자입니다. 아래 컨텍스트만을 사용하여 질문에 답하세요.
모르면 모른다고 하세요. 한국어(존댓말)로 간결히 답변합니다.
항상 "[질문에서 언급된 주체]는/은 [답변]입니다." 형태로 시작하세요."""),
        ("user", """# 질문: {question}

# 컨텍스트:
{context}

# 답변:"""),
    ]
)

CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "주어진 대화 기록과 마지막 질문을 사용하여, 대화의 맥락을 모르는 사람도 이해할 수 있는 완전한 독립형 질문으로 재구성하세요. 질문 외에 다른 말은 덧붙이지 마세요."),
        MessagesPlaceholder("chat_history"),
        ("user", "{question}"),
    ]
)

# --------------------
# FastAPI models
# --------------------
class QuestionRequest(BaseModel):
    question: str
    session_id: str = Field(..., description="각 사용자의 대화를 구분하는 고유 ID")

class UsedDoc(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}

class AnswerResponse(BaseModel):
    question: str
    answer: str
    used_docs: List[UsedDoc]
    status: str
    cfg: str

class HealthResponse(BaseModel):
    status: str
    message: str

# --------------------
# App state & Memory Store
# --------------------
class State:
    chain_with_history = None
    collection = None

state = State()
# 임시 메모리 저장소 (서버 재시작 시 초기화됨)
# key: session_id, value: ChatMessageHistory 객체
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID에 해당하는 대화 기록을 가져오거나 새로 생성합니다."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


app = FastAPI(
    title="프랜차이즈 QA API (LangChain with Memory)",
    description=f"History-Aware RAG 파이프라인 - 설정 {CFG_NAME}",
    version="3.0.0",
)

# --------------------
# Runnable Chain (History-Aware RAG)
# --------------------
@app.on_event("startup")
def on_startup():
    # --- 1. 모델 및 리트리버 초기화 ---
    embeddings = build_embeddings()
    vs, collection = build_qdrant_vectorstore(embeddings)
    dense_retriever = vs.as_retriever(search_kwargs={"k": K_RETRIEVE})
    reranker_model = build_reranker_model()
    llm = build_llm()

    # --- 2. 질문 재구성 체인 정의 ---
    contextualize_q_chain = CONTEXTUALIZE_QUESTION_PROMPT | llm | StrOutputParser()

    # --- 3. 핵심 RAG 파이프라인 정의 (기존 로직) ---
    def score_filter(docs: List[Document]) -> List[Document]:
        filtered = [d for d in docs if _get_qdrant_score(d) >= MIN_SCORE]
        return filtered if filtered else docs

    def precut(docs: List[Document]) -> List[Document]:
        return docs[:PRE_CUT_TOPN]

    def ce_rerank(inputs: Dict) -> List[Document]:
        q, docs = inputs["question"], inputs["docs"]
        if not docs:
            return []
        pairs = [[q, doc.page_content] for doc in docs]
        scores = reranker_model.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:FINAL_TOPK]]

    # --- 4. 전체 체인 결합 ---
    
    # 4a. 대화 기록을 바탕으로 질문을 재구성할지 결정하는 부분
    def route_contextualize_question(input: Dict) -> RunnableLambda:
        if input.get("chat_history"):
            return RunnablePassthrough.assign(
                question=contextualize_q_chain
            )
        else:
            return RunnablePassthrough.assign(question=lambda x: x["question"])
    
    # 4b. RAG 답변 생성 부분
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=lambda x: concat_context(x["docs"])
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    # 4c. 전체 흐름 조립
    conversational_rag_chain = (
        route_contextualize_question
        | RunnablePassthrough.assign(
            docs=lambda x: dense_retriever.invoke(x["question"])
        )
        | RunnablePassthrough.assign(docs=lambda x: score_filter(x["docs"]))
        | RunnablePassthrough.assign(docs=lambda x: precut(x["docs"]))
        | RunnablePassthrough.assign(
            docs=lambda x: ce_rerank({"question": x["question"], "docs": x["docs"]})
        )
        | RunnablePassthrough.assign(
            answer=rag_chain_from_docs
        )
        | (lambda x: {"answer": x["answer"], "used_docs": [serialize_doc(d) for d in x["docs"]]})
    )

    # --- 5. 대화 기록 관리 기능 래핑 ---
    chain_with_history = RunnableWithMessageHistory(
        conversational_rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    state.chain_with_history = chain_with_history
    state.collection = collection

# --------------------
# API Endpoints
# --------------------
@app.get("/", response_model=dict)
def root():
    return {"message": "LangChain RAG API with Memory", "health": "/health", "ask": "/ask"}

@app.get("/health", response_model=HealthResponse)
def health():
    if not state.chain_with_history or not state.collection:
        raise HTTPException(status_code=503, detail="시스템 초기화가 완료되지 않았습니다.")
    return HealthResponse(status="healthy", message="정상 동작 중입니다.")

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")
    if not req.session_id or not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id가 비어 있습니다.")
    try:
        config = {"configurable": {"session_id": req.session_id.strip()}}
        result = state.chain_with_history.invoke({"question": req.question.strip()}, config=config)
        
        return AnswerResponse(
            question=req.question.strip(),
            answer=result["answer"],
            used_docs=[UsedDoc(**ud) for ud in result["used_docs"]],
            status="success",
            cfg=CFG_NAME,
        )
    except Exception as e:
        # 실제 운영 환경에서는 로깅 라이브러리(예: logging)를 사용하는 것이 좋습니다.
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"내부 서버 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", workers=UVICORN_WORKERS)
