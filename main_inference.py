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
from pydantic import BaseModel
import uvicorn

# -------- CPU/Thread & Runtime Tuning (configurable) --------
# Defaults are conservative for 4GB CPU-only servers.
CPU_THREADS = int(os.getenv("TORCH_NUM_THREADS", os.getenv("CPU_THREADS", "2")))
# Hint libraries to limit oversubscription
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
try:
    torch.set_num_threads(CPU_THREADS)
    # Reduce interop thread contention
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass
PORT = int(os.getenv("PORT", "5000"))
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))
# ------------------------------------------------------------

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient

# Reranker (LangChain) — support both modern/community and older paths
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except Exception:
    from langchain_community.document_transformers import CrossEncoderReranker

# --------------------
# Fixed config
# --------------------
CFG_NAME = "GS_k10_fscore_pc5_rrbge_lite2"

EMBED_MODEL = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RETRIEVE = 10            # retrieve k=10
PRE_CUT_TOPN = 5           # cut to 5 before reranker
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINAL_TOPK = 5             # feed 5 docs to LLM

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
    """Build Qdrant vectorstore using REST or GRPC per QDRANT_MODE (remote)."""
    mode = os.getenv("QDRANT_MODE", "GRPC").lower()
    collection = os.getenv("QDRANT_COLLECTION")
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url and not os.getenv("QDRANT_HOST"):
        raise RuntimeError("Qdrant: provide QDRANT_URL or QDRANT_HOST.")

    if mode == "rest":
        # REST MODE
        client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
    else:
        # GRPC MODE
        host = os.getenv("QDRANT_HOST")
        grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
        client = QdrantClient(host=host, grpc_port=grpc_port, api_key=api_key, prefer_grpc=True, https=False)

    if not collection:
        collection = "franchise-db-1"

    vs = LCQdrant(
        client=client,
        collection_name=collection,
        embeddings=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )
    return vs, collection

def build_llm():
    api_key = _require_env("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1,
    )

def build_reranker(top_n: int = FINAL_TOPK):
    # LangChain CrossEncoderReranker wrapper (internally uses sentence-transformers)
    return CrossEncoderReranker(model=RERANK_MODEL, top_n=top_n)

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
    """Try common places where LC Qdrant may stash similarity score."""
    md = d.metadata or {}
    for k in ("_qdrant_score", "score", "similarity"):
        if k in md:
            try:
                return float(md[k])
            except Exception:
                pass
    return float(getattr(d, "score", 0.0))

# --------------------
# Prompt
# --------------------
PROMPT = ChatPromptTemplate.from_template(
    """
    당신은 질의응답 보조자입니다. 아래 컨텍스트만을 사용하여 질문에 답하세요.
    모르면 모른다고 하세요. 한국어(존댓말)로 간결히 답변합니다.
    항상 "[질문에서 언급된 주체]는/은 [답변]입니다." 형태로 시작하세요.

    # 질문: {question}

    # 컨텍스트:
    {context}

    # 답변:
    """.strip()
)

# --------------------
# FastAPI models
# --------------------
class QuestionRequest(BaseModel):
    question: str

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
# App state
# --------------------
class State:
    embeddings = None
    vectorstore = None
    retriever = None
    reranker = None
    llm = None
    collection = None
    chain = None

state = State()
app = FastAPI(
    title="프랜차이즈 QA API (LangChain)",
    description=f"LangChain 기반 RAG 파이프라인 - 설정 {CFG_NAME}",
    version="2.3.0",
)

# --------------------
# Runnable Chain (score filter → precut → rerank → LLM)
# --------------------
PRE_CUT_TOPN = 5
MIN_SCORE = float(os.getenv("QDRANT_MIN_SCORE", "0.5"))

def create_chain(dense_retriever, reranker, prompt, llm):
    # 1) score-only 필터 (Qdrant 점수 임계치 적용)
    def score_filter(inputs):
        q, docs = inputs["question"], inputs["docs"]
        filtered = [d for d in docs if _get_qdrant_score(d) >= MIN_SCORE]
        if not filtered:  # overly strict threshold → fallback to original
            filtered = docs
        return {"question": q, "docs": filtered}

    # 2) pre-cut 5
    def precut(inputs):
        return {"question": inputs["question"], "docs": inputs["docs"][:PRE_CUT_TOPN]}

    # 3) CE rerank top-5 (LangChain wrapper handles scoring + top_n cut)
    def ce_rerank(inputs):
        q, docs = inputs["question"], inputs["docs"]
        if not docs:
            return {"question": q, "docs": []}
        top_docs = reranker.compress_documents(docs, q)  # top_n=5 set in build_reranker
        return {"question": q, "docs": top_docs}

    # 4) payload for LLM
    def make_payload(inputs):
        docs = inputs["docs"]
        return {
            "question": inputs["question"],
            "context": concat_context(docs, max_chars=4000),
            "used_docs": [serialize_doc(d) for d in docs],
        }

    answer_chain = prompt | llm | StrOutputParser()
    final = RunnableParallel(
        answer=answer_chain,
        used_docs=RunnableLambda(lambda x: x["used_docs"]),
    )

    chain = (
        {"question": RunnablePassthrough(), "docs": dense_retriever}  # k=10
        | RunnableLambda(score_filter)   # threshold
        | RunnableLambda(precut)         # pc5
        | RunnableLambda(ce_rerank)      # rr
        | RunnableLambda(make_payload)   # context + used_docs
        | final                          # {"answer": ..., "used_docs": [...]}
    )
    return chain

@app.on_event("startup")
def on_startup():
    embeddings = build_embeddings()
    vs, collection = build_qdrant_vectorstore(embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": K_RETRIEVE})
    reranker = build_reranker(top_n=FINAL_TOPK)
    llm = build_llm()
    chain = create_chain(retriever, reranker, PROMPT, llm)

    state.embeddings = embeddings
    state.vectorstore = vs
    state.retriever = retriever
    state.reranker = reranker
    state.llm = llm
    state.collection = collection
    state.chain = chain

@app.get("/", response_model=dict)
def root():
    return {"message": "LangChain RAG API", "health": "/health", "ask": "/ask"}

@app.get("/health", response_model=HealthResponse)
def health():
    if not all([state.embeddings, state.vectorstore, state.retriever, state.reranker, state.llm, state.collection, state.chain]):
        raise HTTPException(status_code=503, detail="시스템 초기화가 완료되지 않았습니다.")
    return HealthResponse(status="healthy", message="정상 동작 중입니다.")

@app.get("/models/status")
def models_status():
    return {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "env": {
            "GEMINI_API_KEY": "✅" if os.getenv("GEMINI_API_KEY") else "❌",
            "QDRANT_MODE": os.getenv("QDRANT_MODE", "GRPC"),
            "QDRANT_URL": os.getenv("QDRANT_URL", ""),
            "QDRANT_HOST": os.getenv("QDRANT_HOST", ""),
            "QDRANT_GRPC_PORT": os.getenv("QDRANT_GRPC_PORT", ""),
            "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", ""),
            "QDRANT_MIN_SCORE": os.getenv("QDRANT_MIN_SCORE", "0.5"),
            "UVICORN_WORKERS": os.getenv("UVICORN_WORKERS", "1"),
            "TORCH_NUM_THREADS": os.getenv("TORCH_NUM_THREADS", os.getenv("CPU_THREADS", "2")),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", ""),
        },
        "cfg": CFG_NAME,
        "collection": state.collection,
    }

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")
    try:
        result = state.chain.invoke(req.question.strip())
        return AnswerResponse(
            question=req.question.strip(),
            answer=result["answer"],
            used_docs=[UsedDoc(**ud) for ud in result["used_docs"]],
            status="success",
            cfg=CFG_NAME,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main_inference:app", host="0.0.0.0", port=PORT, log_level="info", workers=UVICORN_WORKERS)
