# -*- coding: utf-8 -*-
# main_inference_final.py
# LangGraph의 유연한 구조와 대화 메모리 기능을 결합한 최종 버전입니다.
#
# 주요 특징:
# 1. LangGraph 기반: 각 RAG 단계를 명확한 '노드'로 정의하여 가독성과 확장성을 높였습니다.
# 2. 메모리 기능 완비: RunnableWithMessageHistory를 사용하여 LangGraph 전체에 대화 기록 관리 기능을 적용했습니다.
# 3. 질문 재구성: 대화 맥락을 파악하여 후속 질문을 독립적인 질문으로 재구성하는 'contextualize_question' 노드를 그래프의 시작점으로 추가했습니다.

import os
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
from langchain_postgres import PostgresChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# --------------------
# Config
# --------------------
CFG_NAME = "LangGraph_With_History_v3.1"
EMBED_MODEL = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RETRIEVE = 10
PRE_CUT_TOPN = 5
FINAL_TOPK = 5
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

def build_reranker_model():
    return CrossEncoder(RERANK_MODEL, device=DEVICE)

# --------------------
# Utilities
# --------------------
def concat_context(docs: List[Document], max_chars: int = 4000, sep: str = "\n---\n"):
    # ...(이전과 동일)...
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
    # ...(이전과 동일)...
    md = d.metadata or {}
    for k in ("_qdrant_score", "score", "similarity"):
        if k in md:
            try: return float(md[k])
            except: pass
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
    chat_history: List[Any]

def build_graph(retriever, reranker_model, llm):
    
    contextualize_q_chain = CONTEXTUALIZE_QUESTION_PROMPT | llm | StrOutputParser()
    
    # 1. (NEW) Contextualize Question Node
    def contextualize_question(state: GraphState):
        """대화 기록을 바탕으로 질문을 재구성합니다."""
        if state.get("chat_history"):
            contextualized_question = contextualize_q_chain.invoke(
                {"question": state["question"], "chat_history": state["chat_history"]}
            )
            return {"question": contextualized_question}
        else:
            return {"question": state["question"]}

    # 2. Retrieval Node
    def retrieve(state: GraphState):
        """질문을 사용하여 문서를 검색합니다."""
        docs = retriever.invoke(state["question"])
        return {"docs": docs, "question": state["question"]}

    # 3. Filtering Node
    def filter_docs(state: GraphState):
        """스코어 기준으로 문서를 필터링합니다."""
        docs = [d for d in state["docs"] if _get_qdrant_score(d) >= MIN_SCORE]
        if not docs: docs = state["docs"] # 필터링 후 문서가 없으면 원본 사용
        return {"docs": docs}

    # 4. Precut Node
    def precut(state: GraphState):
        """리랭커에 전달할 문서 수를 자릅니다."""
        return {"docs": state["docs"][:PRE_CUT_TOPN]}

    # 5. Rerank Node
    def rerank(state: GraphState):
        """CrossEncoder를 사용해 문서를 재정렬합니다."""
        question = state["question"]
        docs = state["docs"]
        if not docs:
            return {"docs": []}
            
        pairs = [[question, doc.page_content] for doc in docs]
        scores = reranker_model.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for score, doc in scored_docs[:FINAL_TOPK]]
        return {"docs": reranked_docs}

    # 6. LLM QA Node
    def generate_answer(state: GraphState):
        """최종 답변을 생성합니다."""
        question = state["question"]
        docs = state["docs"]
        
        ctx = concat_context(docs)
        rag_chain = (
            RunnablePassthrough.assign(context=lambda x: ctx)
            | ANSWER_PROMPT
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke({"question": question})
        
        return {
            "answer": answer,
            "used_docs": [serialize_doc(d) for d in docs]
        }

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("contextualize_question", contextualize_question)
    graph.add_node("retrieval", retrieve)
    graph.add_node("filter", filter_docs)
    graph.add_node("precut", precut)
    graph.add_node("rerank", rerank)
    graph.add_node("generate_answer", generate_answer)

    # Build graph
    graph.set_entry_point("contextualize_question")
    graph.add_edge("contextualize_question", "retrieval")
    graph.add_edge("retrieval", "filter")
    graph.add_edge("filter", "precut")
    graph.add_edge("precut", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()

# --------------------
# FastAPI App & Permanent Memory
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

app = FastAPI(title="프랜차이즈 QA API (LangGraph + Memory)", version="3.2.0")

# 임시 메모리 저장소
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return PostgresChatMessageHistory(
        session_id,                      
        "message_store",                 
        connection_string=DATABASE_URL   
    )

@app.on_event("startup")
def on_startup():
    embeddings = build_embeddings()
    retriever, collection = build_qdrant_vectorstore(embeddings)
    reranker_model = build_reranker_model()
    llm = build_llm()
    
    # LangGraph 생성
    graph = build_graph(retriever, reranker_model, llm)
    
    # LangGraph에 메모리 기능 래핑
    app.state.chain_with_history = RunnableWithMessageHistory(
        graph,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    app.state.collection = collection

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id가 비어 있습니다.")
        
    try:
        config = {"configurable": {"session_id": req.session_id.strip()}}
        result = app.state.chain_with_history.invoke(
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
