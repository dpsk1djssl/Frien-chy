# -*- coding: utf-8 -*-

import os
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse # 한글 깨짐 방지를 위해 추가
from pydantic import BaseModel
import uvicorn

# 기존 라이브러리들
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
import torch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

#post-retrieval
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda

# FastAPI 앱 생성
app = FastAPI(
    title="프랜차이즈 QA API",
    description="프랜차이즈 관련 질문에 답변하는 RAG 시스템",
    version="1.0.0"
)

# 글로벌 변수로 모델들 저장
chain = None

# 요청/응답 모델 정의
class QuestionRequest(BaseModel):
    question: str
    max_context: Optional[int] = 10

class AnswerResponse(BaseModel):
    question: str
    answer: str
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str

# --- DB 클라이언트 설정 (수정됨) ---
# 환경 변수에서 DigitalOcean 서버 접속 정보를 읽어옵니다.
qdrant_host = os.getenv("QDRANT_HOST")
collection_name = "qdrant-franchise-db"

# 이 client 객체를 앱 전체에서 사용합니다.
# API 키는 DigitalOcean에 직접 설치한 경우 필요 없습니다.
client = QdrantClient(
    host=qdrant_host,
    port=6333
)
# ------------------------------------


def build_dense_retriever():
    """Dense retriever 구축"""
    try:
        # Initialize embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="nlpai-lab/KURE-v1",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={
                'batch_size': 8,
                'normalize_embeddings': True,
            }
        )

        # VectorStore 래퍼
        # 로컬 경로 대신, 위에서 만든 DigitalOcean 연결용 client를 사용합니다.
        vectorstore = Qdrant(
            client=client, # 전역 client 객체 사용
            collection_name=collection_name,
            embeddings=embedding_model,
        )

        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
    except Exception as e:
        raise Exception(f"Dense retriever 구축 실패: {str(e)}")

def create_prompt():
    """프롬프트 템플릿 생성"""
    return PromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved contexts to answer the question.

        If you don't know the answer, just say that you don't know.
        Answer in Korean using formal, polite language (존댓말).

        Your answers should be concise and direct, providing only the exact information requested.
        Always structure your answer by repeating the subject of the question in your response, following the pattern: "[질문에서 언급된 주체]는/은 [답변]입니다."

        Examples:
        #Question: 주왕산삼계탕부산경남본부의 상호명은 무엇인가요?
        #Answer: 상호명은 하수련주왕산삼계탕부산경남본부입니다.

        #Question: 주왕산삼계탕부산경남본부의 설립일은 언제인가요?
        #Answer: 설립일은 2006년 9월 1일입니다.

        #Question: SBS 아카데미 뷰티스쿨의 대표자는 누구인가요?
        #Answer: SBS 아카데미 뷰티스쿨의 대표자는 이미애입니다.

        #Question: 포카드에는 몇 명의 직원이 있나요?
        #Answer: 2023년 12월 31일 기준으로 포카드는 직원 수가 0명입니다.

        #Question: 밥줄(BAB JOULE)의 업종은 무엇인가요?
        #Answer: 밥줄(BAB JOULE)은 분식 업종에 해당합니다.

        #Question: 버닝스타 가맹점사업자는 어떤 상품이나 용역을 취급할 수 있나요?
        #Answer: 버닝스타 가맹점사업자는 영업지역 내에서 대체재 관계에 있는 동일 또는 유사한 상품이나 용역을 취급할 수 없습니다.

        #Question: {question}

        #Context: {context}

        #Answer:
        """
    )

def initialize_llm():
    """LLM 초기화"""
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0.1,
    )
    return llm

def create_chain(dense_retriever, filter_tokenizer, filter_model, reranker, prompt, llm):
    """RAG 체인 생성"""
    # 필터링 함수
    def filter_docs(inputs):
        docs, query = inputs['context'], inputs['question']
        scored_docs = []
        device = next(filter_model.parameters()).device

        for doc in docs:
            toks = filter_tokenizer(query, doc.page_content, return_tensors="pt", truncation=True, max_length=512)
            toks = {k: v.to(device) for k, v in toks.items()}

            with torch.no_grad():
                outputs = filter_model(**toks)
            score = torch.softmax(outputs.logits, dim=-1)[0][1].item()
            if score > 0.3:
                scored_docs.append((score, doc))

        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in sorted_docs[:10]] if sorted_docs else []
        return {"context": top_docs, "question": query}

    # 리랭킹 함수
    def rerank_docs(docs_query):
        docs, query = docs_query['context'], docs_query['question']
        if not docs:
            return {"context": [], "question": query}
        pairs = [(query, f"브랜드명: {doc.metadata.get('brand_name', '')}\n가맹본부명: {doc.metadata.get('headquarters_name', '')}\n{doc.page_content}") for doc in docs]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in reranked[:5]]
        return {"context": top_docs, "question": query}

    # Create Chain
    chain = (
        {"context": dense_retriever, "question": RunnablePassthrough()}
        | RunnableLambda(filter_docs)
        | RunnableLambda(rerank_docs)
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

async def initialize_models():
    """모든 모델을 초기화하는 함수"""
    try:
        print(" 모델 초기화를 시작합니다...")

        # Dense retriever 구축
        print("Dense retriever 로딩 중...")
        retriever = build_dense_retriever()

        # Prompt 생성
        print("프롬프트 템플릿 생성 중...")
        prompt = create_prompt()

        # LLM 초기화
        print(" LLM 초기화 중...")
        llm = initialize_llm()

        # Post-retrieval 모델들 설정
        print("필터링 모델 로딩 중...")
        filter_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        filter_model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=2)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        filter_model.to(device)
        filter_model.eval()

        print("리랭커 로딩 중...")
        reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

        # 체인 생성
        print("RAG 체인 구성 중...")
        chain = create_chain(retriever, filter_tokenizer, filter_model, reranker, prompt, llm)

        print("모든 모델이 성공적으로 로딩되었습니다!")
        return chain

    except Exception as e:
        print(f"모델 초기화 실패: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로딩"""
    global chain
    try:
        chain = await initialize_models()
        print("서버가 준비되었습니다!")
    except Exception as e:
        print(f"서버 시작 실패: {str(e)}")
        raise

@app.get("/", response_model=dict)
async def root():
    """루트 엔드포인트"""
    return {
        "message": "프랜차이즈 QA API에 오신 것을 환영합니다!",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    global chain
    if chain is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩되지 않았습니다")

    return HealthResponse(
        status="healthy",
        message="모든 시스템이 정상 작동 중입니다"
    )

@app.post("/ask") # 한글 깨짐 방지를 위해 response_model 제거
async def ask_question(request: QuestionRequest):
    """질문에 대한 답변 생성"""
    global chain

    if chain is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩되지 않았습니다")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어있습니다")

    try:
        # RAG 시스템으로 답변 생성
        answer = chain.invoke(request.question)

        # 한글 깨짐 방지를 위해 JSONResponse 사용
        content = {
            "question": request.question,
            "answer": answer,
            "status": "success"
        }
        return JSONResponse(content=content)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/models/status")
async def models_status():
    """모델 상태 확인"""
    global chain

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return {
        "chain_loaded": chain is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "environment_variables": {
            "GEMINI_API_KEY": "✅" if os.getenv("GEMINI_API_KEY") else "❌",
            "QDRANT_HOST": os.getenv("QDRANT_HOST", "not_set")
        }
    }

if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
