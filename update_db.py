# update_db.py
import os
import json
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# --- 1. 설정 영역 ---
# DigitalOcean DB 서버의 IP 주소를 입력하세요.
QDRANT_HOST = "159.223.73.50"

# Qdrant에 생성된 컬렉션(저장소) 이름
COLLECTION_NAME = "qdrant-franchise-db"

# 새로 추가할 데이터가 담긴 원본 파일 경로
NEW_DATA_PATH = "./new_franchise_data.jsonl"

# 사용할 임베딩 모델
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"
# -----------------------------


def load_and_split_documents(file_path):
    """새로운 JSONL 데이터를 로드하고 작은 청크로 분할합니다."""
    print(f"'{file_path}'에서 새로운 문서를 로딩합니다...")
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.',
        content_key="page_content",
        json_lines=True,
        metadata_func=lambda record, metadata: record.get("metadata", {})
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    print(f"새로운 문서를 {len(split_docs)}개의 청크로 분할했습니다.")
    return split_docs


def main():
    """메인 실행 함수"""
    # 1. 새로운 문서 로딩 및 분할
    document_chunks = load_and_split_documents(NEW_DATA_PATH)

    # 2. 임베딩 모델 초기화
    print(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}'을 로딩합니다...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 3. DigitalOcean DB 서버에 연결
    print(f"DigitalOcean DB 서버({QDRANT_HOST})에 연결합니다...")
    client = QdrantClient(host=QDRANT_HOST, port=6333, timeout=60)

    # 4. 새로운 문서 청크를 벡터로 변환
    print("새로운 문서 청크를 벡터로 변환하는 중입니다...")
    chunk_contents = [doc.page_content for doc in document_chunks]
    vectors = embedding_model.embed_documents(chunk_contents)
    
    # 5. Qdrant에 업로드할 데이터 준비
    points_to_upload = []
    for i, doc in enumerate(document_chunks):
        payload = {"page_content": doc.page_content, **doc.metadata}
        points_to_upload.append(
            models.PointStruct(id=str(uuid.uuid4()), vector=vectors[i], payload=payload)
        )

    # 6. DigitalOcean DB 서버에 데이터 추가 (Upsert)
    print(f"{len(points_to_upload)}개의 새로운 데이터를 DB 서버에 추가합니다...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points_to_upload,
        wait=True,
        batch_size=128
    )

    print("\n🎉 데이터베이스 업데이트가 성공적으로 완료되었습니다!")
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"현재 컬렉션의 총 포인트 수: {collection_info.points_count}")


if __name__ == "__main__":
    main()
