# build_production_db_v2.py
import os
import json
import time
import gc
from glob import glob
from typing import List, Iterator

from tqdm import tqdm
import torch
from langchain.schema import Document
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
import ijson # 대용량 JSON 처리를 위한 라이브러리
from huggingface_hub import snapshot_download # 모델 다운로드 진행률 표시를 위해 추가

# ==============================================================================
# ⚙️ 1. 사용자 설정 영역 (이 부분만 수정해주세요!)
# ==============================================================================

# 1-1. 데이터 경로 설정
DATA_ROOT = "./finaldata"

# 1-2. Qdrant 서버 설정
QDRANT_URL = "http://165.22.105.79:6333"
API_KEY = None
COLLECTION_NAME = "qdrant-franchise-db"

# 1-3. 임베딩 모델 및 성능 설정
MODEL_NAME = "nlpai-lab/KURE-v1"
EMBEDDING_BATCH_SIZE = 64
DOCUMENT_UPLOAD_BATCH_SIZE = 64

# ==============================================================================
# 🛠️ 2. 유틸리티 함수 (수정할 필요 없음)
# ==============================================================================

def download_model_with_progress(model_name: str):
    """
    Hugging Face Hub에서 모델을 미리 다운로드하고 진행 상황을 표시합니다.
    """
    print(f"'{model_name}' 모델 다운로드를 시작합니다. (진행률 표시)")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=False, # 윈도우 호환성
            resume_download=True # 이어받기 기능 활성화
        )
        print("✅ 모델 다운로드 완료.")
    except Exception as e:
        print(f"❌ 모델 다운로드 중 오류 발생: {e}")
        print("인터넷 연결을 확인하거나, 모델 이름을 다시 확인해주세요.")
        raise

def create_comprehensive_document(data: dict) -> str:
    # (기존과 동일한 함수)
    ql = data.get('QL', {}) or {}
    jng = data.get('JNG_INFO', {}) or {}
    attr = data.get('ATTRB_INFO', {}) or {}
    brand = jng.get('BRAND_NM', '') or ''
    hq = jng.get('JNGHDQRTRS_CONM_NM', '') or ''
    parts = [f"[브랜드] {brand}", f"[가맹본부] {hq}", ""]
    orig = (ql.get('ORIGINAL_TEXT') or '')
    if orig: parts.extend(['[원본]', orig, ""])
    abstracted = (ql.get('ABSTRACTED_SUMMARY_TEXT') or '').strip()
    if abstracted: parts.extend(['[추상 요약]', abstracted, ""])
    extracted = (ql.get('EXTRACTED_SUMMARY_TEXT') or '').strip()
    if extracted: parts.extend(['[추출 요약]', extracted, ""])
    return '\n'.join(parts)

def create_enhanced_metadata(data: dict, file_name: str, idx: int) -> dict:
    # (기존과 동일한 함수)
    jng = data.get('JNG_INFO', {}) or {}
    attr = data.get('ATTRB_INFO', {}) or {}
    return {
        'brand_name': jng.get('BRAND_NM', '') or '',
        'headquarters_name': jng.get('JNGHDQRTRS_CONM_NM', '') or '',
        'source_file': file_name,
        'index_in_file': idx,
    }

def stream_documents_from_json(file_path: str) -> Iterator[Document]:
    """
    대용량 JSON 파일을 통째로 읽지 않고, 스트리밍 방식으로 하나씩 읽어
    Document 객체를 생성하는 제너레이터(Generator) 함수입니다.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # ijson.items는 파일 전체를 메모리에 올리지 않고, 'item' 키 아래의 객체들을 하나씩 가져옵니다.
        records = ijson.items(f, 'item')
        for i, data in enumerate(records):
            try:
                content = create_comprehensive_document(data)
                if content.strip():
                    metadata = create_enhanced_metadata(data, os.path.basename(file_path), i)
                    yield Document(page_content=content, metadata=metadata)
            except Exception as e:
                print(f"  ⚠️ 항목 {i} 처리 중 오류 발생: {e}")
                continue

def aggressive_memory_cleanup():
    """GPU 메모리를 정리하는 함수"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# ==============================================================================
# 🚀 3. 메인 실행 로직
# ==============================================================================
def main():
    start_time = time.time()
    
    # 0. 모델 미리 다운로드 (진행률 표시)
    download_model_with_progress(MODEL_NAME)

    # 1. 임베딩 모델 로드
    print(f"🤖 임베딩 모델 '{MODEL_NAME}'을 로딩합니다...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'batch_size': EMBEDDING_BATCH_SIZE, 'normalize_embeddings': True}
    )

    # 2. Qdrant 서버 연결 및 DB 초기화
    print(f"\n🔌 Qdrant 서버({QDRANT_URL})에 연결합니다...")
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY, timeout=60)
    print(f"🔥 기존 '{COLLECTION_NAME}' 컬렉션을 삭제하고 새로 생성합니다...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)
    print("✅ DB 초기화 완료.")

    # 3. JSON 파일 목록 가져오기
    json_files = sorted(glob(os.path.join(DATA_ROOT, "**/*.json"), recursive=True))
    if not json_files:
        print(f"❌ '{DATA_ROOT}' 경로에서 JSON 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    print(f"\n📊 총 {len(json_files)}개의 JSON 파일을 처리합니다.")

    # 4. 파일별로 데이터 처리 및 DB에 추가
    total_docs_processed = 0
    for file_path in tqdm(json_files, desc="전체 파일 처리 중"):
        try:
            print(f"\n📄 '{os.path.basename(file_path)}' 파일 스트리밍 처리 시작...")
            
            doc_generator = stream_documents_from_json(file_path)
            
            batch_for_upload = []
            # 파일 내의 문서들을 작은 배치로 나누어 처리합니다.
            with tqdm(desc=f"  -> '{os.path.basename(file_path)}' 업로드 중", unit=" docs") as pbar:
                for doc in doc_generator:
                    batch_for_upload.append(doc)
                    if len(batch_for_upload) >= DOCUMENT_UPLOAD_BATCH_SIZE:
                        vectorstore.add_documents(batch_for_upload)
                        pbar.update(len(batch_for_upload))
                        total_docs_processed += len(batch_for_upload)
                        batch_for_upload = [] # 배치 초기화
                
                # 마지막에 남은 문서들을 처리합니다.
                if batch_for_upload:
                    vectorstore.add_documents(batch_for_upload)
                    pbar.update(len(batch_for_upload))
                    total_docs_processed += len(batch_for_upload)

            aggressive_memory_cleanup()
            
        except Exception as e:
            print(f"\n⚠️ 파일 처리 중 오류 발생: {os.path.basename(file_path)} - {e}")
            continue
            
    end_time = time.time()
    
    print("\n" + "="*60)
    print("🎉 모든 데이터 처리 및 업로드가 성공적으로 완료되었습니다!")
    print("="*60)
    
    final_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"📄 최종 문서(청크) 수: {total_docs_processed:,}개")
    print(f"💾 DB에 저장된 최종 포인트 수: {final_info.points_count:,}개")
    print(f"⏱️ 총 소요 시간: {(end_time - start_time) / 60:.2f}분")

if __name__ == "__main__":
    main()
