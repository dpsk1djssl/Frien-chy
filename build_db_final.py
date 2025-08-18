# build_db.py
import os
import json
import zipfile
import subprocess
from glob import glob
from tqdm import tqdm
import torch

# LangChain 관련 imports
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# --- 1. 경로 설정 (필수!) ---
# 이 4개의 변수만 본인 환경에 맞게 정확히 수정해주세요.

# 압축을 푼 데이터 폴더 경로
TRAIN_PATH = "./finaldata"

# 새로 만들 Qdrant 로컬 DB를 저장할 경로
# 예: "C:/Users/tilon/Downloads/frienchy-backend-v1/finaldb"
SAVE_DIR = "./finaldb"

# 사용할 임베딩 모델
MODEL_NAME = "nlpai-lab/KURE-v1"

# Qdrant 컬렉션 이름
COLLECTION_NAME = "qdrant-franchise-db-no-qa"
# -----------------------------

# --- 2. 배치 사이즈 설정 (성능에 맞게 조절) ---
# 로컬 컴퓨터 CPU로 실행할 경우, 너무 큰 값은 오히려 느릴 수 있습니다. (예: 32, 64)
# GPU가 있다면 메모리에 맞춰 조절하세요. (예: 128, 256, 512)
BATCH_SIZE = 256

# ---------------------------------------------


def create_comprehensive_document(data: dict) -> str:
    """데이터에서 [원본], [추상 요약], [추출 요약]만 포함하는 텍스트를 생성합니다."""
    ql = data.get('QL', {}) or {}
    jng = data.get('JNG_INFO', {}) or {}
    attr = data.get('ATTRB_INFO', {}) or {}

    brand = jng.get('BRAND_NM', '')
    hq = jng.get('JNGHDQRTRS_CONM_NM', '')
    
    parts = []
    parts.append(f"[브랜드] {brand}")
    parts.append(f"[가맹본부] {hq}")
    parts.append("")

    # 원문
    orig = (ql.get('ORIGINAL_TEXT') or '')
    if orig:
        parts.append('[원본]')
        parts.append(orig)
        parts.append("")

    # 추상 요약
    abstracted = (ql.get('ABSTRACTED_SUMMARY_TEXT') or '').strip()
    if abstracted:
        parts.append('[추상 요약]')
        parts.append(abstracted)
        parts.append("")
        
    # 추출 요약 (새로 추가)
    extracted = (ql.get('EXTRACTED_SUMMARY_TEXT') or '').strip()
    if extracted:
        parts.append('[추출 요약]')
        parts.append(extracted)
        parts.append("")

    # Q&A 섹션은 의도적으로 제외합니다.

    return '\n'.join(parts)

def create_enhanced_metadata(data: dict, file_name: str, idx: int) -> dict:
    """데이터에서 메타데이터를 생성합니다."""
    jng = data.get('JNG_INFO', {}) or {}
    attr = data.get('ATTRB_INFO', {}) or {}
    
    md = {
        'brand_name': jng.get('BRAND_NM', ''),
        'headquarters_name': jng.get('JNGHDQRTRS_CONM_NM', ''),
        'source_file': file_name,
        'index_in_file': idx,
        # 필요에 따라 다른 메타데이터 추가 가능
    }
    return md

def process_files_in_batches(file_paths, batch_size=10):
    """파일 목록을 작은 배치로 나누어 처리하는 제너레이터"""
    for i in range(0, len(file_paths), batch_size):
        yield file_paths[i:i + batch_size]

def main():
    """메인 실행 함수"""
    
    # 1. 임베딩 모델 로드
    print(f"임베딩 모델 '{MODEL_NAME}'을 로딩합니다...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={
            'batch_size': BATCH_SIZE,
            'normalize_embeddings': True,
        }
    )

    # 2. Qdrant 벡터 DB 초기화
    print(f"\nQdrant 벡터 스토어를 '{SAVE_DIR}' 경로에 생성합니다...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 처음 한 번만 DB를 생성하고, 이후에는 계속 추가(add_documents)합니다.
    vectorstore = Qdrant.from_texts(
        texts=["초기화용 텍스트입니다."], # DB를 생성하기 위한 임시 텍스트
        embedding=embedding_model,
        path=SAVE_DIR,
        collection_name=COLLECTION_NAME,
        force_recreate=True, # 기존 DB가 있으면 덮어씁니다.
    )
    print("벡터 스토어 초기화 완료.")

    # 3. JSON 파일 목록 가져오기
    json_files = glob(f"{TRAIN_PATH}/**/*.json", recursive=True)
    print(f"총 {len(json_files)}개의 JSON 파일을 처리합니다.")

    # 4. 파일을 작은 배치로 나누어 점진적으로 처리 (메모리 문제 해결!)
    file_batches = process_files_in_batches(json_files, batch_size=5) # 한 번에 5개 파일씩 처리

    for file_batch in tqdm(file_batches, desc="전체 파일 배치 처리 중"):
        batch_documents = []
        for file_path in file_batch:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        data_list = [data_list]
                    
                    for idx, data in enumerate(data_list):
                        content = create_comprehensive_document(data)
                        if content.strip():
                            metadata = create_enhanced_metadata(data, os.path.basename(file_path), idx)
                            batch_documents.append(Document(page_content=content, metadata=metadata))
                except Exception as e:
                    print(f"\n파일 처리 오류 {file_path}: {e}")
                    continue
        
        # 현재 배치의 문서들을 DB에 추가합니다.
        if batch_documents:
            vectorstore.add_documents(batch_documents)

    print(f"\nQdrant 벡터 스토어 생성 및 저장 완료.")
    print(f"저장 위치: {SAVE_DIR}")
    print(f"컬렉션 이름: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()
