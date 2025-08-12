# migrate_data.py
print("--- 스크립트 실행 시작 ---") # 디버깅 체크포인트 1

try:
    from qdrant_client import QdrantClient, models
    from tqdm import tqdm
    print("--- 라이브러리 임포트 성공 ---") # 디버깅 체크포인트 2
except ImportError as e:
    print(f"!!! 라이브러리 임포트 실패: {e}")
    print("!!! 'pip install qdrant-client tqdm'를 실행했는지 확인해주세요.")
    exit()


# --- 설정 영역 ---
# 로컬 DB 폴더 경로
SOURCE_DB_PATH = "./qdrant" 
# 1단계에서 띄운 로컬 서버 주소
TARGET_SERVER_URL = "http://localhost:6333"
# 컬렉션 이름 (기존과 동일해야 함)
COLLECTION_NAME = "qdrant-franchise-db"
# -----------------

def migrate():
    # 1. 원본(로컬 폴더) DB에 연결
    print(f"'{SOURCE_DB_PATH}'에서 원본 DB를 읽습니다...")
    source_client = QdrantClient(path=SOURCE_DB_PATH)

    # 2. 타겟(로컬 서버) DB에 연결
    print(f"'{TARGET_SERVER_URL}'의 타겟 서버에 연결합니다...")
    # 서버 응답을 60초까지 기다리도록 설정하여 타임아웃 오류를 방지합니다.
    target_client = QdrantClient(url=TARGET_SERVER_URL, timeout=60)

    # 3. 타겟 서버에 컬렉션 생성
    print(f"타겟 서버에 '{COLLECTION_NAME}' 컬렉션을 생성합니다...")
    # recreate_collection 대신, 컬렉션 존재 여부를 확인하고 없으면 새로 생성하는 방식을 사용합니다.
    try:
        if not target_client.collection_exists(collection_name=COLLECTION_NAME):
            target_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
            print(f"'{COLLECTION_NAME}' 컬렉션을 새로 생성했습니다.")
        else:
            print(f"'{COLLECTION_NAME}' 컬렉션이 이미 존재하여 생성하지 않았습니다.")
    except Exception as e:
        print(f"!!! 컬렉션 확인/생성 중 오류 발생: {e}")
        return # 오류 발생 시 함수 종료


    # 4. 원본 DB에서 모든 데이터를 읽어와서 타겟 서버로 업로드
    print("데이터 이사를 시작합니다... (데이터 양에 따라 시간이 걸릴 수 있습니다)")
    
    all_points_to_upload = []
    next_offset = None
    
    # scroll API로 읽어온 Record 객체를 PointStruct 객체로 변환해줍니다.
    with tqdm(total=source_client.get_collection(collection_name=COLLECTION_NAME).points_count, desc="Reading points") as pbar:
        while True:
            records, next_offset = source_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256, # 한 번에 256개씩 읽기
                with_payload=True,
                with_vectors=True,
                offset=next_offset
            )
            
            for record in records:
                # Record 형식에서 PointStruct 형식으로 데이터를 옮겨 담습니다.
                all_points_to_upload.append(
                    models.PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=record.payload
                    )
                )
            pbar.update(len(records))

            if next_offset is None:
                break
    
    # --- 수정된 부분 ---
    # 타겟 서버에 데이터를 작은 배치(batch)로 나누어 업로드합니다.
    print("\nTarget server에 데이터 업로드를 시작합니다...")
    batch_size = 128
    for i in tqdm(range(0, len(all_points_to_upload), batch_size), desc="Uploading batches"):
        batch = all_points_to_upload[i:i + batch_size]
        target_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True
        )
    # ------------------

    print("\n🎉 데이터 이사가 성공적으로 완료되었습니다!")
    final_info = target_client.get_collection(collection_name=COLLECTION_NAME)
    print(f"최종 컬렉션의 포인트 수: {final_info.points_count}")


print(f"--- 현재 파일(__name__): {__name__} ---")

if __name__ == "__main__":
    print("--- 메인 실행 블록 진입 ---")
    migrate()
    print("--- 스크립트 실행 완료 ---") 
else:
    print("--- 메인 실행 블록을 건너뛰었습니다 (모듈로 임포트됨) ---")
