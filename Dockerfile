# 1. 파이썬 3.11 버전을 기반으로 서버 환경을 시작합니다.
FROM python:3.11-slim

# 2. 컨테이너 내부에 작업 공간(/app)을 만듭니다.
WORKDIR /app

# 3. 필요한 파이썬 라이브러리 목록을 먼저 복사하고 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 로컬 Qdrant DB 폴더 전체를 이미지 안으로 복사합니다. (가장 중요!)
COPY ./qdrant ./qdrant

# 5. FastAPI 소스 코드를 복사합니다.
COPY main_inference.py .

# 6. 서버를 실행하는 명령어를 지정합니다.
CMD ["uvicorn", "main_inference:app", "--host", "0.0.0.0", "--port", "8000"]