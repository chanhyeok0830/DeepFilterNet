# FastAPI 프레임워크에서 필요한 기능들을 불러온다.
# - FastAPI: 웹 서버의 핵심을 만드는 데 사용
# - File, UploadFile: 파일 업로드를 처리할 때 사용
# - HTTPException: 에러 발생 시 HTTP 상태코드를 함께 반환할 수 있게 해줌
from fastapi import FastAPI, File, UploadFile, HTTPException

# 클라이언트에게 파일을 응답으로 보낼 수 있게 해주는 도구 (예: 다운로드용)
from fastapi.responses import FileResponse

# os: 파일 존재 여부 확인에 사용됨
# shutil: 업로드된 파일을 저장할 때 파일을 복사하는 데 사용
import os
import shutil

# DeepFilterNet 라이브러리
from df import enhance, init_df 


# torchaudio: 파이토치의 오디오 라이브러리
import torchaudio

# CORS
from fastapi.middleware.cors import CORSMiddleware  


# FastAPI 웹 서버 시작작
app = FastAPI()


# CORS 허용 설정 (프론트에서 요청 가능한 출처 명시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],  # 프론트엔드가 띄워지는 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepFilterNet 모델 초기화
model, df_state, _ = init_df()

# 오디오 파일의 경로를 변수로 지정함
# - 업로드된 오디오는 input.wav로 저장
# - 노이즈 제거 후 결과는 output.wav로 저장할 예정
UPLOAD_PATH = "input.wav"     # 입력 파일
CLEAN_PATH = "output.wav"     # 출력 파일


# /upload 주소로 post 요청시
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 input.wav라는 이름으로 저장함
        with open(UPLOAD_PATH, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 노이즈 처리 과정
        waveform, sr = torchaudio.load(UPLOAD_PATH)
        clean = enhance(model, df_state, waveform)
        torchaudio.save(CLEAN_PATH, clean, sample_rate=sr)

        return {"detail": "처리 완료됨!"}
    
    # 만약 위 과정 중 오류가 발생하면, HTTP 500 에러와 함께
    # 에러 메시지를 클라이언트에게 전달함
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# /download 주소로 GET 요청시
@app.get("/download")
async def download_file():
    # output.wav 파일이 존재하지 않을 경우
    # 404 에러를 클라이언트에 응답함
    if not os.path.exists(CLEAN_PATH):
        raise HTTPException(status_code=404, detail="output.wav 파일이 아직 없음!")
    
    # 파일이 존재하면, FastAPI의 FileResponse를 이용해
    # 클라이언트에게 output.wav 파일을 audio/wav 형식으로 다운로드하게 전달함
    return FileResponse(path=CLEAN_PATH, media_type="audio/wav", filename="output.wav")
