from pathlib import Path
import json
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.responses import FileResponse

from src.agent.agent import run_agent


app = FastAPI(title="AI Assistant API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/app/Data/Raw Data/Uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "AI Assistant API is running"
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "saved_to": str(file_path)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/run-task")
def run_task(
    task: str = Form(...),
    file_paths: str = Form("[]"),
    job_title: str | None = Form(None),
    required_skills: str | None = Form(None),
    shortlist_threshold: int | None = Form(None),
):
    try:
        paths = json.loads(file_paths.replace("\\", "/"))

        result = run_agent(
            task=task,
            file_path=paths,
            job_title=job_title,
            required_skills=(
                [skill.strip() for skill in required_skills.split(",")]
                if required_skills
                else None
            ),
            shortlist_threshold=shortlist_threshold,
        )

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "output_file": None
        }


OUTPUT_DIR = Path("/app/Data/Outputs").resolve()


@app.get("/download")
def download(path: str):
    file_path = Path(path).resolve()
    # only allow files inside the Outputs folder
    if OUTPUT_DIR not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=file_path.name)   