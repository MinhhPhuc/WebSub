import os
import tempfile
import subprocess
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deep_translator import GoogleTranslator

app = FastAPI(title="SubEasy AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình API Key của Groq
import os
api_key = os.getenv("GROQ_API_KEY")

def format_srt_time(seconds: float) -> str:
    millis = int((seconds % 1) * 1000)
    secs = int(seconds)
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

class SubtitleItem(BaseModel):
    id: int
    startTime: str
    endTime: str
    startSec: float
    endSec: float
    originalText: str
    translatedText: Optional[str] = ""
    words: Optional[List[dict]] = []

class TranslateRequest(BaseModel):
    subtitles: List[SubtitleItem]
    target_lang: str

# Cho phép Backend tự động phục vụ file HTML khi truy cập link web
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), max_words: int = Form(12)):
    """Bóc băng siêu tốc bằng Groq API (Mô hình whisper-large-v3)"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Chưa cấu hình GROQ_API_KEY trên máy chủ.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = { "Authorization": f"Bearer {GROQ_API_KEY}" }
        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json"
        }
        
        with open(tmp_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            res = requests.post(url, headers=headers, data=data, files=files)
        
        if res.status_code != 200:
            raise Exception(f"Lỗi Groq API: {res.text}")
            
        result = res.json()
        segments = result.get("segments", [])
        
        subtitles = []
        sub_id = 1
        
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            if not text: continue
            
            # Chia nhỏ câu dựa trên giới hạn max_words
            words = text.split()
            if len(words) <= max_words:
                subtitles.append({
                    "id": sub_id, "startTime": format_srt_time(start), "endTime": format_srt_time(end),
                    "startSec": round(start, 3), "endSec": round(end, 3),
                    "originalText": text, "translatedText": "", "words": []
                })
                sub_id += 1
            else:
                duration = end - start
                time_per_word = duration / len(words)
                
                for i in range(0, len(words), max_words):
                    chunk = words[i:i+max_words]
                    chunk_text = " ".join(chunk)
                    chunk_start = start + (i * time_per_word)
                    chunk_end = start + ((i + len(chunk)) * time_per_word)
                    
                    subtitles.append({
                        "id": sub_id, "startTime": format_srt_time(chunk_start), "endTime": format_srt_time(chunk_end),
                        "startSec": round(chunk_start, 3), "endSec": round(chunk_end, 3),
                        "originalText": chunk_text, "translatedText": "", "words": []
                    })
                    sub_id += 1

        return {"subtitles": subtitles}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)


@app.post("/api/translate")
async def translate_subtitles(req: TranslateRequest):
    try:
        translator = GoogleTranslator(source='auto', target=req.target_lang)
        for item in req.subtitles:
            if item.originalText.strip():
                item.translatedText = translator.translate(item.originalText)
        return {"subtitles": [s.dict() for s in req.subtitles]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dịch thuật: {str(e)}")


@app.post("/api/render-video")
async def render_video(
    video_file: UploadFile = File(...), srt_content: str = Form(""),
    has_mask: bool = Form(False), mask_x: float = Form(0.0), mask_y: float = Form(0.0),
    mask_w: float = Form(0.0), mask_h: float = Form(0.0)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(await video_file.read())
        input_vid_path = tmp_vid.name

    output_vid_path = input_vid_path.replace(".mp4", "_rendered.mp4")
    srt_path = input_vid_path.replace(".mp4", ".srt")

    try:
        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)

        filters = []
        if has_mask and mask_w > 0 and mask_h > 0:
            filters.append(f"drawbox=x=iw*{mask_x}:y=ih*{mask_y}:w=iw*{mask_w}:h=ih*{mask_h}:color=black@1:t=fill")
        
        if srt_content.strip():
            escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
            filters.append(f"subtitles='{escaped_srt}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,BorderStyle=1,Outline=2'")

        vf_chain = ",".join(filters) if filters else "null"
        cmd = ["ffmpeg", "-y", "-i", input_vid_path, "-vf", vf_chain, "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "copy", output_vid_path]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0: raise HTTPException(status_code=500, detail="Lỗi Render")

        return FileResponse(output_vid_path, media_type="video/mp4", filename=f"rendered_{video_file.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_vid_path): os.remove(input_vid_path)
        if os.path.exists(srt_path): os.remove(srt_path)