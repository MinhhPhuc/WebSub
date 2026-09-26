import os
import tempfile
import subprocess
import requests
import yt_dlp
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Chưa tìm thấy file index.html</h1>"

# Hàm dọn dẹp file rác
def remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


@app.post("/api/download-douyin")
async def download_douyin(background_tasks: BackgroundTasks, link: str = Form(...)):
    """Tải video từ Douyin, TikTok, YouTube qua link"""
    tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    ydl_opts = {
        'format': 'best',
        'outtmpl': tmp_vid,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
            
        # Lập lịch xóa file này sau khi đã gửi xong cho Frontend
        background_tasks.add_task(remove_file, tmp_vid)
        return FileResponse(tmp_vid, media_type="video/mp4", filename="downloaded_video.mp4")
    except Exception as e:
        remove_file(tmp_vid)
        raise HTTPException(status_code=500, detail=f"Không thể tải video từ link này: {str(e)}")


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), max_words: int = Form(12)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Chưa cấu hình GROQ_API_KEY")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        data = {"model": "whisper-large-v3", "response_format": "verbose_json"}
        
        with open(tmp_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            res = requests.post(url, headers=headers, data=data, files=files)
        
        if res.status_code != 200:
            raise Exception(f"Lỗi Groq API ({res.status_code}): {res.text}")
            
        result = res.json()
        segments = result.get("segments", [])
        
        subtitles = []
        sub_id = 1
        
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            if not text: continue
            
            if " " in text:
                words = text.split()
                if len(words) <= max_words:
                    subtitles.append({"id": sub_id, "startTime": format_srt_time(start), "endTime": format_srt_time(end), "startSec": round(start, 3), "endSec": round(end, 3), "originalText": text, "translatedText": "", "words": []})
                    sub_id += 1
                else:
                    time_per_word = (end - start) / len(words)
                    for i in range(0, len(words), max_words):
                        chunk = words[i:i + max_words]
                        chunk_text = " ".join(chunk)
                        c_start = start + (i * time_per_word)
                        c_end = start + ((i + len(chunk)) * time_per_word)
                        subtitles.append({"id": sub_id, "startTime": format_srt_time(c_start), "endTime": format_srt_time(c_end), "startSec": round(c_start, 3), "endSec": round(c_end, 3), "originalText": chunk_text, "translatedText": "", "words": []})
                        sub_id += 1
            else:
                if len(text) <= max_words:
                    subtitles.append({"id": sub_id, "startTime": format_srt_time(start), "endTime": format_srt_time(end), "startSec": round(start, 3), "endSec": round(end, 3), "originalText": text, "translatedText": "", "words": []})
                    sub_id += 1
                else:
                    time_per_char = (end - start) / len(text)
                    for i in range(0, len(text), max_words):
                        chunk_text = text[i:i + max_words]
                        c_start = start + (i * time_per_char)
                        c_end = start + ((i + len(chunk_text)) * time_per_char)
                        subtitles.append({"id": sub_id, "startTime": format_srt_time(c_start), "endTime": format_srt_time(c_end), "startSec": round(c_start, 3), "endSec": round(c_end, 3), "originalText": chunk_text, "translatedText": "", "words": []})
                        sub_id += 1
        return {"subtitles": subtitles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        remove_file(tmp_path)


@app.post("/api/translate")
async def translate_subtitles(req: TranslateRequest):
    try:
        translator = GoogleTranslator(source='auto', target=req.target_lang)
        for item in req.subtitles:
            if item.originalText.strip():
                try: item.translatedText = translator.translate(item.originalText)
                except: item.translatedText = item.originalText
        return {"subtitles": [s.dict() for s in req.subtitles]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/render-video")
async def render_video(video_file: UploadFile = File(...), srt_content: str = Form(""), has_mask: bool = Form(False), mask_x: float = Form(0.0), mask_y: float = Form(0.0), mask_w: float = Form(0.0), mask_h: float = Form(0.0)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(await video_file.read())
        input_vid_path = tmp_vid.name

    output_vid_path = input_vid_path.replace(".mp4", "_rendered.mp4")
    srt_path = input_vid_path.replace(".mp4", ".srt")

    try:
        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
        filters = []
        if has_mask and mask_w > 0 and mask_h > 0: filters.append(f"drawbox=x=iw*{mask_x}:y=ih*{mask_y}:w=iw*{mask_w}:h=ih*{mask_h}:color=black@1:t=fill")
        if srt_content.strip(): filters.append(f"subtitles='{srt_path.replace(os.sep, '/').replace(':', r'\:')}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,BorderStyle=1,Outline=2'")
        
        vf_chain = ",".join(filters) if filters else "null"
        cmd = ["ffmpeg", "-y", "-i", input_vid_path, "-vf", vf_chain, "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "copy", output_vid_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0: raise HTTPException(status_code=500, detail="Lỗi FFmpeg Render")
        
        return FileResponse(output_vid_path, media_type="video/mp4", filename=f"rendered_{video_file.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        remove_file(input_vid_path)
        remove_file(srt_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)