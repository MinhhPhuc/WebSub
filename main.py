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
async def download_douyin(background_tasks: BackgroundTasks, link: str = Form(...), is_direct: str = Form("false")):
    """Xử lý tải video. Hỗ trợ tải MP4 trực tiếp hoặc fallback API thông minh"""
    tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        # 1. NẾU TRÌNH DUYỆT ĐÃ BÓC ĐƯỢC LINK MP4 (Bỏ qua yt-dlp hoàn toàn -> Không bao giờ dính lỗi Cookie)
        if is_direct == 'true' or '.mp4' in link or 'douyinvod.com' in link or 'tiktokcdn.com' in link:
            res = requests.get(link, stream=True, headers=headers, timeout=30)
            res.raise_for_status()
            with open(tmp_vid, 'wb') as f:
                for chunk in res.iter_content(chunk_size=32768):
                    if chunk: f.write(chunk)
            
            background_tasks.add_task(remove_file, tmp_vid)
            return FileResponse(tmp_vid, media_type="video/mp4", filename="downloaded_video.mp4")

        # 2. FALLBACK TẠI SERVER TRONG TRƯỜNG HỢP XẤU NHẤT (Chạy Cobalt API)
        video_url = None
        try:
            cobalt_req = requests.post(
                "https://api.cobalt.tools/",
                json={"url": link},
                headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": headers["User-Agent"]},
                timeout=10
            )
            if cobalt_req.status_code == 200:
                c_data = cobalt_req.json()
                if c_data.get("status") in ["stream", "redirect", "picker"]:
                    video_url = c_data.get("url")
        except:
            pass

        if video_url:
            res = requests.get(video_url, stream=True, headers=headers, timeout=30)
            res.raise_for_status()
            with open(tmp_vid, 'wb') as f:
                for chunk in res.iter_content(chunk_size=32768):
                    if chunk: f.write(chunk)
        else:
            # 3. YOUTUBE, FACEBOOK HOẶC TẦNG CUỐI CÙNG yt-dlp
            ydl_opts = {'format': 'best', 'outtmpl': tmp_vid, 'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([link])

        background_tasks.add_task(remove_file, tmp_vid)
        return FileResponse(tmp_vid, media_type="video/mp4", filename="downloaded_video.mp4")

    except Exception as e:
        remove_file(tmp_vid)
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)} (Douyin đã thay đổi thuật toán. Bạn vui lòng tải file video MP4 về máy rồi dùng nút 'Tải Video Từ Máy')")

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
async def render_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    srt_content: str = Form(""),
    has_mask: bool = Form(False),
    mask_x: float = Form(0.0),
    mask_y: float = Form(0.0),
    mask_w: float = Form(0.0),
    mask_h: float = Form(0.0)
):
    """Render Video bằng FFmpeg - Đã tối ưu chống tràn RAM 512MB trên Render"""
    import shutil
    tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    output_vid_path = tmp_vid.replace(".mp4", "_rendered.mp4")
    srt_path = tmp_vid.replace(".mp4", ".srt")

    try:
        # 1. TIẾT KIỆM RAM: Dùng shutil stream trực tiếp file vào ổ đĩa thay vì đọc toàn bộ vào RAM
        with open(tmp_vid, 'wb') as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        filters = []
        # Nếu bật che phụ đề gốc (có khung đen kéo thả)
        if has_mask and mask_w > 0 and mask_h > 0:
            filters.append(f"drawbox=x=iw*{mask_x}:y=ih*{mask_y}:w=iw*{mask_w}:h=ih*{mask_h}:color=black@1:t=fill")
        
        # In phụ đề mới lên
        if srt_content.strip():
            escaped_srt = srt_path.replace(os.sep, '/').replace(':', '\\:')
            filters.append(f"subtitles='{escaped_srt}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,BorderStyle=1,Outline=2'")

        vf_chain = ",".join(filters) if filters else "null"
        
        # 2. CẤU HÌNH SIÊU NHẸ CHO RENDER: preset ultrafast, giới hạn 2 threads để không bao giờ tràn 512MB RAM
        cmd = [
            "ffmpeg", "-y", "-i", tmp_vid,
            "-vf", vf_chain,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "copy",
            "-threads", "2",
            output_vid_path
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print("FFmpeg Error:", proc.stderr)
            raise HTTPException(status_code=500, detail=f"Lỗi FFmpeg: {proc.stderr[-200:]}")

        background_tasks.add_task(remove_file, tmp_vid)
        background_tasks.add_task(remove_file, output_vid_path)
        background_tasks.add_task(remove_file, srt_path)

        return FileResponse(output_vid_path, media_type="video/mp4", filename="rendered_video.mp4")
    except Exception as e:
        remove_file(tmp_vid)
        remove_file(output_vid_path)
        remove_file(srt_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)