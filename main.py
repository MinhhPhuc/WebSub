import os
import tempfile
import subprocess
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from fastapi.responses import HTMLResponse

app = FastAPI(title="SubEasy AI Studio API")

# Cấu hình CORS cho phép Frontend tương tác không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải mô hình Whisper (Dùng 'base' hoặc 'small' để đạt tốc độ bóc băng nhanh)
print("⏳ Đang tải mô hình Faster-Whisper...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Mô hình Whisper đã sẵn sàng!")


def format_srt_time(seconds: float) -> str:
    """Chuyển đổi giây sang định dạng chuẩn SRT (HH:MM:SS,mmm)"""
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
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    max_words: int = Form(12)
):
    """API Bóc băng phụ đề tốc độ cao với Word Timestamps"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, _ = whisper_model.transcribe(
            tmp_path,
            word_timestamps=True,
            vad_filter=True,  # Bỏ qua khoảng im lặng
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        subtitles = []
        sub_id = 1

        for segment in segments:
            current_words = []
            words_list = getattr(segment, "words", []) or []

            if not words_list:
                subtitles.append({
                    "id": sub_id,
                    "startTime": format_srt_time(segment.start),
                    "endTime": format_srt_time(segment.end),
                    "startSec": round(segment.start, 3),
                    "endSec": round(segment.end, 3),
                    "originalText": segment.text.strip(),
                    "translatedText": "",
                    "words": []
                })
                sub_id += 1
                continue

            for word in words_list:
                clean_w = word.word.strip()
                if clean_w:
                    current_words.append({
                        "word": clean_w,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3)
                    })

                # Gom nhóm từ theo max_words
                if len(current_words) >= max_words:
                    start_sec = current_words[0]["start"]
                    end_sec = current_words[-1]["end"]
                    text_str = " ".join([w["word"] for w in current_words])

                    subtitles.append({
                        "id": sub_id,
                        "startTime": format_srt_time(start_sec),
                        "endTime": format_srt_time(end_sec),
                        "startSec": start_sec,
                        "endSec": end_sec,
                        "originalText": text_str,
                        "translatedText": "",
                        "words": current_words
                    })
                    sub_id += 1
                    current_words = []

            if current_words:
                start_sec = current_words[0]["start"]
                end_sec = current_words[-1]["end"]
                text_str = " ".join([w["word"] for w in current_words])

                subtitles.append({
                    "id": sub_id,
                    "startTime": format_srt_time(start_sec),
                    "endTime": format_srt_time(end_sec),
                    "startSec": start_sec,
                    "endSec": end_sec,
                    "originalText": text_str,
                    "translatedText": "",
                    "words": current_words
                })
                sub_id += 1

        return {"subtitles": subtitles}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi bóc băng: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/translate")
async def translate_subtitles(req: TranslateRequest):
    """API AI Dịch thuật tự động an toàn (Fallback per item)"""
    try:
        translator = GoogleTranslator(source='auto', target=req.target_lang)
        subtitles = req.subtitles

        for sub in subtitles:
            text = sub.originalText.strip()
            if text:
                try:
                    sub.translatedText = translator.translate(text)
                except Exception:
                    sub.translatedText = text
            else:
                sub.translatedText = ""

        return {"subtitles": [s.dict() for s in subtitles]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dịch thuật: {str(e)}")


@app.post("/api/render-video")
async def render_video(
    video_file: UploadFile = File(...),
    srt_content: str = Form(""),
    has_mask: bool = Form(False),
    mask_x: float = Form(0.0),
    mask_y: float = Form(0.0),
    mask_w: float = Form(0.0),
    mask_h: float = Form(0.0)
):
    """API Render Video chuẩn FFmpeg: Giữ 100% FPS gốc, đè ô che phụ đề cũ + dán phụ đề mới"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(await video_file.read())
        input_vid_path = tmp_vid.name

    output_vid_path = input_vid_path.replace(".mp4", "_rendered.mp4")
    srt_path = input_vid_path.replace(".mp4", ".srt")

    try:
        # Ghi file phụ đề SRT tạm
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        filters = []

        # 1. Vẽ ô đen che phụ đề gốc theo tỷ lệ %
        if has_mask and mask_w > 0 and mask_h > 0:
            drawbox_filter = (
                f"drawbox=x=iw*{mask_x}:y=ih*{mask_y}:"
                f"w=iw*{mask_w}:h=ih*{mask_h}:"
                f"color=black@1:t=fill"
            )
            filters.append(drawbox_filter)

        # 2. Burn phụ đề mới lên Video
        if srt_content.strip():
            escaped_srt_path = srt_path.replace("\\", "/").replace(":", "\\:")
            subtitles_filter = (
                f"subtitles='{escaped_srt_path}':"
                f"force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,BorderStyle=1,Outline=2'"
            )
            filters.append(subtitles_filter)

        vf_chain = ",".join(filters) if filters else "null"

        # Lệnh FFmpeg mã hóa mượt mà (CRF 18 + `-c:a copy` giữ âm thanh gốc)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_vid_path,
            "-vf", vf_chain,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-c:a", "copy",
            output_vid_path
        ]

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            print("FFmpeg Error Log:", process.stderr)
            raise HTTPException(status_code=500, detail="Lỗi khi FFmpeg Render Video")

        return FileResponse(
            output_vid_path,
            media_type="video/mp4",
            filename=f"rendered_{video_file.filename}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_vid_path): os.remove(input_vid_path)
        if os.path.exists(srt_path): os.remove(srt_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)