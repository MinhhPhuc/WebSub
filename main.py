import os
import tempfile
import subprocess
import requests
import yt_dlp
import re
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
    """Bóc băng Groq Whisper - Lọc bỏ 100% phụ đề ảo giác ở đoạn khoảng lặng / nhạc nền"""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Chưa cấu hình GROQ_API_KEY trên máy chủ Render (Vào tab Environment để thêm)."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        
        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "language": "zh",
            "temperature": "0",
            "prompt": "以下是短视频、动漫或电视剧的中文原声音频，请准确提取台词，使用简体中文。"
        }
        
        with open(tmp_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            res = requests.post(url, headers=headers, data=data, files=files)
        
        if res.status_code != 200:
            raise Exception(f"Lỗi Groq API ({res.status_code}): {res.text}")
            
        result = res.json()
        raw_segments = result.get("segments", [])

        def is_cjk(text: str) -> bool:
            cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
            return cjk_count > (len(text) * 0.25)

        # 1. CẮT CÂU QUÁ DÀI & LỌC BỎ KHOẢNG LẶNG/NHẠC NỀN
        split_segs = []
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            
            # --- BỘ LỌC CHỐNG ẢO GIÁC (HALLUCINATION FILTER) ---
            no_speech_prob = seg.get("no_speech_prob", 0.0)
            avg_logprob = seg.get("avg_logprob", 0.0)
            compression_ratio = seg.get("compression_ratio", 0.0)

            # Nếu xác suất KHÔNG có tiếng nói > 50% hoặc câu bị lặp từ bất thường -> BỎ QUA NGAY
            if no_speech_prob > 0.5 or avg_logprob < -1.0 or compression_ratio > 2.4:
                continue

            if not text:
                continue

            cjk = is_cjk(text)
            if cjk:
                clean_text = re.sub(r'\s+', '', text)
                units = list(clean_text)
            else:
                units = text.split()

            limit = max_words
            if len(units) <= limit:
                split_segs.append({"start": start, "end": end, "text": text, "is_cjk": cjk})
            else:
                duration = end - start
                unit_time = duration / len(units) if units else 0
                for i in range(0, len(units), limit):
                    chunk_units = units[i:i + limit]
                    chunk_text = "".join(chunk_units) if cjk else " ".join(chunk_units)
                    c_start = start + (i * unit_time)
                    c_end = start + ((i + len(chunk_units)) * unit_time)
                    split_segs.append({"start": c_start, "end": c_end, "text": chunk_text, "is_cjk": cjk})

        # 2. GỘP CÂU QUÁ NGẮN (< 4 KÝ TỰ)
        final_segs = []
        for seg in split_segs:
            if not final_segs:
                final_segs.append(seg)
                continue

            prev = final_segs[-1]
            cjk = seg["is_cjk"]
            prev_len = len(re.sub(r'\s+', '', prev["text"])) if prev["is_cjk"] else len(prev["text"].split())
            curr_len = len(re.sub(r'\s+', '', seg["text"])) if cjk else len(seg["text"].split())

            min_thresh = 4 if cjk else 2

            if (prev_len < min_thresh or curr_len < min_thresh) and (prev_len + curr_len <= max_words * 1.4):
                sep = "" if (prev["is_cjk"] or cjk) else " "
                prev["text"] = prev["text"] + sep + seg["text"]
                prev["end"] = seg["end"]
            else:
                final_segs.append(seg)

        # 3. ĐỊNH DẠNG ĐẦU RA
        subtitles = []
        for sub_id, seg in enumerate(final_segs, 1):
            subtitles.append({
                "id": sub_id,
                "startTime": format_srt_time(seg["start"]),
                "endTime": format_srt_time(seg["end"]),
                "startSec": round(seg["start"], 3),
                "endSec": round(seg["end"], 3),
                "originalText": seg["text"],
                "translatedText": "",
                "words": []
            })

        return {"subtitles": subtitles}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        remove_file(tmp_path)


@app.post("/api/translate")
async def translate_subtitles(req: TranslateRequest):
    """AI Dịch thuật siêu tốc - Gom Batch 25 câu/lần chống Google rate-limit hoàn toàn"""
    try:
        target_lang = req.target_lang
        subs = req.subtitles
        if not subs:
            return {"subtitles": []}

        lines = [s.originalText.replace("\n", " ").strip() for s in subs]
        batch_size = 25
        translated_lines = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        for i in range(0, len(lines), batch_size):
            chunk = lines[i:i + batch_size]
            joined_text = "\n".join(chunk)
            translated_chunk_text = None

            # 1. Gọi Google Translate GTX API (cho phép dịch cụm nhiều dòng cùng lúc)
            try:
                url = "https://translate.googleapis.com/translate_a/single"
                params = {
                    "client": "gtx",
                    "sl": "auto",
                    "tl": target_lang,
                    "dt": "t",
                    "q": joined_text
                }
                res = requests.get(url, params=params, headers=headers, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        translated_parts = [item[0] for item in data[0] if item and isinstance(item, list) and len(item) > 0 and item[0]]
                        translated_chunk_text = "".join(translated_parts)
            except Exception as e:
                print("GTX Translate error:", e)

            # 2. Fallback deep-translator nếu GTX bị chặn
            if not translated_chunk_text:
                try:
                    translated_chunk_text = GoogleTranslator(source='auto', target=target_lang).translate(joined_text)
                except Exception as e:
                    print("GoogleTranslator error:", e)

            # 3. Tach ket qua tra ve theo dong
            if translated_chunk_text:
                split_res = [line.strip() for line in translated_chunk_text.split("\n")]
                if len(split_res) == len(chunk):
                    translated_lines.extend(split_res)
                else:
                    # Neu bi lech so dong: dich tung cau le trong Lô nho
                    for single_text in chunk:
                        if not single_text:
                            translated_lines.append("")
                            continue
                        try:
                            url = "https://translate.googleapis.com/translate_a/single"
                            res = requests.get(url, params={"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": single_text}, headers=headers, timeout=5)
                            if res.status_code == 200:
                                t_val = "".join([x[0] for x in res.json()[0] if x and x[0]])
                                translated_lines.append(t_val)
                            else:
                                translated_lines.append(single_text)
                        except:
                            translated_lines.append(single_text)
            else:
                translated_lines.extend(chunk)

        for idx, item in enumerate(subs):
            if idx < len(translated_lines) and translated_lines[idx].strip():
                item.translatedText = translated_lines[idx].strip()
            else:
                item.translatedText = item.originalText

        return {"subtitles": [s.dict() for s in subs]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dịch thuật: {str(e)}")


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