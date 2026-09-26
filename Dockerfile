FROM python:3.10-slim

# Cài đặt FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# BẮT BUỘC TRÊN HUGGING FACE: Tạo user non-root để có quyền đọc/ghi file video tạm
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy toàn bộ code vào container và cấp quyền sở hữu cho 'user'
COPY --chown=user . /app

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng 7860 (Hugging Face yêu cầu)
EXPOSE 7860

# Chạy FastAPI trên cổng 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]