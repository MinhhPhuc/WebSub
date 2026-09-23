FROM python:3.10-slim

# Cài đặt FFmpeg và các công cụ hệ thống
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào Container
COPY . .

# Expose Cổng chạy ứng dụng
EXPOSE 8000

# Chạy FastAPI Server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]