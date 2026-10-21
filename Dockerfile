# API image for the Document Q&A Service (Phase 5).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# Install deps first so this layer caches unless requirements change.
# psycopg[binary] ships libpq in the wheel, so no system packages are needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the running service needs.
COPY app ./app
COPY data ./data
COPY frontend ./frontend

EXPOSE 8000

# `python -m uvicorn` puts the working dir (/code) on sys.path so `app` imports.
# --proxy-headers: behind a platform proxy (Render, Cloud Run...) the real client IP
# arrives in X-Forwarded-For, which the per-IP rate limit needs. Hosts that inject
# $PORT (Render) are honored; locally it defaults to 8000.
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
