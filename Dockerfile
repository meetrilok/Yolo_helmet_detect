# ── Stage 1 : runtime image ───────────────────────────────────────────
FROM python:3.11-slim AS runtime

# — system deps: libGL for OpenCV, libglib for jpeg/png, ffmpeg for video —
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# set workdir
WORKDIR /app

# copy Python deps first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code & model
COPY app.py detector.py video_detect.py ./
COPY model/ ./model
COPY templates/ ./templates

# expose the Flask port
EXPOSE 5000

# runtime env (production: disable debug / enable threading)
ENV FLASK_APP=app.py \
    FLASK_ENV=production

# entrypoint – use gunicorn for robustness (single worker ok for demo)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "app:app"]
