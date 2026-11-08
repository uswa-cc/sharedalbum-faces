# SharedAlbum face server for Render (free web service, Docker)
FROM python:3.11-slim

# Compilers are needed to install insightface; libglib is needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a normal user, not root
RUN useradd -m -u 1000 user
USER user
ENV FACE_MODEL=buffalo_s \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Download the face model now, so the first photo isn't slow
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_s', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider']).prepare(ctx_id=-1)"

COPY --chown=user server.py .

# Render tells the app which port to use in $PORT
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
