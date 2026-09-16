# birdlisten image. Based on fleet/templates/app/Dockerfile with two changes:
# Python 3.11 (tflite-runtime has no 3.12 wheel) and ffmpeg from Debian.

FROM python:3.11-slim

# ffmpeg pulls the RTSP audio; libsndfile is librosa's WAV reader. Both come
# from Debian so uv.lock stays pure-Python. The apt cache is cleared in the
# same layer so it never lands in the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && apt-get clean \
 && find /var/lib/apt/lists -mindepth 1 -delete

# uv from its official image: reproducible, no curl|sh.
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first so this layer caches across code-only changes. With
# [tool.uv] package = false the project itself is never installed, so this
# single sync is the whole environment; the code just needs to be present.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the code.
COPY . .

# /data holds the SQLite db and optional clips; compose mounts a volume there.
RUN useradd --create-home --uid 10001 app && mkdir /data && chown -R app:app /app /data
USER app
VOLUME /data

ENV PATH="/app/.venv/bin:$PATH"

# loop.py: run main() back to back (LOOP_INTERVAL=1); each call is one pass
# over every camera.
CMD ["python", "loop.py"]
