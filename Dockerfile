# birdlisten image. Based on fleet/templates/app/Dockerfile with a few changes:
# Python 3.11 (tflite-runtime has no 3.12 wheel), ffmpeg from Debian, and an
# optional collage port (SERVE_PORT, see README).
FROM python:3.11-slim
# ffmpeg pulls the RTSP audio; libsndfile is librosa's WAV reader; tzdata
# gives zoneinfo the IANA zones for TZ. All come from Debian so uv.lock stays
# pure-Python. The apt cache is cleared in the
# same layer so it never lands in the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 tzdata \
 && apt-get clean \
 && find /var/lib/apt/lists -mindepth 1 -delete
# uv from its official image: reproducible, no curl|sh.
# 0.11 matches the uv that writes uv.lock (revision 3).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
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
# The collage decodes Fugleramme's WebP plates with Pillow's bundled libwebp
# and draws labels with its bundled FreeType. Fail the build if a future wheel
# drops either, instead of shipping an image that renders only placeholders.
RUN /app/.venv/bin/python -c "import PIL.features as f; assert f.check('webp') and f.check('freetype2')"
# The pop-up's local times need the zone database from tzdata above.
RUN /app/.venv/bin/python -c "import zoneinfo; zoneinfo.ZoneInfo('America/Los_Angeles')"
# Then the code, including fonts/ (the collage's serif, also served to the
# pop-up), audubon.json (the Audubon plate map), facts.py, taxa.py (the
# non-bird label list) and static/ (the
# page's script and style); .dockerignore keeps them in. Fail here rather than
# ship an image that falls back to the sans, placeholder cards or a dead page.
COPY . .
RUN test -f fonts/LibreBaskerville.ttf && test -f fonts/LibreBaskerville-Italic.ttf && test -f fonts/OFL.txt \
 && test -f audubon.json && test -f static/page.js && test -f static/page.css && test -f facts.py && test -f taxa.py
# /data holds the SQLite db, optional clips, and the artwork cache; compose
# mounts a volume there.
RUN useradd --create-home --uid 10001 app && mkdir /data && chown -R app:app /app /data
USER app
VOLUME /data
# Documentation only: the collage server listens on whatever SERVE_PORT says
# (and not at all when it is unset). Publish the port in compose.
EXPOSE 8085
ENV PATH="/app/.venv/bin:$PATH"
# loop.py calls main(). In stream mode (the default) main() runs until the
# container stops; with CAPTURE_MODE=roundrobin each call is one pass over
# every camera (LOOP_INTERVAL=1 runs them back to back).
CMD ["python", "loop.py"]
