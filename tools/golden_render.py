"""Print the collage golden hashes used by test_frame.test_render_matches_golden.

Run once, before frame.py changes, and paste the output into test_frame.py:
  uv run --no-project --python 3.11 --with pillow==12.3.0 --with pytest==8.3.4 python tools/golden_render.py
Never regenerate the constants from changed rendering code.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import frame  # noqa: E402
from test_frame import golden_renders  # noqa: E402


def refuse(url, timeout=None):
    raise frame.NotFound(url)


def main() -> None:
    frame.fetch_url = refuse
    with tempfile.TemporaryDirectory() as tmp:
        renders = golden_renders(Path(tmp))
    px, raw = {}, {}
    for k, png in renders.items():
        px[k] = hashlib.sha256(Image.open(io.BytesIO(png)).convert("RGB").tobytes()).hexdigest()
        raw[k] = hashlib.sha256(png).hexdigest()
    print("GOLDEN_PX = {")
    for k, v in px.items():
        print(f"    {k!r}: {v!r},")
    print("}")
    print("GOLDEN_RAW = {")
    for k, v in raw.items():
        print(f"    {k!r}: {v!r},")
    print("}")


if __name__ == "__main__":
    main()
