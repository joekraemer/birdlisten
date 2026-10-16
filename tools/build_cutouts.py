# /// script
# requires-python = ">=3.12"   # numpy 2.5; the app itself stays on 3.11
# dependencies = [
#     "numpy==2.5.4",
#     "open_clip_torch==3.3.0",
#     "pillow==12.3.0",
#     "scipy==1.18.1",
#     "timm==1.0.30",
#     "torch==2.14.1",
#     "transformers==5.20.1",
# ]
# ///
"""Build cutouts/: one bird cut out of its Audubon plate, for every species in
audubon.json that Fugleramme has no plate for (#19). Offline; not in the image.

  uv run tools/build_cutouts.py --work ~/.cache/birdlisten-cutouts [--review]
      [--species NAME ...] [--full]

By default only species missing from cutouts/index.json are built (a new
audubon.json entry), so a routine run is seconds and loads no models when
there is nothing to do. --species rebuilds the named ones (after an override
edit); --full rebuilds everything (after a model or crop change).

Needs the network (Commons thumbnails, the Fugleramme manifest, and about
4 GB of model weights from Hugging Face on the first run) and about 9 s per
plate on a laptop CPU. The image ships only cutouts/.

Per plate (the same 960 px Commons thumbnail and crop the server's vignette
uses, frame._vignette_geometry):
  1. Grounding DINO ("a bird.") boxes every bird.
  2. SAM 2.1 cuts each box out.
  3. Each cut-out, flattened onto paper, is labelled by BioCLIP and BioCLIP 2
     against only the species mapped to that plate plus "plant" and "nest".
Per species: the largest bird that both models give the species with
p >= MIN_P; on a plate showing only that species, the largest bird. With
neither, the species keeps its vignette and the report lists it.

tools/cutout_overrides.json fixes what the models get wrong, keyed by
scientific name:
  {"box": [x0, y0, x1, y1]}  the bird to cut, in the cropped plate's pixels
                             (--review prints every box);
  "pos" / "neg": [[x, y]]    extra SAM points on / off the bird; the largest
                             connected piece is kept;
  "source": {...}            a public-domain picture that is not the Audubon
                             plate (file, page, title, credit, credit_url,
                             license); the box is then in its 960 px Commons
                             thumbnail, uncropped;
  {"skip": true}             keep the vignette.
A stale override (unknown species, or one Fugleramme now has) fails the build.
Nothing here runs in the Docker build or at serve time: the server reads the
committed WebPs.

Detections and labels are cached in --work/results.json, so a re-run after an
override edit only re-cuts. --review writes --work/review/index.html: every
plate with its numbered boxes, the chosen cut-out, and both labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import frame  # noqa: E402

OUT = ROOT / "cutouts"
OVERRIDES = ROOT / "tools" / "cutout_overrides.json"
MANIFEST = "https://raw.githubusercontent.com/arnegiacomo/fugleramme/{ref}/assets/artwork/classic/manifest.json"
DETECTOR = "IDEA-Research/grounding-dino-base"
SEGMENTER = "facebook/sam2.1-hiera-large"
CLASSIFIERS = ("bioclip", "bioclip-2")
DET_T = 0.2          # Grounding DINO box and text threshold
MAX_BOX = 0.6        # a box over this share of the picture is the whole scene, not a bird
SAME_BOX = 0.7       # boxes overlapping this much (of the smaller) are one bird
MIN_P = 0.6          # both classifiers must reach this
MAX_SIDE = 800       # px, longest side of a stored cut-out (VIGNETTE_MAX)
QUALITY = 90
RESULTS_VERSION = 1  # bump when detection or labelling changes; the cache is then ignored


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame._write_atomic(dest, frame.fetch_url(url, timeout=60))
        time.sleep(0.5)   # be gentle with Commons
    return dest


def thumb(file: str, work: Path, name: str) -> Path:
    return fetch(frame.COMMONS_THUMB.format(file=quote(file.replace(" ", "_"), safe="")),
                 work / "plates" / f"{name}.jpg")


def at_work_width(im):
    from PIL import Image
    im = im.convert("RGB")
    if im.width == frame.WORK_W:
        return im
    return im.resize((frame.WORK_W, round(im.height * frame.WORK_W / im.width)), Image.LANCZOS)


def plate_picture(path: Path, sci: str):
    """The cropped picture the boxes are measured in."""
    from PIL import Image
    with Image.open(path) as full:
        g = frame._vignette_geometry(full, sci)
        return at_work_width(full).crop(g.crop)


def overlap(a, b) -> float:
    """Intersection over the smaller box."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


class Models:
    def __init__(self):
        import open_clip
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval()
        self.seg_proc = Sam2Processor.from_pretrained(SEGMENTER)
        self.seg = Sam2Model.from_pretrained(SEGMENTER).eval()
        self.clips = {}
        for name in CLASSIFIERS:
            m, _, pre = open_clip.create_model_and_transforms(f"hf-hub:imageomics/{name}")
            self.clips[name] = (m.eval(), pre, open_clip.get_tokenizer(f"hf-hub:imageomics/{name}"))

    def boxes(self, im) -> list[tuple[float, list[int]]]:
        import torch
        inp = self.det_proc(images=im, text="a bird.", return_tensors="pt")
        with torch.no_grad():
            o = self.det(**inp)
        r = self.det_proc.post_process_grounded_object_detection(
            o, inp.input_ids, threshold=DET_T, text_threshold=DET_T, target_sizes=[im.size[::-1]])[0]
        found = sorted(((float(s), [round(v) for v in b.tolist()]) for s, b in zip(r["scores"], r["boxes"])),
                       key=lambda x: -x[0])
        small = [x for x in found if (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]) < MAX_BOX * im.width * im.height]
        keep: list[tuple[float, list[int]]] = []
        for x in small or found[:1]:   # one large bird (the Wild Turkey) fills the picture
            if all(overlap(x[1], k[1]) < SAME_BOX for k in keep):
                keep.append(x)
        return keep

    def cut(self, im, box, pos=(), neg=()):
        """RGBA cut-out (trimmed) and mask area, or (None, 0)."""
        import numpy as np
        import torch
        from PIL import Image, ImageFilter
        from scipy import ndimage
        kw = {}
        if pos or neg:
            kw = {"input_points": [[[list(p) for p in [*pos, *neg]]]],
                  "input_labels": [[[1] * len(pos) + [0] * len(neg)]]}
        pin = self.seg_proc(images=im, input_boxes=[[list(box)]], return_tensors="pt", **kw)
        with torch.no_grad():
            so = self.seg(**pin, multimask_output=False)
        mk = self.seg_proc.post_process_masks(so.pred_masks.cpu(), pin["original_sizes"])[0][0, 0].numpy() > 0
        if pos or neg:
            lbl, n = ndimage.label(mk)
            if n > 1:
                mk = lbl == 1 + int(np.argmax(ndimage.sum(mk, lbl, range(1, n + 1))))
            mk = ndimage.binary_fill_holes(mk)
        a = Image.fromarray(mk.astype("uint8") * 255).filter(ImageFilter.GaussianBlur(0.8))
        bb = a.getbbox()
        if not bb:
            return None, 0
        out = im.convert("RGBA")
        out.putalpha(a)
        return out.crop(bb), int(mk.sum())

    def label(self, cuts, cands: list[str]) -> list[dict]:
        """Per cut-out, each classifier's best label and its probability."""
        import torch
        from PIL import Image
        names = [*cands, "plant", "nest"]
        prompts = [f"a photo of {c}." for c in cands] + ["a photo of a plant.", "a photo of a bird nest."]
        flat = []
        for c in cuts:
            bg = Image.new("RGBA", c.size, (*frame.PAPER, 255))
            bg.alpha_composite(c)
            flat.append(bg.convert("RGB"))
        out = [{} for _ in cuts]
        for name, (m, pre, tok) in self.clips.items():
            with torch.no_grad():
                t = m.encode_text(tok(prompts))
                t /= t.norm(dim=-1, keepdim=True)
                f = m.encode_image(torch.stack([pre(x) for x in flat]))
                f /= f.norm(dim=-1, keepdim=True)
                p = (100 * f @ t.T).softmax(-1).numpy()
            for row, o in zip(p, out):
                j = int(row.argmax())
                o[name] = {"label": names[j], "p": round(float(row[j]), 3)}
        return out


def detect_plate(models: Models, im, cands: list[str]) -> list[dict]:
    birds, cuts = [], []
    for i, (score, box) in enumerate(models.boxes(im)):
        c, area = models.cut(im, box)
        if c is not None:
            birds.append({"i": i, "box": box, "det": round(score, 2), "area": area})
            cuts.append(c)
    for b, lab in zip(birds, models.label(cuts, cands) if cuts else []):
        b.update(lab)
    return birds


def choose(sci: str, e: dict, birds: list[dict]) -> tuple[dict | None, list[str]]:
    """The bird to cut for `sci`, and review notes."""
    ok = [b for b in birds if all(b.get(m, {}).get("label") == sci and b[m]["p"] >= MIN_P for m in CLASSIFIERS)]
    notes = []
    if not ok and e.get("on_plate") == 1 and birds:
        ok = [max(birds, key=lambda b: b["area"])]
        notes.append("only species on the plate")
    if not ok:
        return None, ["models disagree or are unsure"]
    if len(ok) > 1:
        notes.append(f"{len(ok) - 1} more of this species on the plate, kept the largest")
    return max(ok, key=lambda b: b["area"]), notes


def fugleramme_stems(ref: str) -> set[str]:
    doc = json.loads(frame.fetch_url(MANIFEST.format(ref=ref), timeout=60))
    return {Path(k).stem for k in doc if k.startswith("birds/")}


def load_overrides(path: Path, wanted: dict) -> dict:
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc = {k: v for k, v in doc.items() if not k.startswith("_")}
    stale = sorted(k for k in doc if k not in wanted)
    if stale:
        sys.exit(f"stale overrides (not an Audubon-only species): {', '.join(stale)}")
    for k, o in doc.items():
        if not o.get("skip") and not (isinstance(o.get("box"), list) and len(o["box"]) == 4):
            sys.exit(f"override {k}: needs a 4-number box or skip")
        s = o.get("source")
        if s is not None and not (isinstance(s.get("file"), str)
                                  and str(s.get("page", "")).startswith(frame.COMMONS_PREFIX)):
            sys.exit(f"override {k}: source needs file and a Commons page")
    return doc


def save_webp(cut, path: Path) -> None:
    from PIL import Image
    if max(cut.size) > MAX_SIDE:
        cut = cut.copy()
        cut.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    cut.save(buf, "WEBP", quality=QUALITY, alpha_quality=100, method=6)
    frame._write_atomic(path, buf.getvalue())


def review_page(work: Path, rows: list[dict]) -> Path:
    """One card per species: the plate with numbered boxes (the chosen one red),
    the cut-out, and both labels for every bird."""
    from PIL import Image, ImageDraw, ImageFont
    out = work / "review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=22)
    esc = html.escape
    cards = []
    for r in rows:
        s = frame.stem(r["sci"])
        im = r["picture"].copy()
        d = ImageDraw.Draw(im)
        for b in r["birds"]:
            d.rectangle(b["box"], outline=(150, 150, 150), width=2)
            d.text((b["box"][0] + 4, b["box"][1] + 2), f"#{b['i']}", fill=(90, 90, 90), font=font)
        if r["box"]:
            d.rectangle(r["box"], outline=(220, 0, 0), width=4)
        im.thumbnail((520, 520))
        im.save(out / "img" / f"plate-{s}.jpg", quality=85)
        cut = ""
        if r["cut"] is not None:
            c = r["cut"].copy()
            c.thumbnail((360, 360))
            c.save(out / "img" / f"cut-{s}.png")
            cut = f'<img src="img/cut-{s}.png" alt="cut-out">'
        labels = "<br>".join(
            f"#{b['i']}: " + " / ".join(f"{esc(b[m]['label'])} {b[m]['p']}" for m in CLASSIFIERS if m in b)
            for b in r["birds"])
        cards.append(f'<div class="card {r["status"]}"><h2>{esc(r["sci"])}</h2>'
                     f'<p>{esc("; ".join(r["notes"]))}</p>'
                     f'<div class="row"><img src="img/plate-{s}.jpg" alt="plate">{cut}</div>'
                     f'<p class="lab">{labels}</p></div>')
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>cut-out review</title><style>"
        "body{font:14px sans-serif;background:#f4ecd8;margin:1em}.card{background:#fff;margin:1em 0;"
        "padding:.5em 1em;border-left:8px solid #4a4}.card.flag{border-color:#c33}.card.over{border-color:#36c}"
        ".row{display:flex;gap:1em;align-items:flex-end}.row img{max-height:360px}.lab{color:#555}</style>"
        + "\n".join(cards))
    return out / "index.html"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True, help="cache: thumbnails, results.json, review/")
    ap.add_argument("--species", nargs="*", help="only these (scientific names), rebuilt even if present")
    ap.add_argument("--full", action="store_true",
                    help="rebuild every species and drop stale files (default: only species not in index.json)")
    ap.add_argument("--review", action="store_true", help="write WORK/review/index.html")
    args = ap.parse_args(argv)
    from PIL import Image
    work = args.work.expanduser()
    work.mkdir(parents=True, exist_ok=True)

    aud = json.loads(frame.AUDUBON_MAP.read_text())["species"]
    fug = fugleramme_stems(frame.DEFAULT_ARTWORK_REF)
    wanted = {k: e for k, e in aud.items() if frame.stem(k) not in fug}
    overrides = load_overrides(OVERRIDES, wanted)
    ipath = OUT / "index.json"
    old = {} if args.full or not ipath.exists() else json.loads(ipath.read_text())["species"]
    if args.species:
        todo = {k: wanted[k] for k in args.species}
    elif args.full:
        todo = wanted
    else:   # new audubon.json species, or Fugleramme dropped one
        todo = {k: e for k, e in wanted.items() if k not in old}
        print(f"{len(old)} cut-outs present, {len(todo)} to build (--full rebuilds all)")
    if not todo and not args.review:
        return 0
    byplate: dict[int, list[str]] = {}
    for k, e in aud.items():
        byplate.setdefault(e["plate"], []).append(k)

    rpath = work / "results.json"
    cache = json.loads(rpath.read_text()) if rpath.exists() else {}
    if cache.get("version") != RESULTS_VERSION:
        cache = {"version": RESULTS_VERSION, "plates": {}}
    models = Models()   # the slow part: ~4 GB of weights

    index: dict[str, dict] = {}
    rows, missing = [], []
    for n, (sci, e) in enumerate(sorted(todo.items())):
        o = overrides.get(sci, {})
        if o.get("skip"):
            missing.append((sci, "skipped by override"))
            continue
        src = o.get("source")
        if src:
            with Image.open(thumb(src["file"], work, frame.stem(sci))) as im:
                picture = at_work_width(im)
            birds = []
        else:
            cands = byplate[e["plate"]]
            picture = plate_picture(thumb(e["file"], work, str(e["plate"])), cands[0])
            key = str(e["plate"])
            if key not in cache["plates"]:
                cache["plates"][key] = detect_plate(models, picture, cands)
                frame._write_atomic(rpath, json.dumps(cache, indent=1).encode())
            birds = cache["plates"][key]
        if o.get("box"):
            box, notes, status = o["box"], ["override"], "over"
        else:
            pick, notes = choose(sci, e, birds)
            box, status = (pick["box"], "auto") if pick else (None, "flag")
        cut = None
        if box:
            cut, _ = models.cut(picture, box, o.get("pos", ()), o.get("neg", ()))
        rows.append({"sci": sci, "picture": picture, "birds": birds, "box": box, "cut": cut,
                     "notes": notes, "status": status})
        if cut is None:
            missing.append((sci, "; ".join(notes)))
            continue
        save_webp(cut, OUT / f"{frame.stem(sci)}.webp")
        meta = src or {k: e.get(k) for k in ("plate", "title", "page", "credit", "credit_url")}
        index[sci] = {"file": f"{frame.stem(sci)}.webp", "source": "other" if src else "audubon",
                      **{k: v for k, v in meta.items() if k != "file" and v is not None}}
        print(f"{n + 1}/{len(todo)} {sci}: {status}", flush=True)

    old.update(index)
    if args.full:   # a species dropped from the table, now skipped, or now in Fugleramme
        for p in OUT.glob("*.webp"):
            if p.name not in {v["file"] for v in old.values()}:
                p.unlink()
    frame._write_atomic(ipath, json.dumps(
        {"generated": dt.date.today().isoformat(), "species": dict(sorted(old.items()))},
        indent=1, ensure_ascii=False).encode() + b"\n")
    print(f"\n{len(old)} cut-outs in {OUT.relative_to(ROOT)}/; {len(missing)} keep the vignette:")
    for sci, why in missing:
        print(f"  {sci}: {why}")
    if args.review:
        print("review:", review_page(work, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
