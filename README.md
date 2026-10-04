# birdlisten

Listens to the audio track of Reolink cameras and logs which birds
[BirdNET](https://birdnet.cornell.edu/) (Cornell Lab of Ornithology) hears.
The cameras you already have become distributed microphones; a SQLite table of
detections and optional phone pushes come out the other end.

```
Reolink RTSP ──ffmpeg──▶ 30 s WAV ──BirdNET──▶ species + confidence ──▶ SQLite
                                                                       └─▶ ntfy push (new species, 60-min cooldown)
```

Runs as a container on the [fleet](https://github.com/joekraemer/fleet) host.
`loop.py` calls `main()` back to back; each call is one listen window per
camera, cameras in turn, so the Mac only ever runs one analysis at a time.

## Status: stub, verified as far as it can be without a camera

Verified in the built image: ffmpeg is present and the `capture()` flags are
valid; the BirdNET model loads under tflite-runtime; a full pass with a
synthetic clip runs through analysis and storage; 98 test functions (168 cases
with parametrization) cover config parsing, dedupe, cooldown, storage, the
per-pass error handling, the collage page (query, packer, renderer, both
plate caches, vignette processing, HTTP routes), and the `audubon.json`
build rules. NOT yet
verified: an actual Reolink RTSP stream, and real bird detections. Expect to
tune `MIN_CONFIDENCE` once you see what the yard sounds like to BirdNET.

## Setup

1. On each Reolink camera: Settings → Camera → Audio → **Record Audio ON**.
   Without this the RTSP stream has no audio track and `capture()` fails with
   "ffmpeg produced no audio".
2. Find the RTSP URL. Reolink: `rtsp://USER:PASS@IP:554/h264Preview_01_sub`
   (older firmware) or `rtsp://USER:PASS@IP:554/Preview_01_sub` (newer). Use the
   sub-stream: less video to discard, identical audio. Test from any laptop
   with `ffplay -rtsp_transport tcp <url>`.
3. Give the cameras fixed IPs (DHCP reservation in the eero app).
4. `cp .env.example .env`, fill in `CAMERAS`, `LATITUDE`, `LONGITUDE`.
5. Prove it:
   ```
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--check" app     # config, ffmpeg, model, 3 s probe of each camera
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--dry-run" app   # one real capture+analysis, no writes
   docker compose run --rm -e LOOP_ONCE=1 app                            # one real pass
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--report 7" app  # what was heard this week
   ```
6. Fleet: on the Mac, create `~/.config/fleet/birdlisten.env` with the same
   content as `.env`. Then in `fleet/compose.yaml` change `services: {}` to
   `services:`, uncomment the `birdlisten` block and the `volumes:` block, push
   fleet.

## Notifications

Set `NTFY_TOPIC` to an unguessable string, install the ntfy app on your phone,
subscribe to that topic. You get one push per species per hour: "Varied Thrush
— 07:12 on back camera, 83% confidence". `NOTIFY_COOLDOWN_MIN` tunes the hour.

## Data

`/data/birdlisten.sqlite` on the container volume. Two tables: `detections`
(one row per species per clip, best window) and `notified` (cooldown state).
`--report N` summarizes. `KEEP_CLIPS=1` also saves the WAV of any clip with a
detection under `/data/clips/YYYY-MM-DD/`, useful for checking BirdNET's work
by ear. From the Mac: `docker compose -f ~/fleet/compose.yaml exec birdlisten python birdlisten.py --report 7`.

## Tuning

* `MIN_CONFIDENCE` 0.5 is a reasonable start. BirdNET's location/date filter
  already removes implausible species (for Seattle in September it considers
  ~160 of the model's 6,500); the remaining false positives are usually
  mechanical noise scored as a bird at 0.3–0.5. birdnetlib clamps the value to
  0.01–0.99 and the comparison is strict (a detection at exactly 0.5 is dropped).
* `CLIP_SECONDS` 30 gives ten 3-second BirdNET windows per camera per pass.
  Longer clips catch more but delay notifications.
* CPU: on the 2017 Intel MacBook a 30 s clip analyzes in a few seconds.
  Two cameras is comfortable; six would still be fine. Measure it with the
  timing lines below.

## Performance logging

Every pass logs `timing` lines at INFO, for sizing concurrent capture (#4):

```
timing model_load_s=4.10 cpu_s=3.90 rss_mb=400            # once per process
timing camera=garage clip_s=30.0 capture_s=31.40 analyze_s=3.82 detections=2 cpu_s=3.70 rss_mb=612
timing camera=yard clip_s=30.0 capture_s=40.02 rss_mb=612 error=ffmpeg_failed:_Connection_refused
timing pass cameras=5 ok=4 wall_s=171.30 analyze_total_s=15.20 rss_mb=612
```

* `capture_s`: wall time of the ffmpeg call. `analyze_s`: wall time of the
  BirdNET analysis of that clip, model load excluded.
* `cpu_s`: user+sys CPU of the process (all threads) plus reaped children,
  from `getrusage`, during analysis. Above `analyze_s` means BirdNET used more
  than one core.
* `rss_mb`: `ru_maxrss`, the process's peak RSS so far, not current usage.
* `detections`: species kept after `MIN_CONFIDENCE`.
* A failed camera still gets a line, with `error=` (spaces become `_`) and
  whatever was measured before the failure.
* The model is loaded once per process (loop.py calls `main()` in the same
  process), so `model_load_s` appears once after each container start.
* Duty cycle per camera is `clip_s / wall_s` of the pass line.

`--dry-run` prints the same lines for one pass without writing to the DB or
notifying: `docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--dry-run" app`.

On the fleet Mac:

```
docker compose -f ~/fleet/compose.yaml logs birdlisten --since 24h | grep ' timing '
```

Summary (mean/p95 `analyze_s` per camera, errors, mean pass `wall_s`):

```
docker compose -f ~/fleet/compose.yaml logs birdlisten --since 24h | grep ' timing ' | python3 -c '
import sys,statistics as st,collections as C
a=C.defaultdict(list);w=[];e=C.Counter()
for l in sys.stdin:
  f=dict(t.split("=",1) for t in l.split(" timing ",1)[1].split() if "=" in t)
  if "wall_s" in f: w.append(float(f["wall_s"]))
  elif "error" in f: e[f["camera"]]+=1
  elif "analyze_s" in f: a[f["camera"]].append(float(f["analyze_s"]))
for c,v in sorted(a.items()): v.sort(); print(f"{c:<14} n={len(v):<5} mean={st.mean(v):.2f}s p95={v[int(.95*(len(v)-1))]:.2f}s errors={e[c]}")
for c in sorted(set(e)-set(a)): print(f"{c:<14} n=0     errors={e[c]}")
w and print(f"pass           n={len(w):<5} mean_wall={st.mean(w):.1f}s")
'
```

## Collage page

Optional. Set `SERVE_PORT` and the container also serves a "recently heard
birds" page laid out like a field-guide plate: a centred title ("Heard in the
last 24 hours") and species count, then one plate per species heard in the
last N hours (default 24), most recent first, with its common name beneath.
Cameras, times and first-ever sightings are left off the picture and stay in
`/api/recent`. Only detections at or above `MIN_CONFIDENCE` count, so raising
it takes older low-confidence rows off the page straight away. It is meant to sit in a Home Assistant Webpage card (below). The server
is a daemon thread beside the capture loop and reads the SQLite db through its
own read-only connection; it never changes what the loop records or notifies.

| route | what |
|---|---|
| `GET /` | HTML page showing `/collage.png`, swaps the image every 60 s (meta refresh fallback without JS). `?hours=` |
| `GET /collage.png` | the collage. `?hours=1..720` (default `COLLAGE_HOURS`), `?w=`, `?h=` 200..4000 (default 1600x1200) |
| `GET /api/recent` | JSON `{hours, generated_at, species: [{scientific_name, common_name, last_heard, count, cameras, first_ever, has_plate}]}`. `has_plate`: a Fugleramme cut-out or Audubon plate is on disk. `?hours=` |
| `GET /attribution` | artwork credit: Fugleramme plus its `ATTRIBUTION.md`, and the Audubon plates fetched so far |
| `GET /favicon.ico` | 204 |

Artwork comes from two sources, tried in this order for each species:

1. **Fugleramme cut-out.** The bird on a transparent background, pasted
   straight onto the page.
2. **Audubon vignette.** If Fugleramme has no plate under the BirdNET name,
   `audubon.json` maps the name to one Havell plate of Audubon's *The Birds of
   America* (413 BirdNET species on 381 plates). The server fetches a 960 px
   Wikimedia Commons thumbnail of it, never the full-size scan, crops it to
   the picture (the engraved caption, plate number and margins are removed),
   recolours the paper to the placeholder card's tone, and draws it inside the
   card's double-rule frame. On a plate with several birds the whole plate is
   shown.
3. **Placeholder card.** A plain paper card of the same size when neither
   source has the species.

Art is fetched lazily, one species at a time on first need, into
`$ARTWORK_DIR` (`/data/artwork` in the container, so it lives on the same
volume as the db; vignettes go under `audubon/v1/`) and never re-fetched.
Fetching shares a 15 s budget per render; whatever is left over is fetched on
a later render. A miss is remembered in a `<stem>.missing` marker and retried
after a day (a 404, or an Audubon scan the cropper cannot use) or an hour
(a network error). The PNG is re-rendered only when something visible
changed (a species, its name or order, art arriving or changing source), so
the 60 s page refresh normally costs nothing.

Layout: every cell is the same size, species are spread evenly over the rows
(22 at 1600x1200 is 6, 6, 5, 5) and each row is centred. Every name on a page
is set in one size; a long name wraps onto a second line rather than
shrinking. Capacity: cells shrink to an 80 px minimum, then the oldest species
are dropped and the count reads `N species, M not shown`; that is 32 species
at 800x600 and 98 at the default 1600x1200 (`frame.capacity(w, h)` computes
it).

Type is Libre Baskerville (regular and italic) from `fonts/`, SIL Open Font
License 1.1 (`fonts/OFL.txt`; source commit in `fonts/SOURCE.txt`). Without
those files the renderer falls back to Pillow's bundled sans.

### Artwork credit

Bird plates are from the [Fugleramme](https://github.com/arnegiacomo/fugleramme)
project (`assets/artwork/classic`), licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Full
per-plate sources are in the project's ATTRIBUTION.md, which the server
fetches and exposes at `/attribution`. Because the plates are BY-SA, the
generated collage inherits CC BY-SA 4.0 for the plate content. The collage
layout here is this repo's own code, not Fugleramme's renderer; `ARTWORK_REF`
pins the Fugleramme commit the plates come from.

Fallback plates are from John James Audubon, *The Birds of America* (London,
1827–1838), Havell edition: hand-coloured engravings by Robert Havell Jr.
(plates 1–10 first engraved by W. H. Lizars). The scans are on
[Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:The_Birds_of_America),
almost all credited to the University of Pittsburgh, and are public domain
(PD-Art), so no share-alike applies to them. The vignettes shown are cropped
and recoloured. `/attribution` lists each plate fetched so far with its
Commons page and credit. Thumbnails are requested from
`commons.wikimedia.org/wiki/Special:FilePath/<file>?width=960` with the
User-Agent `birdlisten/1.0 (https://github.com/joekraemer/birdlisten)`, as
Wikimedia's User-Agent policy asks; each species' thumbnail is fetched once and cached.

#### Rebuilding audubon.json

`audubon.json` is committed and generated by hand with
`tools/build_audubon_map.py` (stdlib only, needs the network; it queries
Wikidata and the Commons API, which the server never does):

```
uv run --no-project --python 3.11 python tools/build_audubon_map.py --out audubon.json
```

It unions Wikidata's plate items (`depicts`) with Commons'
`(illustrations)` categories, then applies the manual tables at the top of the
script (`NAME_ALIASES`, `PLATE_ADDITIONS`, `PAIR_DENY`); a stale override
fails the build. BirdNET labels come from `--labels PATH|URL`, else the
installed birdnetlib, else `LABELS_URL`, pinned to birdnetlib tag 0.18.0
(commit `8746db6`); update that commit when `birdnetlib` is bumped in
`pyproject.toml`. The script prints a coverage report; review it in the CR,
especially the "CHOSEN Commons-only pairs", which only one source supports
and should be checked by eye against the plate. If the vignette processing
in `frame.py` changes, bump `VIGNETTE_VERSION` so cached vignettes are
rebuilt.

### Home Assistant

Webpage card (Lovelace, "Manual" card or YAML mode):

```yaml
type: iframe
url: http://192.168.1.10:8085/
aspect_ratio: 75%     # 4:3, matches the default 1600x1200 collage
title: Birds heard today
```

If HA is opened over https the browser blocks a plain-http iframe (mixed
content); open HA over http on the LAN or put the collage behind the same TLS
proxy. Optional REST sensor for the species count:

```yaml
sensor:
  - platform: rest
    name: birdlisten_species_24h
    resource: http://192.168.1.10:8085/api/recent?hours=24
    value_template: "{{ value_json.species | count }}"
    scan_interval: 300
```

### Configuration

| var | default | meaning |
|---|---|---|
| `SERVE_PORT` | unset | port for the collage server. Unset or empty = no server, no network traffic, nothing changes. |
| `COLLAGE_HOURS` | `24` | window for `/`, `/collage.png` and `/api/recent`, 1..720 |
| `MIN_CONFIDENCE` | `0.5` | the capture loop's threshold, also applied when reading: rows below it are left off the page and `/api/recent`, 0..1 |
| `ARTWORK_REF` | `8e8b0034f069b4d3b021bc7195482c1fe7caf880` | Fugleramme commit (or branch) the plates are fetched from |
| `ARTWORK_DIR` | `$DATA_DIR/artwork` | plate cache, a few hundred KB per species |
| `AUDUBON_FALLBACK` | `1` | `1` = Audubon plates for species Fugleramme lacks, `0` = Fugleramme only (pages are then exactly as before) |

A bad value logs `config error: ... (server disabled)` and the loop runs on
without the server; a busy port logs `cannot bind SERVE_PORT=...` and does the
same. The server never changes the loop's exit code.

### Running it on the fleet host

Mirror two things in `fleet/compose.yaml`: add `ports: ["8085:8085"]` and
`SERVE_PORT=8085` to the birdlisten service, and note that the existing
`/data` volume now also holds `artwork/` (a few hundred KB per species).

## Tests

```
uv run --group dev pytest -q                     # Linux / inside the image
uv run --no-project --python 3.11 --with pillow==12.3.0 --with pytest==8.3.4 pytest -q \
  test_birdlisten.py test_frame.py test_serve.py test_build_audubon_map.py   # arm64 macOS
```

The second form exists because `tflite-runtime` has no macOS arm64 wheel, so
the project environment cannot resolve there; the tests only need stdlib plus
Pillow (`birdnetlib` is imported lazily by the analyzer). Both must pass. No
test touches the network: artwork fetches are monkeypatched to 404, and the
build script's fetch layer is monkeypatched too. The vignette tests use
synthetic sheets plus three public-domain 480 px Havell thumbnails in
`tests/fixtures/audubon/` (plates 8, 362, 376).

## Ideas not built

* Write detections to a Notion database (the movie-review pattern) or a
  weekly digest to ntfy.
* Per-camera `MIN_CONFIDENCE`, since a camera near a road is noisier.
* Trim the image: birdnetlib depends on matplotlib and librosa pulls in
  scikit-learn, making it ~2.5 GB. A slimmer inference path (`ai-edge-litert`
  + hand-rolled spectrogram) would be ~300 MB but is real work.
* Detect and alert on *new* species for the yard (first ever sighting) as a
  separate notification tier. (`/api/recent` already flags them as
  `first_ever`; this is about a push.)
* MQTT / Home Assistant discovery for the collage data. The REST sensor above
  covers the species count.
* E-ink output. Fugleramme already does this well; this project stops at a PNG.
* Scaling plates by body mass so a crow is bigger than a chickadee. Needs a
  mass table, and Fugleramme's manifest has none.
* A BirdNET-to-Fugleramme name alias table for species whose scientific names
  differ between the two. Many of those now get an Audubon plate instead of a
  placeholder, but a Fugleramme cut-out would still look better.
* Audubon's octavo edition (Bowen lithographs) for the dozen or so species
  first figured there, and per-figure crops of multi-species plates. Neither
  has machine-readable metadata, so both need hand work.
* An index on `detections(scientific_name, heard_at)` if the db ever grows
  enough for the recent-species query to show up in render time.

## Files

| file | role |
|---|---|
| `birdlisten.py` | the app; `main() -> int`, `--check`, `--dry-run`, `--report N` |
| `frame.py` | collage: recent-species query, Fugleramme and Audubon plate caches, vignette processing, packer, Pillow renderer, render cache |
| `audubon.json` | BirdNET scientific name → Havell plate on Commons (file, page, credit); generated, committed |
| `tools/build_audubon_map.py` | offline builder of `audubon.json`; not in the image |
| `serve.py` | the collage HTTP server; `start_from_env()` is what `loop.py` calls |
| `loop.py` | container entrypoint (fleet template), runs `main()` back to back; starts the server when `SERVE_PORT` is set |
| `Dockerfile` | fleet template + Python 3.11 + Debian ffmpeg/libsndfile |
| `pyproject.toml`, `uv.lock` | birdnetlib 0.18, tflite-runtime 2.14, numpy<2, pillow 12.3 |
| `compose.yaml` | local dev; production compose lives in the fleet repo |
| `.github/workflows/build.yml` | build + push `ghcr.io/joekraemer/birdlisten:main` |
| `test_birdlisten.py`, `test_frame.py`, `test_serve.py`, `test_build_audubon_map.py` | see Tests |
| `tests/fixtures/audubon/` | three Havell thumbnails for the vignette tests |
