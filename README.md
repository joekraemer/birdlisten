# birdlisten

Listens to the audio track of Reolink cameras and logs which birds
[BirdNET](https://birdnet.cornell.edu/) (Cornell Lab of Ornithology) hears.
The cameras you already have become distributed microphones; a SQLite table of
detections and optional phone pushes come out the other end.

```
camera 1 RTSP ──ffmpeg (always on)──▶ 30 s WAV segments ─┐
camera 2 RTSP ──ffmpeg (always on)──▶ 30 s WAV segments ─┼─▶ queue ──▶ one BirdNET worker ──▶ SQLite
   ...                                                    │   (drops oldest       └─▶ ntfy push (new species, 60-min cooldown)
camera N RTSP ──ffmpeg (always on)──▶ 30 s WAV segments ─┘    when behind)
```

Runs as a container on the [fleet](https://github.com/joekraemer/fleet) host.
Every camera listens all the time: each one has a long-lived ffmpeg that cuts
its audio into back-to-back `CLIP_SECONDS` WAV segments, so no audio is missed
between clips. Finished segments go on a small queue to a single analysis
worker, so the Mac only ever runs one BirdNET analysis at a time, and that
worker is the only thing that writes to SQLite or sends a push. `loop.py`
calls `main()`, which runs this pipeline until the container stops. The old
behaviour, one camera at a time in passes, is still there as
`CAPTURE_MODE=roundrobin` (see [Capture](#capture)).

## Status: stub, verified as far as it can be without a camera

Verified in the built image: ffmpeg is present and the `capture()` flags are
valid; the BirdNET model loads under tflite-runtime; a full pass with a
synthetic clip runs through analysis and storage; the stream-mode segment
muxer cuts a test tone into the expected clips with the image's own ffmpeg
(`python stream.py --selftest`, run by CI). 266 test functions (369 cases
with parametrization, one of which needs a local ffmpeg and is skipped on
macOS) cover config parsing, dedupe, cooldown, storage, the per-pass error
handling, stream capture (segment hand-off, restarts and backoff, the queue,
the worker, the summary line, shutdown, crash safety and signals, against a
fake ffmpeg), the collage page (query, packer, renderer, both
plate caches, vignette processing, HTTP routes, click targets), the pop-up
(species stats, fact fetching and caching, input validation, the eBird key),
and the `audubon.json` build rules. NOT yet
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
   docker compose run --rm -e LOOP_ONCE=1 app                            # one real stream window
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--report 7" app  # what was heard this week
   ```
   `--dry-run` is always one round-robin pass. With `LOOP_ONCE=1` and no
   arguments, stream mode listens for one window: every camera starts its
   ffmpeg (1.5 s apart), hands off one complete clip, and the worker analyzes
   them all, then it exits. That takes about 45 s for 4 cameras with 30 s
   clips. In round-robin mode it is one pass.
6. Fleet: on the Mac, create `~/.config/fleet/birdlisten.env` with the same
   content as `.env`. Then in `fleet/compose.yaml` change `services: {}` to
   `services:`, uncomment the `birdlisten` block and the `volumes:` block, push
   fleet.

## Capture

| var | default | meaning |
|---|---|---|
| `CAPTURE_MODE` | `stream` | `stream`: every camera at once, one long-lived ffmpeg each, one BirdNET worker. `roundrobin`: the old passes, one camera at a time, each `main()` call one pass |
| `QUEUE_SIZE` | 2 × cameras | clips waiting for analysis before the oldest is dropped, 1..1000 |
| `SUMMARY_MINUTES` | `5` | minutes between `timing summary` log lines, 1..1440 |
| `SEGMENT_DIR` | `/tmp/birdlisten-segments` | in-flight WAV segments, about 3 MB each and deleted after analysis. Absolute; must not be `DATA_DIR` or inside it |
| `CLIP_SECONDS` | `30` | seconds per analyzed segment, at least 3 (one BirdNET window). In stream mode it sets the analysis window and `heard_at` granularity; in round-robin mode it is the audio per camera per pass |

A bad value is a config error (`config error: ...`, exit code 2). At startup
stream mode deletes any segments a crashed run left in `SEGMENT_DIR`; it only
touches files matching its own naming pattern. Peak use is about
(cameras + `QUEUE_SIZE` + 1) × 2.9 MB, about 37 MB for 4 cameras. On the
fleet a `tmpfs` at `SEGMENT_DIR` keeps those writes off the VM's disk.

`--check` prints the capture line, for example
`capture: stream, queue 8, summary every 5 min, segments /tmp/birdlisten-segments (writable, 812 MB free)`,
and an unwritable `SEGMENT_DIR` counts as a problem. It only writes and
removes a `.check-<pid>` probe file there and never cleans anything up,
because it can run next to a live container that shares the directory. A
camera whose URL looks like a main stream gets a
`hint: <camera> uses a main-stream URL; use the sub-stream` line.

Shutdown: on `docker stop`, every ffmpeg is stopped and reaped, the worker
finishes the clip in hand and takes no more, queued clips are deleted, and the
process exits in under 9 s at worst, inside Docker's 10 s grace period. If the pipeline
hits a fatal error (the model or the db fails to load, a thread dies), it
logs it at ERROR, waits 60 s and returns 1, and `loop.py` starts it again.

### Reolink NVR stream limits

All cameras are channels on one Reolink NVR, and stream mode keeps one RTSP
session per channel open permanently. If the summary shows
`cameras_up < cameras` or `refused > 0`, or the log has
`still failing ... 453 Not Enough Bandwidth` or `503`, the NVR is refusing
sessions. In order:

1. Close other viewers: the Reolink app, and Home Assistant's live view or
   camera stream.
2. Use sub-stream URLs (`..._sub`), which have the same audio at a fraction
   of the bandwidth. The startup WARNING and `--check` name any main-stream
   camera.
3. Set `CAPTURE_MODE=roundrobin` in the env file. It opens one session at a
   time, as before.

A `--check` run next to a live stream-mode container (for example
`docker compose exec birdlisten python birdlisten.py --check`) opens an extra
RTSP session per camera for its 3 s probe, and the NVR may refuse it. Stop
the container first, or read the latest `timing summary` line instead.

## Notifications

Set `NTFY_TOPIC` to an unguessable string, install the ntfy app on your phone,
subscribe to that topic. You get one push per species per hour: "Varied Thrush
— 07:12 on back camera, 83% confidence". `NOTIFY_COOLDOWN_MIN` tunes the hour.
BirdNET's non-bird labels (Dog, Engine, Human vocal, frogs, insects, mammals;
see `taxa.py`) are recorded but never pushed.

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
* `CLIP_SECONDS` 30 gives ten 3-second BirdNET windows per clip. In stream
  mode every camera listens continuously whatever the value; shorter clips
  give finer `heard_at` times and faster pushes, at a little more overhead
  per clip.
* CPU: on the 2017 Intel MacBook a 30 s clip analyzes in about 1.2 s, so one
  worker is busy about 4 × 1.2 / 30 ≈ 16% of the time with 4 cameras (the
  summary's `duty`). Each always-on ffmpeg is expected to use a few percent
  of a core. Analysis falls behind only near `duty` 1.0, roughly 25 cameras
  at that speed; the summary's `dropped` says when it does.

## Performance logging

Stream mode logs `timing` lines at INFO on logger `birdlisten`. Each is
`timing ` followed by `key=value` tokens only; `error=` is always last, with
spaces turned into `_`.

```
timing model_load_s=4.10 cpu_s=3.90 rss_mb=400
timing camera=garage clip_s=30.0 segment_lag_s=0.62 queue_wait_s=0.41 analyze_s=1.21 detections=2 cpu_s=1.15 rss_mb=611
timing camera=garage clip_s=30.0 segment_lag_s=0.55 queue_wait_s=0.03 analyze_s=1.18 cpu_s=1.10 rss_mb=611 error=database_is_locked
timing camera=yard clip_s=30.0 queue_wait_s=148.20 dropped=queue_full rss_mb=611
timing camera=doorbell up_s=812.4 segments=27 backoff_s=4.3 error=ffmpeg_exited:_Connection_refused
timing camera=garage up_s=3.1 segments=0 backoff_s=61.7 refused=1 error=ffmpeg_exited:_method_DESCRIBE_failed:_453_Not_Enough_Bandwidth
timing summary window_s=300 cameras=4 cameras_up=4 clips=40 analyzed=40 dropped=0 short=0 failed=0 restarts=0 refused=0 analyze_mean_s=1.21 analyze_p95_s=1.40 duty=0.16 notify_s=0.8 queue_max=2 queue_len=0 cpu_s=58.3 ffmpeg_cpu_s=9.1 rss_mb=640 ffmpeg_rss_mb=96
timing summary window_s=212 final=1 cameras=4 cameras_up=0 clips=28 analyzed=28 dropped=0 short=0 failed=0 restarts=0 refused=0 analyze_mean_s=1.20 analyze_p95_s=1.33 duty=0.16 notify_s=0.0 queue_max=1 queue_len=0 cpu_s=41.0 rss_mb=640
```

The model is loaded once per process, so `model_load_s` appears once after
each container start.

**Per clip** (one line per analyzed, failed or dropped clip):

* `clip_s`: the real segment length from the WAV header.
* `segment_lag_s`: from the end of the segment to its hand-off to the queue
  (poll latency plus ffmpeg opening the next file). The start time comes from
  the file name, so it has ±1 s resolution and can be slightly negative.
* `queue_wait_s`: how long the clip waited for the worker. Growing values
  mean analysis is falling behind.
* `analyze_s`, `cpu_s`: wall time and process CPU (`getrusage`, all threads)
  of the BirdNET call only. `detections`: species kept after
  `MIN_CONFIDENCE`.
* `error=`: the clip could not be analyzed or stored. It counts as `failed`
  in the summary.
* `dropped=queue_full`: the queue was full and this, the oldest clip, was
  deleted unanalyzed. The first drop in a summary window is also a WARNING.
* `rss_mb`: `ru_maxrss`, the process's peak RSS so far, not current usage.

**Per ffmpeg exit** (`up_s`, `segments`, `backoff_s`): one camera's ffmpeg
exited or stalled after `up_s` seconds and `segments` clips, and is retried
after `backoff_s` (doubling from 1 s to 60 s, ±20% jitter, reset after 2 min
of healthy running). `refused=1` marks an NVR refusal (453, 503, connection
refused). The ERROR line is logged once per new reason, then a
`still failing` WARNING every 10 min, and `recovered after N attempts` at
INFO once it is healthy again.

**Summary**, every `SUMMARY_MINUTES` and once more with `final=1` at
shutdown. It is the stream-mode liveness line: `loop.py`'s `run finished`
now appears only at shutdown.

* `window_s`: seconds since the previous summary.
* `cameras`: configured. `cameras_up`: cameras whose ffmpeg was writing audio
  at that moment.
* `clips`: segments handed to the queue. `analyzed`: clips analyzed and
  stored. `dropped`: queue-full drops. `short`: segments under 3 s (normally
  only at a restart). `failed`: bad segments, duplicate names and per-clip
  analysis or storage failures. `restarts`: unexpected ffmpeg exits, stalls
  and spawn failures. `refused` is the subset that were NVR refusals.
* `analyze_mean_s`, `analyze_p95_s`: over the clips analyzed in the window,
  omitted when there were none.
* `duty`: the fraction of the window the worker was busy (analysis, storage
  and ntfy), 0..1. Expect about cameras × `analyze_s` / `clip_s`, 0.16 for
  4 cameras. **`duty` at 0.9 or more, or `dropped > 0`, sustained, means one
  worker cannot keep up.**
* `notify_s`: seconds spent sending ntfy pushes in the window.
  `duty - notify_s / window_s` is the BirdNET share, so a slow ntfy can be
  told apart from slow analysis.
* `queue_max`: the longest the queue got in the window. `queue_len`: its
  length now.
* `cpu_s`: Python's CPU over the window, all threads.
* `ffmpeg_cpu_s`, `ffmpeg_rss_mb` (Linux only, from `/proc`): CPU and summed
  RSS of the ffmpeg processes. The RSS sum counts shared libraries once per
  process, so it overstates the real cost. Both are omitted where `/proc` is
  unavailable.
* `rss_mb`: as above.

Normal-path lines (per-clip OK lines, summaries, `recovered`, startup and
shutdown) avoid the words the fleet health check greps for. That is why the
summary says `failed=`, not `errors=`.

### Round-robin lines (`CAPTURE_MODE=roundrobin`)

Unchanged from before stream mode:

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
* A failed camera still gets a line, with `error=` and whatever was measured
  before the failure.
* Duty cycle per camera is `clip_s / wall_s` of the pass line.

`--dry-run` prints these lines for one pass without writing to the DB or
notifying: `docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--dry-run" app`.

### Reading the logs

On the fleet Mac:

```
docker compose -f ~/fleet/compose.yaml logs birdlisten --since 24h | grep ' timing '
```

Summary (mean/p95 `analyze_s` and errors per camera, mean pass `wall_s` in
round-robin mode, and totals over the stream summaries):

```
docker compose -f ~/fleet/compose.yaml logs birdlisten --since 24h | grep ' timing ' | python3 -c '
import sys,statistics as st,collections as C
a=C.defaultdict(list);w=[];e=C.Counter();s=[]
for l in sys.stdin:
  f=dict(t.split("=",1) for t in l.split(" timing ",1)[1].split() if "=" in t)
  if "window_s" in f: s.append(f)
  elif "wall_s" in f: w.append(float(f["wall_s"]))
  elif "error" in f: e[f.get("camera","?")]+=1
  elif "analyze_s" in f: a[f["camera"]].append(float(f["analyze_s"]))
for c,v in sorted(a.items()): v.sort(); print(f"{c:<14} n={len(v):<5} mean={st.mean(v):.2f}s p95={v[int(.95*(len(v)-1))]:.2f}s errors={e[c]}")
for c in sorted(set(e)-set(a)): print(f"{c:<14} n=0     errors={e[c]}")
w and print(f"pass           n={len(w):<5} mean_wall={st.mean(w):.1f}s")
if s:
  d=[float(x["duty"]) for x in s]; g=lambda k: sum(int(x.get(k,0)) for x in s)
  q=max(int(x["queue_max"]) for x in s); r=max(float(x["rss_mb"]) for x in s); u=min(int(x["cameras_up"]) for x in s if "final" not in x) if any("final" not in x for x in s) else 0
  print("summary        n=%d duty_mean=%.2f duty_max=%.2f clips=%d dropped=%d refused=%d restarts=%d min_cameras_up=%d queue_max=%d rss_max_mb=%.0f" % (len(s),st.mean(d),max(d),g("clips"),g("dropped"),g("refused"),g("restarts"),u,q,r))
'
```

`errors=` per camera counts per-clip failures and ffmpeg restarts. Dropped
clips have no `analyze_s` and are not counted per camera; the summary line
totals them.

## Collage page

Optional. Set `SERVE_PORT` and the container also serves a "recently heard
birds" page laid out like a field-guide plate: a centred title ("Heard in the
last 24 hours") and species count, then one plate per species heard in the
last N hours (default 24), most recent first, with its common name beneath.
Cameras, times and first-ever sightings are left off the picture and stay in
`/api/recent`. Only detections at or above `MIN_CONFIDENCE` count, so raising
it takes older low-confidence rows off the page straight away. BirdNET's
non-bird labels (Dog, Engine, frogs, crickets; the list is `taxa.py`) stay in
SQLite and in `--report`, but are left off the page and `/api/recent`, and
their pop-up is a 404. It is meant to sit in a Home Assistant Webpage card (below). The server
is a daemon thread beside the capture loop and reads the SQLite db through its
own read-only connection; it never changes what the loop records or notifies.

| route | what |
|---|---|
| `GET /` | HTML page: the collage with a tappable bird pop-up (below), refreshed every 60 s (meta refresh fallback without JS). `?hours=`, `?w=`, `?h=` as for `/collage.png`. It renders in the request, so the first load after a restart can wait for plates (up to the 15 s fetch budget). |
| `GET /collage.png` | the collage. `?hours=1..720` (default `COLLAGE_HOURS`), `?w=`, `?h=` 200..4000 (default 1600x1200) |
| `GET /collage.png?v=<token>` | one cached render by its 16-hex content token (the last 8 are kept), `Cache-Control: public, max-age=31536000, immutable`; 404 once evicted |
| `GET /api/layout` | JSON `{token, w, h, shown, dropped, png, hours, targets: [{scientific_name, common_name, stem, art, x, y, w, h}]}`: the click targets of the current render as percentages of the image, and the URL of exactly that PNG. Same params as `/collage.png` |
| `GET /api/species/<scientific name>` | JSON for the pop-up: `{scientific_name, common_name, hours, binomial, art, plate_url, heard: {count, max_conf, median_conf, first_heard, last_heard, first_local, last_local, cameras, by_hour[24], busiest_hour}, facts: {wikipedia, size, ebird, nearby}, links: {allaboutbirds}, pending, tz}`. Only names heard at `MIN_CONFIDENCE` or in the Audubon table; anything else is 404, a malformed name 400. `?hours=` |
| `GET /api/recent` | JSON `{hours, generated_at, species: [{scientific_name, common_name, last_heard, count, cameras, first_ever, has_plate}]}`. `has_plate`: a Fugleramme cut-out or Audubon plate is on disk. `?hours=` |
| `GET /plate/<stem>.png` | a cached cut-out or vignette as a 480 px PNG, from disk only |
| `GET /fonts/LibreBaskerville.ttf`, `-Italic.ttf`, `OFL.txt` | the page's type and its licence |
| `GET /static/page.js`, `/static/page.css` | the page's script and style (HTML responses carry a strict same-origin CSP) |
| `GET /attribution` | credits: Fugleramme plus its `ATTRIBUTION.md`, the Audubon plates fetched so far, the Wikipedia articles quoted so far, Wikidata, eBird, and Libre Baskerville |
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

### Pop-up

Tap a bird (or Tab to it and press Enter) and a field-guide card opens over
the page: the plate, common and scientific name, then **What we heard**
(detections in the window, best and typical confidence, first and last heard,
cameras, and a small chart of detections per hour of day) and **About the
bird** (a Wikipedia excerpt, Wikidata sizes when it has them, nearby eBird
reports, and links to eBird, All About Birds and Wikipedia). Esc, the close
button or a tap outside closes it; under 600 px wide it is a bottom sheet. The
60 s refresh swaps the image and its tap targets together and never touches an
open card. `/#species=<scientific name>` opens a card directly.

Facts are fetched lazily the first time a card opens, never during a render:
Wikidata (taxon name P225 to the English Wikipedia article, sizes, eBird taxon
ID), the Wikipedia summary of that article, and with `EBIRD_API_KEY` the eBird
taxonomy (species code) and recent reports within 25 km of
`LATITUDE`/`LONGITUDE`. A card waits at most 2.5 s for them; anything still
running shows on a follow-up a few seconds later. Results are cached per
source and species under `FACTS_DIR` (`/data/facts`): 30 days, a miss 24 h,
an error 1 h, nearby reports 6 h. Without a key no eBird request is made and
the eBird link comes from Wikidata. Times and hours are local to `TZ`
(default `America/Los_Angeles`; an unknown zone logs a warning and falls back,
it never disables the server). Wikidata covers sizes for only about 60% of
species, so many cards have no size line.

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

Pop-up notes are excerpts from [Wikipedia](https://en.wikipedia.org/) articles,
licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each
card links its article and `/attribution` lists every article quoted so far.
Sizes are from [Wikidata](https://www.wikidata.org/) (CC0). Species links and
nearby reports are data from [eBird.org](https://ebird.org) (Cornell Lab of
Ornithology), credited where shown as the eBird API terms ask. Wikimedia
requests send the same User-Agent. The page's type, Libre Baskerville, is
served from `/fonts/` under the SIL Open Font License 1.1 (`/fonts/OFL.txt`).

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
| `FACTS_FETCH` | `1` | `1` = fetch pop-up facts from Wikidata, Wikipedia and eBird; `0` = serve only what is already cached, no outbound requests |
| `FACTS_DIR` | `$DATA_DIR/facts` | pop-up facts cache, a few KB per species plus a ~600 KB eBird taxonomy |
| `EBIRD_API_KEY` | unset | secret. Unset = no eBird requests (the eBird link then comes from Wikidata, and there are no nearby reports). Sent only as the `X-eBirdApiToken` header to api.ebird.org, never logged or shown. On the fleet it lives in `~/.config/fleet/birdlisten.env`, never in git. A malformed key logs a warning and turns eBird off |
| `LATITUDE`, `LONGITUDE` | unset | also used (rounded to 0.01°) for nearby eBird reports; invalid or missing turns only those off |
| `TZ` | `America/Los_Angeles` | zone for the pop-up's times and hour chart; unknown values warn and fall back |

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
  test_birdlisten.py test_stream.py test_frame.py test_serve.py test_build_audubon_map.py test_facts.py test_taxa.py   # arm64 macOS
```

The stream-capture tests (`test_stream.py`) drive the real supervisors,
queue and worker against a fake ffmpeg that writes real WAV segments, so they
run without ffmpeg. The one test that needs a real ffmpeg is skipped when
none is installed. CI covers it in the built image with
`python stream.py --selftest`, which cuts a 7 s test tone into clips with the
exact stream-mode ffmpeg flags and fails the build if the result is wrong.

The second form exists because `tflite-runtime` has no macOS arm64 wheel, so
the project environment cannot resolve there; the tests only need stdlib plus
Pillow (`birdnetlib` is imported lazily by the analyzer). Both must pass. No
test touches the network: artwork fetches are monkeypatched to 404, and the
build script's fetch layer is monkeypatched too. The vignette tests use
synthetic sheets plus three public-domain 480 px Havell thumbnails in
`tests/fixtures/audubon/` (plates 8, 362, 376). Fact fetches
(`facts.http_get`) are refused too; the facts tests use fake upstreams.
`test_render_matches_golden` pins the collage pixels to hashes recorded by
`tools/golden_render.py` before the pop-up work.

Manual checks, not part of pytest:

```
# browser checks of the pop-up and screenshots (Playwright 1.59.0, chromium 1217)
uv run --no-project --python 3.11 --with playwright==1.59.0 --with pillow==12.3.0 python tools/popup_shots.py
# real-network fact coverage for the heard species
uv run --no-project --python 3.11 --with pillow==12.3.0 python tools/facts_coverage.py
```

## Ideas not built

Stream capture follow-ups, worth building only if the summaries call for them:

* A silence gate: skip BirdNET for clips whose peak is below a threshold.
  Worth it only if `duty` stays above about 0.6 or clips are dropped. At the
  measured 0.16 it saves nothing and risks skipping faint distant calls.
* More than one analysis worker, if `duty` above 0.9 or drops persist after
  a silence gate and the VM has 400 MB to spare per extra BirdNET model. The
  ntfy cooldown would then need a single notifier.
* `-allowed_media_types audio` on the RTSP input, so the NVR never sets up
  video. Needs testing against the real NVR.
* A status file in `SEGMENT_DIR`, written at every summary, so `--check` or
  `exec` can print live drop counts.

Other ideas:

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
| `birdlisten.py` | the app; `main() -> int`, `--check`, `--dry-run`, `--report N`; round-robin capture |
| `stream.py` | stream capture: one ffmpeg supervisor per camera, the drop-oldest queue, the analysis worker, the summary line, shutdown and crash safety; `python stream.py --selftest` for CI |
| `frame.py` | collage: recent-species query, Fugleramme and Audubon plate caches, vignette processing, packer, Pillow renderer, render cache |
| `audubon.json` | BirdNET scientific name → Havell plate on Commons (file, page, credit); generated, committed |
| `tools/build_audubon_map.py` | offline builder of `audubon.json`; not in the image |
| `serve.py` | the collage HTTP server; `start_from_env()` is what `loop.py` calls |
| `facts.py` | pop-up facts: Wikidata, Wikipedia and eBird fetching, single flight, cache under `FACTS_DIR` |
| `taxa.py` | `is_bird()`: BirdNET's non-bird labels and genera, kept off the page and ntfy |
| `static/` | `page.js` (tap targets, refresh swap, pop-up card) and `page.css` |
| `tools/facts_coverage.py`, `tools/popup_shots.py`, `tools/golden_render.py` | manual checks (real-network fact coverage, browser checks and screenshots, collage golden hashes); not in the image |
| `loop.py` | container entrypoint (fleet template), runs `main()` back to back (in stream mode `main()` returns only at shutdown or after a fatal error); starts the server when `SERVE_PORT` is set |
| `Dockerfile` | fleet template + Python 3.11 + Debian ffmpeg/libsndfile |
| `pyproject.toml`, `uv.lock` | birdnetlib 0.18, tflite-runtime 2.14, numpy<2, pillow 12.3 |
| `compose.yaml` | local dev; production compose lives in the fleet repo |
| `.github/workflows/build.yml` | build + push `ghcr.io/joekraemer/birdlisten:main`, then a smoke test and the segment muxer self-test |
| `test_birdlisten.py`, `test_stream.py`, `test_frame.py`, `test_serve.py`, `test_build_audubon_map.py`, `test_facts.py`, `test_taxa.py` | see Tests |
| `tests/fixtures/audubon/` | three Havell thumbnails for the vignette tests |
