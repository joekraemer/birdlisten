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
synthetic clip runs through analysis and storage; 12 unit tests cover config
parsing, dedupe, cooldown, storage, and the per-pass error handling. NOT yet
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
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--check" app     # config, ffmpeg, model
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--dry-run" app   # one real capture+analysis, no writes
   docker compose run --rm -e LOOP_ONCE=1 app                            # one real pass
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--report 7" app  # what was heard this week
   ```
6. Fleet: uncomment the `birdlisten` block in `fleet/compose.yaml`, create
   `~/.config/fleet/birdlisten.env` on the Mac with the same content as `.env`,
   push fleet.

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
  already removes implausible species; the remaining false positives are
  usually mechanical noise scored as a bird at 0.3–0.5.
* `CLIP_SECONDS` 30 gives ten 3-second BirdNET windows per camera per pass.
  Longer clips catch more but delay notifications.
* CPU: on the 2017 Intel MacBook a 30 s clip analyzes in a few seconds.
  Two cameras is comfortable; six would still be fine.

## Ideas not built

* Write detections to a Notion database (the movie-review pattern) or a
  weekly digest to ntfy.
* Per-camera `MIN_CONFIDENCE`, since a camera near a road is noisier.
* Trim the image: birdnetlib depends on matplotlib and librosa pulls in
  scikit-learn, making it ~2.5 GB. A slimmer inference path (`ai-edge-litert`
  + hand-rolled spectrogram) would be ~300 MB but is real work.
* Detect and alert on *new* species for the yard (first ever sighting) as a
  separate notification tier.

## Files

| file | role |
|---|---|
| `birdlisten.py` | the app; `main() -> int`, `--check`, `--dry-run`, `--report N` |
| `loop.py` | container entrypoint (fleet template), runs `main()` back to back |
| `Dockerfile` | fleet template + Python 3.11 + Debian ffmpeg/libsndfile |
| `pyproject.toml`, `uv.lock` | birdnetlib 0.18, tflite-runtime 2.14, numpy<2 |
| `compose.yaml` | local dev; production compose lives in the fleet repo |
| `.github/workflows/build.yml` | build + push `ghcr.io/joekraemer/birdlisten:main` |
| `test_birdlisten.py` | `uv run --group dev pytest -q` |
