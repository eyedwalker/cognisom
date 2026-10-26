# Viewing a Cognisom run in Isaac Sim 6.0

This is the setup that works. It replaces the custom Kit extension and
MJPEG capture in `../kit_extension/`, which never produced a live RTX
frame of a Bio-USD stage. Verified 2026-10-26 on an A10G: the stock
Isaac Sim 6.0 browser viewer streamed a Cognisom run at 37–50 fps with
working navigation.

Nothing here is ours except two small files. The rest is NVIDIA's
published stack, run as documented.

| file | what it does |
|---|---|
| `open_cognisom_stage.py` | startup script: opens the stage, lights it for RTX, locks the home camera, places the free camera at the home view, plays the run |
| `docker-compose.override.yml` | passes that script to Isaac Sim. NVIDIA's `docker-compose.yml` is used unmodified |

## Bring-up

On an Ubuntu GPU host with Docker, the NVIDIA container toolkit and an
NGC login:

```bash
# 1. NVIDIA's compose files, unmodified
git clone --depth 1 --filter=blob:none --sparse https://github.com/isaac-sim/IsaacSim.git isaacsim-repo
cd isaacsim-repo && git sparse-checkout set tools/docker

# 2. Data directories MUST be owned by uid 1234, the container user.
#    Skip this and RTX cannot create its texture cache; the app never
#    reaches AppReady and the healthcheck stays "unhealthy".
sudo mkdir -p ~/docker/isaac-sim/{cache/main,cache/computecache,cache/kit,config,data,logs,pkg} ~/.cache/ov/hub
sudo chown -R 1234:1234 ~/docker/isaac-sim ~/.cache/ov/hub

# 3. Our stage and startup script, also owned by 1234
sudo mkdir -p ~/docker/isaac-sim/data/cognisom
sudo cp open_cognisom_stage.py <run>.usdc ~/docker/isaac-sim/data/cognisom/
sudo chown -R 1234:1234 ~/docker/isaac-sim/data/cognisom
cp docker-compose.override.yml isaacsim-repo/tools/docker/

# 4. Environment. ISAACSIM_HOST is the PUBLIC IP: Isaac Sim advertises it
#    to remote WebRTC clients, and the web viewer compiles it into its
#    JavaScript. Left at the default 127.0.0.1, a remote browser is told
#    to connect to itself.
cat > isaacsim-repo/tools/docker/.env <<ENV
ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0
ISAAC_SIM_DATA=/home/ubuntu/docker/isaac-sim
ISAACSIM_HUB_CACHE_PATH=/home/ubuntu/.cache/ov/hub
ISAACSIM_HOST=<public ip>
ENV

# 5. Up
cd isaacsim-repo && docker compose -p isim \
    -f tools/docker/docker-compose.yml \
    -f tools/docker/docker-compose.override.yml up --build -d
```

Then open `http://<public ip>:8210` in a Chromium-based browser. Only one
browser session can be connected at a time.

Ports, open to the viewer's address only: **8210/tcp** web viewer,
**49100/tcp** WebRTC signalling, **47998/udp** media.

## Things that cost a day each to learn

- **`--network=host` is required.** NVIDIA's docs say bridge networking
  with `-p` port publishing will not deliver video. This project ran
  bridge networking for months. Compose sets host networking for you.
- **The container runs as uid 1234.** Mounted directories owned by your
  login user break the RTX texture cache silently.
- **`ISAACSIM_HOST` must be the public IP**, and the web viewer bakes it
  in at build time. A stopped and restarted EC2 instance gets a NEW
  public IP, so after a restart, update `.env` and rebuild:
  `docker compose ... up --build -d`.
- **A write_run stage renders near black under RTX** without the
  startup script: its dome light is authored at 1.0 for Storm, and RTX
  needs about 1000. The script raises it in the session layer, so the
  file on disk is untouched.
- **Never `import omni.something` inside the async function** in the
  startup script. It makes `omni` a local name for the whole function,
  and the first line fails with `UnboundLocalError` before anything
  opens.

## Navigating

The viewer starts on Kit's free **Perspective** camera, placed exactly at
the run's home view. Drag freely.

- **Reset:** camera dropdown at the top of the viewport → **EventCam**.
  It is locked (`omni:kit:cameraLock`), so exploring cannot move it.
- **Lost:** pick Perspective, click **Cells** in the Stage panel, hover
  the viewport, press **F**.
- **Renderer:** keep **RTX – Real-Time 2.0**. "RTX – Interactive" is the
  path tracer: slow, noisy while it accumulates, and black around the
  subject.
- **Full reset:** restart the service, which re-runs the startup script:
  `docker compose -p isim -f tools/docker/docker-compose.yml -f tools/docker/docker-compose.override.yml up -d --force-recreate isaac-sim`

Known: Isaac Sim 6.0.0 logs an `AttributeError` from its own
`PlayerController` (`_suppress_non_select_viewport_interaction`). It is
NVIDIA's code and has not affected rendering. Their compose example now
points at 6.0.1, which may fix it.
