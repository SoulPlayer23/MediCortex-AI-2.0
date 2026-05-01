# MediCortex AI 2.0 — Deployment Plan

> **Status:** Authored 2026-05-01, revised 2026-05-01 for Tailscale-only access. Persistent reference — survives session boundaries.
> **Target scale:** ~10 concurrent users on a private Tailscale network (internal/thesis tool).
> **Access model:** All 10 users are on the tailnet. The SPA is public on GitHub Pages but the API is reachable **only** over Tailscale — no Cloudflare Tunnel, no port-forwarding, no public exposure of the homeserver.
> **Topology:**
> - **Frontend SPA** → GitHub Pages (this repo, public).
> - **Orchestrator (FastAPI)** → Docker container on homeserver, exposed on the Tailscale interface only.
> - **Data services (Postgres, ArangoDB, Redis, MinIO)** → already running natively on the homeserver, **left as-is**.
> - **Ollama / Gemma 4 e2b** → already running natively on the homeserver, **left as-is**. (7.2 GB model, partial GPU offload on GTX 1650 Super 4 GB + 16 GB system RAM — adequate for 10 users.)
> - **MedGemma 1.5 4B** → RunPod Serverless container, scale-to-zero.

> **What gets Dockerized:** exactly two things — the orchestrator (homeserver) and the RunPod worker. Everything else stays native.

---

## 0. Hardware Inventory

| Asset | Spec | Role |
|---|---|---|
| **Homeserver** | Ryzen 5 3500 (6c/6t), 16 GB DDR4, GTX 1650 Super (4 GB VRAM), 750 GB SSD | Native: Postgres, ArangoDB, Redis, MinIO, Ollama (Gemma 4 e2b 7.2 GB, partial-GPU). Docker: orchestrator. |
| **Laptop** | (current MedGemma + dev host) | Dev machine only after RunPod is live. |
| **RunPod Serverless** | A4000 16 GB, scale-to-zero | MedGemma 1.5 4B inference only. |
| **GitHub Pages** | static hosting | React/Vite SPA (public, but only functional when browser is on the tailnet). |
| **Tailscale** | free plan, MagicDNS + HTTPS certs | Network access layer for all users. Replaces Cloudflare Tunnel. |

**Resource budget (homeserver, all services running):**

| Component | RAM | GPU |
|---|---|---|
| Postgres + ArangoDB + Redis + MinIO (already running) | ~3–4 GB | — |
| Ollama serving `gemma4:e2b` 7.2 GB | ~3 GB sys (partial offload) | ~3.5 GB / 4 GB VRAM |
| Orchestrator container (new) | ~1.5–2 GB | — |
| OS + buffers | ~2 GB | — |
| **Total** | **~9–11 GB** of 16 GB | **~3.5 GB** of 4 GB |

VRAM is the tight slot — keep `gemma4:e2b` as the only Ollama model loaded. If you ever want to swap to a bigger Ollama model, route it through RunPod instead.

---

## 1. Endpoints

No public domains needed for the API. Tailscale supplies a stable HTTPS endpoint inside the tailnet.

| Endpoint | Where | Reachable from |
|---|---|---|
| `https://<user>.github.io/MediCortex-AI-2.0/` (or custom domain) | GitHub Pages | public internet (but useless without tailnet) |
| `https://homeserver.tail<XXXXX>.ts.net:8001` | orchestrator container on homeserver | tailnet only |
| `homeserver:5432 / 6379 / 8529 / 9000 / 11434` | native services on homeserver | tailnet only (already working) |

**Action:**
1. Find your tailnet domain: `tailscale status` on homeserver shows the FQDN (e.g. `homeserver.tail1234a.ts.net`).
2. Issue a TLS cert: `sudo tailscale cert homeserver.tail1234a.ts.net` — produces `homeserver.tail1234a.ts.net.crt` + `.key` valid for 90 days, auto-renewable.
3. Optionally enable Tailscale's MagicDNS short names so users hit `https://homeserver:8001` directly inside the tailnet (still requires the cert).
4. Onboard all 10 users to the tailnet (Tailscale Free supports 100 users / 3 users on Personal — verify your plan; Personal Pro is $5/user/mo if needed). For a thesis cohort, Tailscale's free Personal plan is usually enough.

---

## 2. Models on RunPod Serverless

**Goal:** MedGemma 1.5 4B served behind a stable HTTPS URL with scale-to-zero billing. Gemma 4 e2b stays on the homeserver Ollama (already working, free, fits in 4 GB VRAM).

### 2.1 Image build

Repo layout for the RunPod worker (create a new sibling repo `medicortex-runpod-worker`):

```
medicortex-runpod-worker/
├── Dockerfile
├── handler.py           # RunPod serverless handler
├── requirements.txt
├── start.sh             # downloads model on cold-boot, starts FastAPI shim
└── README.md
```

**`Dockerfile`** (skeleton):

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3.11 python3-pip git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY handler.py start.sh ./
RUN chmod +x start.sh
CMD ["./start.sh"]
```

**`requirements.txt`:**

```
torch==2.4.0
transformers==4.45.0
accelerate==0.34.0
fastapi==0.115.0
uvicorn==0.30.0
runpod==1.7.0
pillow==10.4.0
```

**`handler.py`** — exposes the same `/predict` contract that `medgemma-host` currently provides on `localhost:8000`. Use `transformers` + `bitsandbytes` 4-bit quant to fit in A4000 16 GB with VLM headroom. Reference signature (port your existing `medgemma-host/main.py` logic verbatim):

```python
import runpod, base64, io
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

MODEL_ID = "google/medgemma-4b-it"  # MedGemma 1.5 4B (multimodal)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

def handler(event):
    inp = event["input"]
    prompt = inp["prompt"]
    image_b64 = inp.get("image_base64")
    images = [Image.open(io.BytesIO(base64.b64decode(image_b64)))] if image_b64 else None
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=inp.get("max_tokens", 1024))
    return {"text": processor.batch_decode(out, skip_special_tokens=True)[0]}

runpod.serverless.start({"handler": handler})
```

> **Important — adapt the request/response shape to match your existing `MedGemmaLLM._call()` in `specialized_agents/medgemma_llm.py`.** Whatever JSON keys it sends today (`prompt`, `image_base64`, `max_tokens`, …) must be honored verbatim so no client-side changes are needed beyond the URL swap.

### 2.2 Deploy on RunPod

1. Create a public Docker image: `docker build -t <user>/medicortex-runpod:1.0 . && docker push`.
2. RunPod Console → Serverless → New Endpoint:
   - Container image: `<user>/medicortex-runpod:1.0`
   - GPU: **A4000 (16 GB)** — sufficient for MedGemma 1.5 4B at bf16. A5000 if you add larger contexts.
   - Min workers: **0** (scale to zero)
   - Max workers: **2**
   - Idle timeout: **300s** (5 min — balances cost vs. cold start)
   - Container disk: 30 GB
3. Copy the endpoint URL (`https://api.runpod.ai/v2/<endpoint-id>/runsync`) and an API key.
4. Set on the homeserver `.env`:
   ```
   MEDGEMMA_API_URL=https://api.runpod.ai/v2/<endpoint-id>/runsync
   RUNPOD_API_KEY=<key>
   ```
5. **Cold-start mitigation (DEPLOY-2):** add a keepwarm cron on the homeserver — a `curl` against the endpoint every 4 minutes during business hours. Cheaper than always-on (RunPod bills per second the worker is alive, not per request).

### 2.3 Code changes for RunPod

In `specialized_agents/medgemma_llm.py`:
- Add `Authorization: Bearer ${RUNPOD_API_KEY}` header.
- RunPod's `/runsync` returns `{"output": {"text": "..."}}` — wrap the response unpack accordingly. Keep the existing fallback to Gemma 4 unchanged.
- Reduce `timeout` from 120 → 30s (DEPLOY-2). Trigger Gemma 4 fallback faster on cold-start.

### 2.4 Verification

- `curl -X POST $MEDGEMMA_API_URL -H "Authorization: Bearer $RUNPOD_API_KEY" -d '{"input":{"prompt":"Test"}}'` returns 200 with `output.text`.
- Cold start (after >5 min idle): first request completes in ≤45s; subsequent requests in <10s.
- `Test.md` T7.3 passes (Gemma 4 fallback fires within 30s when MedGemma endpoint is offline).

---

## 3. Backend on Homeserver

### 3.1 Hardening checklist (do BEFORE first public exposure)

These map to existing tickets in `Todo.md` — do them in order:

- [ ] **SEC-3** — `DEBUG=False` default in `config.py`; empty all secret defaults; add Pydantic validator that raises if any prod secret is empty.
- [ ] Move `Ollama API Key MediCortex-AI-2.0.txt` out of repo, add to `.gitignore`, **rotate the key on Ollama**.
- [ ] **DEPLOY-3** — `ALLOWED_ORIGINS: list[str]` env var; CORS middleware uses it instead of `["*"]`.
- [ ] **DEPLOY-2** — keepwarm + 30s timeout (see §2.5).
- [ ] **SEC-1** — 50 MB upload cap, 100 MB download cap.
- [ ] **BUG-5** — disconnect-safe DB save in `event_generator`.
- [ ] **OPS-1** — DB pool `pool_size=20, max_overflow=10, pool_pre_ping=True`.
- [ ] **OPS-3** — assert `WEB_CONCURRENCY=1` at startup until Redis-backed `ACTIVE_STREAMS` lands.
- [ ] **OPS-5** — `slowapi` rate limits.
- [ ] **OBS-2** — split `/health` into `/livez` + `/readyz`.

### 3.2 Dockerizing the orchestrator (only)

Postgres, ArangoDB, Redis, MinIO, and Ollama already run natively on the homeserver and work over Tailscale. **Leave them alone.** The orchestrator is the only new thing that lands on the homeserver, and it's the one that benefits most from containerization (heavy Python deps, clean rollback).

**Repo:** add a `Dockerfile` at the repo root.

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps for Presidio, asyncpg, Pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg   # Presidio default analyzer

COPY . .

EXPOSE 8001
CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

**`docker-compose.prod.yml`** (at repo root):

```yaml
version: "3.9"

services:
  orchestrator:
    build: .
    image: medicortex/orchestrator:latest
    restart: unless-stopped
    env_file: .env.prod
    # Bind to the Tailscale interface IP only — never to 0.0.0.0
    ports:
      - "100.x.x.x:8001:8001"   # replace 100.x.x.x with `tailscale ip -4`
    extra_hosts:
      - "host.docker.internal:host-gateway"   # so container can reach native services on host
    volumes:
      - ./knowledge_core/assets:/app/knowledge_core/assets:ro
      - ./logs:/app/logs
```

Single worker until OPS-3 (Redis-backed `ACTIVE_STREAMS`) lands.

**Why bind to the Tailscale IP (`100.x.x.x`) instead of `0.0.0.0`:** belt-and-braces — even if your home router accidentally forwards 8001, the container won't accept it. Find the IP with `tailscale ip -4` on the homeserver. Reserve it in the Tailscale admin console so it's stable across reboots.

**Native services keep working unchanged:**
- Postgres on `homeserver:5432`
- ArangoDB on `homeserver:8529`
- Redis on `homeserver:6379`
- MinIO on `homeserver:9000`
- Ollama on `homeserver:11434`

The container reaches them via `host.docker.internal` (set by `extra_hosts` above). See §3.3 for the env values.

**Systemd auto-start unit** for the compose stack (`/etc/systemd/system/medicortex.service`):

```ini
[Unit]
Description=MediCortex orchestrator (Docker Compose)
Requires=docker.service tailscaled.service
After=docker.service tailscaled.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=true
WorkingDirectory=/home/<user>/MediCortex-AI-2.0
ExecStart=/usr/bin/docker compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.prod.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now medicortex` — survives reboots.

### 3.3 `.env.prod` (homeserver — never commit; .gitignore it)

```
DEBUG=false
ALLOWED_ORIGINS=https://<gh-user>.github.io

# Postgres (native on host, reached via host.docker.internal)
DATABASE_URL=postgresql+asyncpg://medicortex:<pw>@host.docker.internal:5432/medicortex

# ArangoDB (native on host)
ARANGODB_HOST=host.docker.internal
ARANGODB_PORT=8529
ARANGODB_USER=root
ARANGODB_PASSWORD=<pw>

# Redis (native on host)
REDIS_URL=redis://host.docker.internal:6379/0

# MinIO (native on host). Presigned URL host MUST be reachable from the user's browser
# (i.e. the Tailscale-resolvable hostname), not the Docker-internal hostname.
MINIO_URL=http://host.docker.internal:9000             # used by the container internally
MINIO_PUBLIC_URL=https://homeserver.tail<XXXXX>.ts.net:9000   # baked into presigned URLs
MINIO_ACCESS_KEY=<key>
MINIO_SECRET_KEY=<key>
MINIO_BUCKET=medicortex-uploads

# Models — Gemma 4 stays on host Ollama
OLLAMA_CLOUD_URL=http://host.docker.internal:11434
OLLAMA_CLOUD_MODEL=gemma4:e2b

# MedGemma → RunPod
MEDGEMMA_API_URL=https://api.runpod.ai/v2/<endpoint-id>/runsync
RUNPOD_API_KEY=<key>
MEDGEMMA_KEEPWARM_URL=https://api.runpod.ai/v2/<endpoint-id>/health

# Judge
GROQ_API_KEY=<key>

# Concurrency / pool
WEB_CONCURRENCY=1
SQLALCHEMY_POOL_SIZE=20
```

> The MinIO `MINIO_PUBLIC_URL` distinction matters: presigned URLs include the host that's signed. If the URL embeds `host.docker.internal` or `localhost`, the user's browser cannot resolve it. Use the tailnet FQDN. Same applies if a user's browser ever uploads/downloads via presigned URL directly. Code in `services/minio_service.py` likely already supports a public-URL override; if not, add one (small change).

### 3.4 HTTPS via Tailscale (replaces Cloudflare Tunnel)

```bash
# On homeserver, one-time
sudo tailscale cert homeserver.tail<XXXXX>.ts.net
# produces homeserver.tail<XXXXX>.ts.net.crt and .key in CWD
```

Two options for terminating TLS:

**Option A — Caddy in front of the orchestrator (recommended).**
Add a tiny Caddy container alongside the orchestrator that owns the certs:

```yaml
# docker-compose.prod.yml — add this service alongside `orchestrator`
caddy:
  image: caddy:2-alpine
  restart: unless-stopped
  ports:
    - "100.x.x.x:443:443"   # tailnet IP only
  volumes:
    - ./Caddyfile:/etc/caddy/Caddyfile:ro
    - ./certs:/certs:ro     # mount the tailscale-issued cert/key here
  depends_on: [orchestrator]
```

`Caddyfile`:
```
homeserver.tail<XXXXX>.ts.net {
    tls /certs/homeserver.tail<XXXXX>.ts.net.crt /certs/homeserver.tail<XXXXX>.ts.net.key
    reverse_proxy orchestrator:8001
    # SSE needs streaming-friendly defaults — Caddy handles this by default
    encode gzip
}
```

Now users hit `https://homeserver.tail<XXXXX>.ts.net` (port 443, no `:8001` exposed). Renew the cert via a monthly cron: `tailscale cert homeserver.tail<XXXXX>.ts.net && docker compose restart caddy`.

**Option B — let `tailscale serve` do it.** Tailscale has a built-in reverse proxy:
```bash
sudo tailscale serve --bg --https=443 http://localhost:8001
```
Simpler, no Caddy. The tradeoff: less flexibility for headers, gzip, multiple routes. For a single-route deploy, Option B is fine. **Pick B unless you need Caddy's flexibility.**

### 3.5 Power, uptime, and resilience for residential hosting

This is the actual weak link, not CPU/RAM:

- **UPS** — even a 600 VA APC unit ($60) gives 10–15 min on this load, enough to survive brownouts and to do a clean shutdown via NUT.
- **Auto-restart on boot** — `docker compose -f docker-compose.prod.yml up -d` should be wired to a systemd unit (`After=docker.service`). Same for the host Ollama and `cloudflared`.
- **Daily backups** — a simple cron:
  - `pg_dump` → `~/backups/pg_$(date +%F).sql.gz`
  - `arangodump` → `~/backups/arango_$(date +%F).tar.gz`
  - `mc mirror minio/medicortex-uploads ~/backups/minio/`
  - Optional: `rclone sync ~/backups/ b2:medicortex-backups/` to off-site storage (Backblaze B2 free 10 GB).
- **Heartbeat** — Better Stack Free monitor against `https://api.medicortex.<domain>/livez` every 60s; SMS/email on 2 consecutive failures.
- **ISP fallback** — if uptime matters: a 4G/5G LTE backup router with failover. Optional, ~$30/mo for a SIM.

### 3.6 Verification

- `curl https://api.medicortex.<domain>/livez` returns 200 from any external network.
- `curl https://api.medicortex.<domain>/readyz` returns 200 with all sub-checks green (Postgres, ArangoDB, Redis, MinIO, Ollama, MedGemma).
- A full `/chat/stream` request from outside the LAN streams SSE events end-to-end and the assistant message is persisted (verify via DB query).
- Test Suite 7 (Test.md T7.1–T7.9) passes against the deployed backend.

---

## 4. Frontend on GitHub Pages

This repo is already on GitHub. The frontend lives at `frontend/` (React 19 + Vite + Tailwind).

### 4.1 Code changes (DEPLOY-4)

Replace every `http://localhost:8001` in:
- `frontend/src/components/ChatArea.tsx:107,142`
- `frontend/src/components/InputArea.tsx:34`
- `frontend/src/components/Sidebar.tsx` (if present)

with:

```typescript
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8001';
// then: `${API_BASE}/chats/${id}` etc.
```

Add two env files:

`frontend/.env.development`:
```
VITE_API_BASE_URL=http://localhost:8001
```

`frontend/.env.production`:
```
# Actions secret VITE_API_BASE_URL overrides this at build time.
# Tailscale FQDN — only resolvable from the tailnet, by design.
VITE_API_BASE_URL=https://homeserver.tail<XXXXX>.ts.net
```

### 4.2 Vite base path

GitHub Pages serves project sites under `https://<user>.github.io/<repo>/`. Set `base` in `vite.config.ts`:

```typescript
export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? '/',
  // ...
});
```

If you wire a custom domain (`medicortex.<domain>`), keep `base: '/'`. Without a custom domain, set `VITE_BASE_PATH=/MediCortex-AI-2.0/` in the workflow.

### 4.3 GitHub Actions workflow

Create `.github/workflows/deploy-frontend.yml`:

```yaml
name: Deploy Frontend
on:
  push:
    branches: [main]
    paths: ['frontend/**', '.github/workflows/deploy-frontend.yml']
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
        working-directory: frontend
      - run: npm run build
        working-directory: frontend
        env:
          VITE_API_BASE_URL: ${{ secrets.VITE_API_BASE_URL }}
          # VITE_BASE_PATH: /MediCortex-AI-2.0/   # uncomment if NOT using custom domain
      - uses: actions/upload-pages-artifact@v3
        with:
          path: frontend/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

### 4.4 GitHub repo configuration

1. **Settings → Pages** → Source: **GitHub Actions**.
2. **Settings → Secrets and variables → Actions → New repository secret:**
   - Name: `VITE_API_BASE_URL`
   - Value: `https://api.medicortex.<domain>`
3. **(Optional) Custom domain:** Settings → Pages → Custom domain → `medicortex.<domain>`. Add a CNAME record at Cloudflare: `medicortex` → `<user>.github.io`. Enable "Enforce HTTPS".
4. Add `frontend/public/CNAME` containing `medicortex.<domain>` (only if using custom domain — the deploy action picks this up).

### 4.5 SPA route handling on GitHub Pages

GitHub Pages doesn't support SPA fallback to `index.html`. If you use client-side routes (e.g. `/chat/<uuid>`), add a `frontend/public/404.html` that redirects to `/index.html#<original-path>` (the well-known [`spa-github-pages` trick](https://github.com/rafgraph/spa-github-pages)). Without this, hard-refreshing on a route returns 404. Already noted as UI-4 risk — the existing `App.tsx` lazy initializer reads `window.location.pathname`, so as long as `index.html` *is* served, it works.

### 4.6 Verification

- Push to `main`. Actions tab shows the workflow green.
- `https://medicortex.<domain>` loads the SPA, shows the welcome screen.
- Chrome DevTools → Network → `EventSource` request goes to `https://api.medicortex.<domain>/chat/stream` (not localhost).
- A new chat round-trips end-to-end with sources, judge metadata, and SSE thinking steps.

---

## 5. Order of Operations (do in this exact order)

1. **Tailscale prep** — confirm all 10 users are on the tailnet; capture the homeserver's tailnet FQDN (`tailscale status`).
2. **Code prep on a feature branch:**
   - SEC-3, DEPLOY-3, DEPLOY-4, SEC-1, BUG-5 (see `Todo.md`).
   - Add MinIO `PUBLIC_URL` override if not already present.
   - Add `Dockerfile` + `docker-compose.prod.yml`.
   - Test locally with `start_medicortex.sh`.
   - Merge to `main`.
3. **RunPod endpoint live** (§2). Update `MEDGEMMA_API_URL` and `RUNPOD_API_KEY`.
4. **Homeserver:**
   - Install Docker if not already (`curl -fsSL https://get.docker.com | sh`).
   - Native Postgres / ArangoDB / Redis / MinIO / Ollama: leave running as today.
   - Issue the Tailscale cert (`tailscale cert homeserver.tail<XXXXX>.ts.net`) OR enable `tailscale serve --bg --https=443 http://localhost:8001`.
   - `docker compose -f docker-compose.prod.yml up -d --build`.
   - One-shot: `docker compose exec orchestrator python -m database.init_db` (only if schema changed).
   - One-shot: `docker compose exec orchestrator python3 -m knowledge_core.build_fast_assets` (or run on host and bind-mount).
   - Enable systemd unit (§3.2) so it survives reboots.
5. **GitHub Pages:**
   - Add the workflow + repo secret `VITE_API_BASE_URL=https://homeserver.tail<XXXXX>.ts.net`.
   - Push → confirm SPA loads.
6. **End-to-end test (from a machine on the tailnet):** Test Suite 7 from Test.md.
7. **Monitoring:** Better Stack heartbeat against the Tailscale URL of `/livez` (Better Stack supports custom DNS via their probe locations — or run a self-hosted Uptime Kuma container on the homeserver). Daily backup cron. UPS connected.

---

## 6. Cost Summary (monthly, 10 users)

| Item | Cost |
|---|---|
| GitHub Pages | $0 |
| Tailscale (free Personal plan; Personal Pro $5/user if cohort exceeds free limits) | $0–5 |
| Domain | not required |
| RunPod Serverless A4000 (scale-to-zero, ~200 req/day × 30s + keepwarm pings) | ~$5–15 |
| Homeserver electricity (~80 W avg) | ~$5–8 |
| Backups (Backblaze B2 ~10 GB) | $0 free tier |
| **Total** | **~$10–28/mo** |

---

## 7. When to Migrate Off Homeserver

The current setup is fine for 10 users at internal/thesis stage. Migrate when any of these happens:

- **Uptime SLA > 99%** is required → move backend to Hetzner CPX21 (~$8/mo).
- **Real HIPAA BAA** is needed → AWS-only, full re-platform.
- **>50 concurrent users** → split orchestrator from data services; horizontal scale needs Redis-backed `ACTIVE_STREAMS` (OPS-3) regardless.
- **Home internet upload < 20 Mbps** sustained — SSE + file uploads will stutter.

---

## 8. Open Risks (track these)

| Risk | Mitigation |
|---|---|
| Homeserver power outage during a thesis demo | UPS + monitoring + document the manual restart procedure |
| GTX 1650 Super VRAM exhausted by future Ollama model swap | Keep Gemma 4 e2b only on host; route any larger model via RunPod |
| Tailscale outage (rare; affects auth + relay) | Direct connections via DERP keep working for established sessions; document the rare-case impact |
| RunPod cold-start visible to users | DEPLOY-2 keepwarm + 30s fallback to Gemma 4 |
| ISP rotates IP / blocks outbound | Tailscale is outbound-only via DERP/WireGuard; IP rotation has no effect |
| Single SSD I/O bottleneck (Postgres + ArangoDB + MinIO + Ollama) | At 10 users, fine. Watch `iostat -xz 1` during peaks; add a second SSD if `await > 50ms` |
| Gemma 4 partial-GPU offload latency under burst | At 7.2 GB on a 4 GB VRAM card, Ollama spills to CPU. Acceptable at 10 users; if latency becomes an issue, switch Gemma 4 to a smaller quant (Q4_K_M ~1.5 GB fits fully on GPU) |
| Tailscale cert manual renewal every 90 days | Add a monthly `tailscale cert` cron to auto-renew before expiry |
