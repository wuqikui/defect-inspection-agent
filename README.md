# Multimodal Defect Inspection Agent

> An industrial surface-defect inspection full-stack system built on a **RAG rule vector knowledge base**, **local deep vision pipeline** (PatchCore + YOLO-World-S), and **optional cloud VLM judgment**.
> FastAPI (Python 3.12) backend · React + TypeScript (Vite) frontend · SQLite + ChromaDB · Docker for cross-platform deployment.
> PatchCore + YOLO-World-S ONNX models are bundled with the project (`backend/models/*.onnx`) — CPU inference, **no weight download required**.

中文文档见 [README.zh-CN.md](./README.zh-CN.md)。

---

## Table of Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Quick Start (Local)](#4-quick-start-local)
5. [Enabling VLM](#5-enabling-vlm)
6. [Docker Deployment](#6-docker-deployment)
7. [User Guide](#7-user-guide)
8. [Core Algorithms](#8-core-algorithms)
9. [API Reference](#9-api-reference)
10. [Project Structure](#10-project-structure)
11. [Learning Highlights](#11-learning-highlights)
12. [FAQ](#12-faq)

---

## 1. Overview

Conventional machine-vision inspection requires specialized models retrained for every defect type and production line, making rule updates expensive. This system turns a company's **illustrated inspection-standard documents (PDF/DOCX) into a searchable, reasoning-capable knowledge base**:

- Engineers upload rule documents → the system parses, chunks, embeds, and extracts structured rules;
- When documents contradict each other, every conflict is presented one-by-one for a human decision, guaranteeing a single operative standard;
- An industrial image of **any size** is scanned with a local **PatchCore global sentinel** to lock high-risk regions → **YOLO-World-S** zero-shot localization on dynamic tiles → **cloud VLM or offline heuristic judge** against RAG rules → **OpenCV pixel contours** → defects are drawn on the original image with rule-based evidence;
- Every inspection is stored and retrievable from history, with name-based search and pagination;
- **LLM API keys can be configured from the frontend with one click**, taking effect immediately without restarting the backend.

**Key property:** the system runs fully **offline without any API key** (local PatchCore + YOLO-World + heuristic judge), and automatically upgrades to cloud VLM judgment once a key is configured.

## 2. Features

| Module | Capabilities |
| --- | --- |
| Documents | PDF/DOCX upload (max 10 docs, 50 MB each); paragraph text + figure-caption recognition; table extraction; semantic chunking with overlap |
| RAG store | Zhipu / Qwen embeddings (1024-d) with a deterministic local hashing fallback; persistent ChromaDB (HNSW / cosine); enforced similarity threshold; page-level provenance |
| Rule understanding | Structured rule extraction with Zhipu GLM / Qwen / DeepSeek; tolerant JSON parsing; automatic index rebuild after embedding-provider switch |
| Conflicts | LLM-based detection of cross-document contradictions (e.g., clashing thresholds); one-by-one adjudication (A / B / custom); inspection is blocked until all conflicts are resolved |
| Local vision pipeline (CPU) | **PatchCore + ResNet18 sentinel** for full-image anomaly heatmap, filtering normal regions; **YOLO-World-S** zero-shot localization with inline defect labels on dynamic tiles; ONNX models bundled with the project — no download needed |
| Judgment | Cloud VLM (GLM-4V / Qwen VL) audits candidates against RAG rules, filtering false positives and correcting defect types; automatically falls back to **heuristic judge** (contrast significance + sentinel margin dual evidence) when no VLM key is configured |
| Pixel落地 | OpenCV Otsu + morphological contour extraction, working only inside final defect ROIs, mapped back to original coordinates |
| API key management | **Frontend one-click save** (SQLite local storage), takes effect immediately; masked display (e.g. `sk-1****abcd`); set as default text / vision model with instant switch-over; automatic fallback when a key is removed |
| Output | Pass/fail verdict, bounding boxes, translucent masks, confidence, coordinates, rule evidence (similarity / source document / page), narrative summary; **SVG overlay auto-adapts to image size**, annotations always land on true defect positions |
| History | SQLite persistence; thumbnail list + full result drill-down; **name/summary search + 20-item pagination** |
| Engineering | Unified exception handling, magic-byte and path-traversal protection, decompression-bomb guard, lazy per-tile memory usage, live progress reporting, CORS, health checks, response timing header |

## 3. Architecture

```
Browser (React + TypeScript + Vite)
  Documents │ Conflicts │ Detection (progress polling) │ SVG overlay │ History (search+page) │ Model status (one-click key)
                                  │ HTTP / multipart
                                  ▼
FastAPI (api layer: documents / conflicts / inspection / system / providers/{name}/key)
Services:
  parser(PDF/DOCX) → chunker ─┐
  conflict service (LLM/heuristic)
  inspection orchestrator (progress / conflict gate / evidence / history)
        └─ vision: PatchCore sentinel → dynamic tile slicing → YOLO-World → VLM/heuristic judge → OpenCV contour → annotator
Model layer (facade pattern):
  BaseLLM:  Zhipu GLM │ Qwen │ DeepSeek │ Mock (offline)
  EmbeddingClient: Zhipu │ Qwen │ local hashing
  Local vision: PatchCore(ResNet18-ONNX) │ YOLO-World-S(ONNX)
Key source: frontend SQLite save > .env file
Storage:
  SQLite: documents / chunk source text / rules / conflicts / jobs / API keys / provider preferences
  ChromaDB: rule chunk embeddings (HNSW + cosine)
  backend/models/: resnet18_features.onnx + yolov8s-worldv2.onnx (bundled with project)
```

Design principles:

- **Facade-based access**: routers call service singletons only; models and the vector store are obtained through factories — easy to swap and test;
- **Interface-first**: `BaseLLM`, `EmbeddingClient`, and `BaseParser` contracts mean adding a provider or format is a new adapter, not a rewrite;
- **Online/offline dual track**: every external AI capability has a deterministic local fallback.
- **API key hot reload**: frontend save → `lru_cache` clear + embedding provider switch + RAG background reindex, zero restart needed;
- **CPU-friendly vision**: PatchCore + YOLO-World-S are ONNX models bundled with the repo — no external download, runs on CPU.

## 4. Quick Start (Local)

### 4.1 Prerequisites

- Python ≥ 3.10 (validated on **Python 3.12**, Windows 11)
- Node.js ≥ 18 (validated on Node 24)
- No external database or Docker required for local mode
- Local vision pipeline: CPU only, models already bundled under `backend/models/`

### 4.2 Backend

```bash
cd defect-inspection-agent/backend

# 1) Create a virtual environment
conda create -y -p ./.conda-env python=3.12
conda activate ./.conda-env
#   or: python -m venv .venv && source .venv/bin/activate   (Linux/macOS)
#       .venv\Scripts\Activate.ps1                            (Windows)

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Pre-configure LLM API keys via .env
cp .env.example .env   # Windows: copy .env.example .env
#   Set any of ZHIPU_API_KEY / QWEN_API_KEY / DEEPSEEK_API_KEY
#   Or configure them from the frontend (recommended, no restart needed)

# 4) Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/health>

### 4.3 Frontend

```bash
cd defect-inspection-agent/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to `localhost:8000`.

### 4.4 Try it

1. Upload one or more PDF/DOCX standard documents on the **Documents** tab;
2. Resolve every item in the **Conflict Center** (inspection stays blocked otherwise);
3. Upload one product image on the **Inspection** tab and watch progress (sentinel scanning → high-risk regions → dynamic tiles → localization → judgment → contours);
4. Review boxes, confidence, and rule evidence;
5. Browse past runs in **History** (with name search + pagination); check active providers in **Model Status**, where you can also configure LLM API keys with one click.

## 5. Enabling VLM

Two ways, pick whichever suits you:

### Option A: Frontend one-click (recommended, no restart)

1. Open the **Model Status** tab in the frontend;
2. Paste your API key under Zhipu GLM or Qwen, click **Save**;
3. Takes effect immediately — the banner at the top switches to online mode;
4. Click **Set as Vision** to fix that provider as the VLM judgment model.

### Option B: `.env` environment variables

```bash
# Edit backend/.env, fill in any one of:
ZHIPU_API_KEY=sk-xxxx
QWEN_API_KEY=sk-xxxx
DEEPSEEK_API_KEY=sk-xxxx
# Optional: pin the provider
LLM_PROVIDER=zhipu        # or qwen / deepseek / leave empty for auto-select
VISION_PROVIDER=zhipu     # or qwen / leave empty for auto-select
```

Priority: **frontend SQLite save > .env environment variable**. Clearing the frontend save automatically falls back to `.env`.

## 6. Docker Deployment

One-command startup (Nginx on :80, API on :8000):

```bash
cd defect-inspection-agent
# optional: cp backend/.env.example backend/.env and fill in keys
docker compose up -d --build
# open http://localhost
```

- All backend data (documents, SQLite, ChromaDB) persists in the `backend-data` volume;
- Images are based on `python:3.12-slim` and `node:20-alpine / nginx:1.27-alpine` (x86_64 / arm64);
- Local ONNX models are bundled — containers start with PatchCore + YOLO-World-S ready out of the box.

## 7. User Guide

### 7.1 Documents & conflicts

- Only `.pdf` / `.docx` are accepted, with extension whitelist, **magic-byte sniffing**, and size limits enforced server-side;
- Scanned (image-only) PDFs are rejected with an explicit message;
- Each conflict card shows both documents' wording and page locations; choosing A/B marks the losing rule `superseded`; CUSTOM adds the user's text as the operative rule.

### 7.2 Local vision pipeline parameters (backend/.env)

| Variable | Default | Meaning |
| --- | --- | --- |
| `DNN_PIPELINE_ENABLED` | true | Local DNN pipeline master switch |
| `SENTINEL_INPUT_SIZE` | 448 | PatchCore sentinel inference size (multiple of 32) |
| `SENTINEL_CORESET_RATIO` | 0.01 | PatchCore coreset sampling ratio (memory control) |
| `YOLO_INPUT_SIZE` | 640 | YOLO-World inference size |
| `YOLO_CONF_THRESHOLD` | 0.01 | Zero-shot localization floor (low-confidence candidates filtered later) |
| `TILE_MAX_SIZE` | 1024 | Dynamic tile max edge |
| `JUDGE_HEURISTIC_CONTRAST` | 50 | Offline judge: contrast significance floor |
| `JUDGE_SENTINEL_MARGIN_SPARSE` | 0.24 | Sentinel margin threshold (sparse regions ≤3) |
| `JUDGE_SENTINEL_MARGIN_DENSE` | 0.40 | Sentinel margin threshold (dense regions >3) |

### 7.3 Detection / RAG parameters (backend/.env)

| Variable | Default | Meaning |
| --- | --- | --- |
| `WINDOW_SIZE` | 1024 | Fixed square tile edge |
| `DET_CONFIDENCE_THRESHOLD` | 0.35 | Minimum defect confidence |
| `NMS_IOU_THRESHOLD` | 0.30 | IoU threshold for cross-tile NMS |
| `RETRIEVAL_TOP_K` | 4 | Retrieved rule snippets per query |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | 0.30 | Hard cosine-similarity floor for RAG |
| `IMAGE_MAX_PIXELS` | 800000000 | Decompression-bomb pixel guard (0 = unlimited) |

### 7.4 Updating ONNX models

```bash
pip install torch torchvision ultralytics onnx onnxscript onnxslim
python backend/scripts/prepare_models.py
# produces backend/models/resnet18_features.onnx + yolov8s-worldv2.onnx
```

## 8. Core Algorithms

### 8.1 RAG pipeline

```
parse (paragraphs / captions / tables)
  → chunk (480 chars, 80-char overlap, captions merged into preceding text)
  → embed (online 1024-d / offline char+bigram signed hashing)
  → Chroma persistent index (HNSW, cosine)
retrieve: query embedding → HNSW ANN → similarity = 1 - distance
  → hard threshold filter → near-duplicate pruning → context assembly with structured rules
```

The similarity threshold is **strictly enforced**, so weakly related snippets never enter the model prompt.

### 8.2 Local vision pipeline (4 stages)

```
Raw image
  │
  ├─① PatchCore + ResNet18 sentinel (ONNX, CPU)
  │    Full-image anomaly heatmap → P99 threshold + absolute floor + region merge
  │    → locks N high-risk regions (up to 12, filters normal regions)
  │
  ├─② YOLO-World-S spatial localization (ONNX, CPU)
  │    Dynamic tile slicing on high-risk regions (tile edge = min(region_long_side × 3.0, 1024))
  │    Inline English defect labels, zero-shot output candidate boxes
  │    Discards degenerate near-full-frame boxes (both sides > 85% of tile)
  │
  ├─③ Judgment
  │    · With VLM → crop ROI → send with RAG rules to vision LLM → audit / classify
  │    · Without VLM → heuristic judge: contrast significance ≥ 50 OR sentinel margin ≥ tiered threshold
  │    → Confidence filter + same-type IoU-NMS de-duplication
  │
  └─④ OpenCV pixel落地
       Inside final defect ROI only: Otsu + morphological opening + largest contour
       Map back to original coordinates, produce translucent mask
```

Why this works:
- **Sentinel → Localization → Judgment** cascades dramatically reduce VLM calls (only high-risk regions go to judgment);
- YOLO-World uses **inline defect labels** (e.g. `"porosity, scratch, crack, crater, chipping, ..."`) — no class-specific retraining needed;
- The heuristic judge's **sentinel margin** (region sentinel score − full-image distance median) acts as a second evidence channel: clean long-strip images saturate P99 regions (12 flat-margin regions), while defect images have few regions with elevated margins;
- ONNX models **bundle with the project** (`backend/models/*.onnx`) — zero download, ready to run.

### 8.3 Fixed square tiling (full, exact coverage)

- If `max(W,H) ≤ window`: resize the whole image to `window×window` for a **single inspection**; boxes are scaled back by per-axis ratios;
- Otherwise: grid count `n = ceil(dim / window)`, stride `step = (dim - window)/(n - 1)`, with the last window forced to start at `dim - window`. Every tile is a full square, needs no padding, their union is exactly the image, and edges align exactly;
- Coordinate restoration: grid tiles add the tile offset; resize mode multiplies by `W/window` / `H/window`; results are clamped to image bounds.

### 8.4 Frontend SVG coordinate mapping

```
Original coordinates (px) ← viewBox="0 0 W H" → SVG overlay
         ↑                                        ↑
    Backend bbox returns                      CSS width:100%, height:100%
    mask_polygon points                       preserveAspectRatio="xMidYMid meet"
                                              wrapper: display:inline-block, max-width:100%
                                                └─ shrinks to <img> actual rendered size
                                                   (wrapper = img = overlay width/height identical)
```

Pre-fix root cause: wrapper forced to 100% panel width but `<img>` kept original aspect ratio; `preserveAspectRatio="none"` stretched the SVG viewBox into a deformed coordinate space, pushing boxes outside the image. **Post-fix the three align exactly, SVG adapts naturally.**

## 9. API Reference

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Liveness probe |
| GET | `/api/models` | Provider + local vision pipeline status |
| **PUT** | `/api/providers/{name}/key` | **Save API key (takes effect immediately, one-click frontend)** |
| **DELETE** | `/api/providers/{name}/key` | **Clear saved API key (.env untouched)** |
| **PUT** | `/api/providers/{name}/select` | **Set as default text/vision provider** |
| POST | `/api/documents/upload` | Upload a rule document (multipart) |
| GET | `/api/documents` | List documents (with max_documents) |
| GET | `/api/documents/rules` | Extracted structured rules |
| DELETE | `/api/documents/{id}` | Delete a document and all derived data |
| GET | `/api/conflicts` | List conflicts (pending/resolved) |
| POST | `/api/conflicts/{id}/resolve` | Adjudicate (A/B/CUSTOM) |
| POST | `/api/inspection/detect` | Upload an image and create a job |
| GET | `/api/inspection/jobs/{id}` | Poll progress / fetch result |
| GET | `/api/inspection/history` | History list (frontend does local search + pagination) |
| GET | `/api/inspection/image/{id}` | Original image |
| GET | `/api/inspection/results/{name}` | Annotated result image |

## 10. Project Structure

```
defect-inspection-agent/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                 # FastAPI entrypoint (lifespan/CORS/errors/routers)
│  │  ├─ config.py               # All settings (.env)
│  │  ├─ core/                   # Exceptions, upload security (magic/path traversal)
│  │  ├─ db/database.py          # SQLite: docs/chunks/rules/conflicts/jobs + app_meta(keys/refs)
│  │  ├─ schemas/                # Pydantic models
│  │  └─ services/
│  │     ├─ parser/              # PDF / DOCX parsers (extensible)
│  │     ├─ chunker.py           # Semantic chunking
│  │     ├─ embeddings.py        # Embedding abstraction + online/offline impls
│  │     ├─ vector_store.py      # Chroma facade (thresholded retrieval/reindex/hot-swap)
│  │     ├─ llm/                 # LLM adapters: base/online/mock/factory
│  │     ├─ key_store.py         # API key + provider preference SQLite storage / masking
│  │     ├─ conflict_service.py  # Conflict detection & adjudication
│  │     ├─ document_service.py  # Document ingestion pipeline
│  │     ├─ inspection_service.py# Job orchestration / history
│  │     └─ vision/              # Sentinel / locator / judge / sliding window / NMS / OpenCV / annotator
│  ├─ models/                    # Bundled ONNX models
│  │  ├─ resnet18_features.onnx   # PatchCore sentinel backbone
│  │  └─ yolov8s-worldv2.onnx     # YOLO-World-S zero-shot locator
│  ├─ scripts/                   # Self-tests / model export
│  ├─ requirements.txt
│  ├─ Dockerfile  .env.example
├─ frontend/
│  ├─ src/
│  │  ├─ api/client.ts           # Fetch wrapper with normalized errors
│  │  ├─ types.ts                # TS types mirroring backend schemas
│  │  └─ components/             # Five feature panels + result overlay
│  │     ├─ ModelStatusPanel.tsx # One-click API key config + default provider selection
│  │     ├─ HistoryPanel.tsx     # History with name search + pagination
│  │     └─ ResultView.tsx       # SVG overlay (auto-adapts, always on true defect position)
│  ├─ package.json  vite.config.ts  Dockerfile  nginx.conf
└─ docker-compose.yml
```

## 11. Learning Highlights

1. **Local DNN pipeline architecture**: sentinel → localization → judgment cascade, PatchCore does "should we look here", YOLO-World does "where", VLM/heuristic does "is it a defect" — drastically cuts VLM calls while keeping zero-shot generalization;
2. **End-to-end RAG engineering**: parsing → chunking → embeddings → thresholded retrieval → provenance; metadata is fixed at write time, making every result explainable;
3. **Interface-based model abstraction**: swapping providers is configuration-only; tolerant JSON parsing and exponential-backoff retries are essential LLM guardrails;
4. **API key hot reload**: frontend SQLite save + `lru_cache` clear + embedding provider switch + background RAG reindex = "save and it works", no restart;
5. **SVG coordinate mapping**: `wrapper = img = overlay` at identical dimensions + `preserveAspectRatio="xMidYMid meet"` guarantees correct annotation placement at any aspect ratio;
6. **Large images & concurrency**: lazy per-window cropping, pixel guard, `asyncio.to_thread` for CPU work, background jobs with polling;
7. **Security baseline**: magic-byte checks, filename sanitization, `safe_join`, default env-only keys (frontend optional save goes to DB), unified errors that never leak stack traces;
8. **Graceful degradation**: local fallbacks for every external dependency (local ONNX pipeline, hashing embeddings, heuristic judge, mock LLM, OpenCV contours) keep the full flow demonstrable offline;
9. **Cross-platform delivery**: pure-Python parsers (no poppler/LibreOffice needed), multi-arch containers, decoupled frontend/backend, ONNX CPU inference — identical experience on Windows/Linux/macOS.

## 12. FAQ

**Can I use it without API keys?**
Yes. Offline mode uses the local PatchCore + YOLO-World + heuristic judge pipeline — ideal for validating the flow and batch-running images. Adding a key substantially improves judgment and generalization.

**Why does inspection say "N unresolved conflicts remain"?**
It is a deliberate quality gate. Adjudicate every item in the Conflict Center so the operative standard is contradiction-free.

**Inspection of huge images is slow?**
Tile count is roughly `ceil(W/1024) × ceil(H/1024)`, and online mode makes one VLM call per tile. Increase `WINDOW_SIZE`, or scale the stateless detector horizontally. The local sentinel+locator scan is CPU ONNX, typically 5–10 s for a 4000×15000 strip image.

**What if I switch embedding providers / clear a frontend key?**
On restart the system detects the incompatible vector space, clears the collection, and rebuilds it from the chunk source text stored in SQLite — no re-upload needed. Background reindex is also triggered when the provider changes at runtime.

**What about scanned PDFs?**
Run OCR first (OCR is intentionally not bundled to preserve zero system-level dependencies), then upload the text-layered PDF/DOCX.

**Why are annotations flying outside the image?**
This was a CSS bug in earlier versions (wrapper forced to 100% panel width + SVG `preserveAspectRatio="none"`). The current version is fixed: wrapper shrinks to the img's actual size + SVG adapts naturally, coordinates map correctly. Just refresh the page.

**Where are the ONNX models? Can I update them?**
Under `backend/models/`, bundled with the project. To update (new YOLO-World version, new PatchCore backbone) run `pip install torch torchvision ultralytics onnx onnxscript onnxslim && python backend/scripts/prepare_models.py`.
