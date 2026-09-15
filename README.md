# Multimodal Defect Inspection Agent

> An industrial surface-defect inspection full-stack system built on a **RAG rule vector knowledge base**, **multimodal LLMs**, and **SAM/OpenCV segmentation**.
> FastAPI (Python 3.12) backend · React + TypeScript (Vite) frontend · SQLite + ChromaDB · Docker for cross-platform deployment.

中文文档见 [README.zh-CN.md](./README.zh-CN.md)。

---

## Table of Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Quick Start (Local)](#4-quick-start-local)
5. [Docker Deployment](#5-docker-deployment)
6. [User Guide](#6-user-guide)
7. [Core Algorithms](#7-core-algorithms)
8. [API Reference](#8-api-reference)
9. [Project Structure](#9-project-structure)
10. [Learning Highlights](#10-learning-highlights)
11. [FAQ](#11-faq)

---

## 1. Overview

Conventional machine-vision inspection requires specialized models retrained for every defect type and production line, making rule updates expensive. This system turns a company's **illustrated inspection-standard documents (PDF/DOCX) into a searchable, reasoning-capable knowledge base**:

- Engineers upload rule documents → the system parses, chunks, embeds, and extracts structured rules;
- When documents contradict each other, every conflict is presented one-by-one for a human decision, guaranteeing a single operative standard;
- An industrial image of **any size** is scanned with fixed square sliding windows → a multimodal model judges defects against the rules → SAM/OpenCV produces precise segmentation → defects are drawn on the original image with rule-based evidence;
- Every inspection is stored and retrievable from history.

**Key property:** the system runs fully **offline without any API key** (heuristic rule engine + OpenCV fallback) and automatically upgrades to LLM/VLM capabilities once a key is configured.

## 2. Features

| Module | Capabilities |
| --- | --- |
| Documents | PDF/DOCX upload (max 10 docs, 50 MB each); paragraph text + figure-caption recognition; table extraction; semantic chunking with overlap |
| RAG store | Zhipu / Qwen embeddings (1024-d) with a deterministic local hashing fallback; persistent ChromaDB (HNSW / cosine); enforced similarity threshold; page-level provenance |
| Rule understanding | Structured rule extraction with Zhipu GLM / Qwen / DeepSeek; tolerant JSON parsing; automatic index rebuild after embedding-provider switch |
| Conflicts | LLM-based detection of cross-document contradictions (e.g., clashing thresholds); one-by-one adjudication (A / B / custom); inspection is blocked until all conflicts are resolved |
| Sliding windows | Fixed 1024 px square tiles; small images resized and inspected once; large images tiled with a uniform stride that aligns to all edges (full coverage, no overflow); boxes mapped back exactly to original coordinates |
| Detection | Per-tile multimodal judgment (GLM-4V / Qwen VL) grounded in RAG rules; confidence filtering; cross-tile NMS; automatic fallback to OpenCV (adaptive threshold + morphology) when no VLM is available |
| Segmentation | SAM pixel masks from box prompts (optional); automatic OpenCV contour fallback |
| Output | Pass/fail verdict, bounding boxes, translucent masks, confidence, coordinates, rule evidence (similarity / source document / page), narrative summary |
| History | All jobs persisted in SQLite; thumbnail list with full result drill-down |
| Engineering | Unified exception handling, magic-byte and path-traversal protection, decompression-bomb guard, lazy per-tile memory usage, live progress reporting, CORS, health checks |

## 3. Architecture

```
Browser (React + TypeScript + Vite)
  Documents │ Conflicts │ Inspection (progress polling) │ SVG overlay │ History │ Model status
                                  │ HTTP / multipart
                                  ▼
FastAPI (api layer: documents / conflicts / inspection / system)
Services:
  parser(PDF/DOCX) → chunker ─┐
  conflict service (LLM/heuristic)
  inspection orchestrator (progress / conflict gate / evidence / history)
        └─ vision: sliding window → VLM/CV → NMS → SAM/OpenCV → annotator
Model layer (facade pattern):
  BaseLLM:  Zhipu GLM │ Qwen │ DeepSeek │ Mock (offline)
  EmbeddingClient: Zhipu │ Qwen │ local hashing
Storage:
  ChromaDB (HNSW + cosine, persistent)
  SQLite: documents / chunk source text / rules / conflicts / inspection jobs
  data/: uploads, original & annotated images, vectors, SAM checkpoints
```

Design principles:

- **Facade-based access**: routers call service singletons only; models and the vector store are obtained through factories — easy to swap and test;
- **Interface-first**: `BaseLLM`, `EmbeddingClient`, and `BaseParser` contracts mean adding a provider or format is a new adapter, not a rewrite;
- **Online/offline dual track**: every external AI capability has a deterministic local fallback.

## 4. Quick Start (Local)

### 4.1 Prerequisites

- Python ≥ 3.10 (validated on **Python 3.12**, Windows 11)
- Node.js ≥ 18 (validated on Node 24)
- No external database or Docker required for local mode

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

# 3) Optional: configure LLM API keys
cp .env.example .env   # Windows: copy .env.example .env
#   Set any of ZHIPU_API_KEY / QWEN_API_KEY / DEEPSEEK_API_KEY

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
3. Upload one product image on the **Inspection** tab and watch progress
   (window planning → per-tile detection → merging → segmentation);
4. Review boxes, confidence, and rule evidence;
5. Browse past runs in **History**; check active providers in **Model Status**.

## 5. Docker Deployment

One-command startup (Nginx on :80, API on :8000):

```bash
cd defect-inspection-agent
# optional: cp backend/.env.example backend/.env and fill in keys
docker compose up -d --build
# open http://localhost
```

- All backend data (documents, SQLite, ChromaDB) persists in the `backend-data` volume;
- Images are based on `python:3.12-slim` and `node:20-alpine / nginx:1.27-alpine` (x86_64 / arm64);
- To enable SAM, extend the backend image with `torch` + `segment-anything` and mount a checkpoint.

## 6. User Guide

### 6.1 Documents & conflicts

- Only `.pdf` / `.docx` are accepted, with extension whitelist, **magic-byte sniffing**, and size limits enforced server-side;
- Scanned (image-only) PDFs are rejected with an explicit message;
- Each conflict card shows both documents' wording and page locations; choosing A/B marks the losing rule `superseded`; CUSTOM adds the user's text as the operative rule.

### 6.2 Detection settings (backend/.env)

| Variable | Default | Meaning |
| --- | --- | --- |
| `WINDOW_SIZE` | 1024 | Fixed square tile edge length |
| `DET_CONFIDENCE_THRESHOLD` | 0.35 | Minimum defect confidence |
| `NMS_IOU_THRESHOLD` | 0.30 | IoU threshold for cross-tile NMS |
| `RETRIEVAL_TOP_K` | 4 | Retrieved rule snippets per query |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | 0.30 | Hard cosine-similarity floor for RAG |
| `IMAGE_MAX_PIXELS` | 800000000 | Decompression-bomb pixel guard (0 = unlimited) |

### 6.3 Enabling SAM (optional)

```bash
pip install torch segment-anything
# place sam_vit_b_01ec64.pth under backend/data/sam_checkpoints/
# restart; the Model Status page will show "SAM available"
```

## 7. Core Algorithms

### 7.1 RAG pipeline

```
parse (paragraphs / captions / tables)
  → chunk (480 chars, 80-char overlap, captions merged into preceding text)
  → embed (online 1024-d / offline char+bigram signed hashing)
  → Chroma persistent index (HNSW, cosine)
retrieve: query embedding → HNSW ANN → similarity = 1 - distance
  → hard threshold filter → near-duplicate pruning → context assembly with structured rules
```

The similarity threshold is **strictly enforced**, so weakly related snippets never enter the model prompt.

### 7.2 Fixed square tiling (full, exact coverage)

- If `max(W,H) ≤ window`: resize the whole image to `window×window` for a **single inspection**; boxes are scaled back by per-axis ratios;
- Otherwise: grid counts `n = ceil(dim / window)`, stride `step = (dim - window)/(n - 1)`, with the last window forced to start at `dim - window`. Every tile is a full square, needs no padding, their union is exactly the image, and edges align exactly;
- Coordinate restoration: grid tiles add the tile offset; resize mode multiplies by `W/window` / `H/window`; results are clamped to image bounds.

### 7.3 Detection, merging, segmentation

1. Each tile is JPEG-encoded and sent to the VLM with RAG rules; the model returns JSON `[type, confidence, bbox, description]` (tolerantly parsed);
2. Confidence thresholding → same-type IoU NMS removes duplicate boxes from overlapping tiles;
3. Boxes become SAM box prompts for pixel masks; offline mode uses CLAHE + Otsu + largest contour;
4. Offline CV fallback: median blur → adaptive Gaussian threshold (dark spots) → morphological opening → area/circularity/local-contrast filtering → compact=porosity, elongated=scratch.

### 7.4 Memory strategy for huge images

At any moment only the **original image plus one 1024² tile** reside in working memory: tiles are cropped lazily and released per window. The saved annotated preview is downscaled only when its longest side exceeds 6000 px (the frontend overlays coordinates independently from the original). `PIL.Image.MAX_IMAGE_PIXELS` blocks decompression bombs before decoding.

## 8. API Reference

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Liveness probe |
| GET | `/api/models` | Provider & SAM status |
| POST | `/api/documents/upload` | Upload a rule document (multipart) |
| GET | `/api/documents` | List documents (with max_documents) |
| GET | `/api/documents/rules` | Extracted structured rules |
| DELETE | `/api/documents/{id}` | Delete a document and all derived data |
| GET | `/api/conflicts` | List conflicts (pending/resolved) |
| POST | `/api/conflicts/{id}/resolve` | Adjudicate (A/B/CUSTOM) |
| POST | `/api/inspection/detect` | Upload an image and create a job |
| GET | `/api/inspection/jobs/{id}` | Poll progress / fetch result |
| GET | `/api/inspection/history` | History list |
| GET | `/api/inspection/image/{id}` | Original image |
| GET | `/api/inspection/results/{name}` | Annotated result image |

## 9. Project Structure

```
defect-inspection-agent/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                 # FastAPI entrypoint (lifespan/CORS/errors/routers)
│  │  ├─ config.py               # All settings (.env)
│  │  ├─ core/                   # Exceptions, upload security (magic/path traversal)
│  │  ├─ db/database.py          # SQLite: docs/chunks/rules/conflicts/jobs
│  │  ├─ schemas/                # Pydantic models
│  │  └─ services/
│  │     ├─ parser/              # PDF / DOCX parsers (extensible)
│  │     ├─ chunker.py           # Semantic chunking
│  │     ├─ embeddings.py        # Embedding abstraction + online/offline impls
│  │     ├─ vector_store.py      # Chroma facade (thresholded retrieval/reindex)
│  │     ├─ llm/                 # LLM adapters: base/online/mock/factory
│  │     ├─ conflict_service.py  # Conflict detection & adjudication
│  │     ├─ document_service.py  # Document ingestion pipeline
│  │     ├─ inspection_service.py# Job orchestration / history
│  │     └─ vision/              # sliding window / NMS / CV / VLM / SAM / annotator
│  ├─ requirements.txt
│  ├─ Dockerfile  .env.example
├─ frontend/
│  ├─ src/
│  │  ├─ api/client.ts           # Fetch wrapper with normalized errors
│  │  ├─ types.ts                # TS types mirroring backend schemas
│  │  └─ components/             # Five feature panels + result overlay
│  ├─ package.json  vite.config.ts  Dockerfile  nginx.conf
└─ docker-compose.yml
```

## 10. Learning Highlights

1. **End-to-end RAG engineering**: parsing → chunking → embeddings → thresholded retrieval → provenance; metadata (page/doc/order) is fixed at write time, making every result explainable;
2. **Interface-based model abstraction**: swapping providers is configuration-only; tolerant JSON parsing and exponential-backoff retries are essential LLM guardrails;
3. **One coordinate system**: all boxes are normalized to original-image coordinates, shared by NMS, segmentation, and the SVG frontend overlay;
4. **Large images & concurrency**: lazy per-window cropping, pixel guard, `asyncio.to_thread` for CPU work, background jobs with polling;
5. **Security baseline**: magic-byte checks, filename sanitization, `safe_join`, env-only keys, error responses that never leak stack traces;
6. **Graceful degradation**: local fallbacks for every external dependency (mock LLM, hashing embeddings, OpenCV detection/segmentation) keep the full flow demonstrable offline;
7. **Cross-platform delivery**: pure-Python parsers (no poppler/LibreOffice needed), multi-arch containers, decoupled frontend and backend — identical experience on Windows/Linux/macOS.

## 11. FAQ

**Can I use it without API keys?**
Yes. Offline mode uses heuristic rule extraction (best on documents with explicit numeric thresholds) and OpenCV-based vision — ideal for validating the flow; adding a key substantially improves judgment and generalization.

**Why does inspection say "N unresolved conflicts remain"?**
It is a deliberate quality gate. Adjudicate every item in the Conflict Center so the operative standard is contradiction-free.

**Inspection of huge images is slow?**
Tile count is roughly `ceil(W/1024) × ceil(H/1024)`, and online mode makes one VLM call per tile. Increase `WINDOW_SIZE`, or scale the stateless detector horizontally.

**What if I switch embedding providers?**
On restart the system detects the incompatible vector space, clears the collection, and rebuilds it from the chunk source text stored in SQLite — no re-upload needed.

**What about scanned PDFs?**
Run OCR first (OCR is intentionally not bundled to preserve zero system-level dependencies), then upload the text-layered PDF/DOCX.
