# 多模态缺陷检测智能体（Multimodal Defect Inspection Agent）

> 基于 **RAG 规则向量知识库 + 多模态大模型 + SAM/OpenCV 分割** 的工业表面缺陷检测全栈系统。
> FastAPI（Python 3.12）后端 · React + TypeScript（Vite）前端 · SQLite + ChromaDB · Docker 跨平台部署。

---

## 目录

1. [系统概览](#1-系统概览)
2. [核心功能](#2-核心功能)
3. [系统架构](#3-系统架构)
4. [快速开始（本地开发）](#4-快速开始本地开发)
5. [Docker 容器化部署](#5-docker-容器化部署)
6. [使用指南](#6-使用指南)
7. [核心算法思路](#7-核心算法思路)
8. [API 一览](#8-api-一览)
9. [目录结构](#9-目录结构)
10. [学习要点](#10-学习要点)
11. [常见问题（FAQ）](#11-常见问题faq)

---

## 1. 系统概览

传统机器视觉检测依赖针对单一缺陷、单一产线训练的专用模型，规则更新成本高。
本系统把**企业的图文检测标准文档（PDF/DOCX）变成可检索、可推理的知识库**：

- 质检员上传规则文档 → 系统自动解析、向量化、抽取结构化规则；
- 多份文档规则冲突时，逐条呈现给用户人工裁决，保证执行标准唯一；
- 上传任意尺寸的工业品图片 → 固定方形滑窗扫描 → 多模态模型按规则判定缺陷
  → SAM/OpenCV 精确分割 → 在原图标注缺陷坐标并给出规则依据；
- 全部检测记录可历史回溯。

**关键特性：未配置任何大模型 API Key 时，系统仍可离线完整运行**
（规则启发式引擎 + OpenCV 经典视觉兜底），配 Key 后自动切换大模型能力。

## 2. 核心功能

| 模块 | 能力 |
| --- | --- |
| 文档处理 | PDF/DOCX 上传（最多 10 份、单文件 50MB）；文字段落 + 图片说明识别；表格抽取；按段落语义分块（带 80 字重叠窗口） |
| RAG 知识库 | 智谱 / 千问 Embedding（1024 维）+ 离线哈希向量兜底；Chroma 持久化向量库（HNSW / cosine）；相似度阈值显式过滤；片段带来源页码可溯源 |
| 规则理解 | 智谱 GLM / 千问 / DeepSeek 抽取结构化检测规则；JSON 容错解析；切换 Embedding 模型自动重建索引 |
| 冲突管理 | LLM 自动识别跨文档规则矛盾（阈值冲突等）；A / B / CUSTOM 逐条人工裁决；裁决结果直接生效为执行标准；未解决冲突时拒绝检测 |
| 滑窗检测 | 固定 1024 方形窗；小图整图 resize 单检；大图均匀步长网格扫描（边缘对齐、不重不漏）；检测框坐标精确复原到原图 |
| 缺陷判定 | 多模态视觉模型（GLM-4V / 千问 VL）按 RAG 规则逐窗判定；置信度过滤；跨窗 NMS 去重；模型不可用时自动降级 OpenCV（自适应阈值 + 形态分类） |
| 精确分割 | SAM（box prompt 像素级掩膜，可选）；未安装时自动降级 OpenCV 轮廓分割 |
| 结果展示 | 是否合格结论、缺陷框 + 半透明掩膜叠加、置信度、缺陷坐标、规则依据片段（含相似度/来源文档/页码）、文字结论 |
| 历史记录 | SQLite 持久化全部任务，缩略图列表 + 完整结果回看 |
| 工程质量 | 统一异常体系与友好错误；文件魔数校验 / 路径穿越防护；大图像素安全阀与逐窗内存控制；处理进度实时提示；CORS、健康检查、响应耗时头 |

## 3. 系统架构

```
┌──────────────────────────── 浏览器 (React + TypeScript + Vite) ───────────────────────────┐
│ 规则文档管理 │ 冲突中心(逐条裁决) │ 图片检测(进度轮询) │ 结果SVG叠加可视化 │ 检测历史 │ 模型状态 │
└───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                        │ HTTP / multipart（开发经 Vite 代理；容器经 Nginx 反代）
┌───────────────────────────────────────▼──────────────────────── FastAPI ──────────────────┐
│ api 层     documents / conflicts / inspection / system（统一异常处理 / CORS / 参数校验）    │
│                                                                                             │
│ services 层                                                                                 │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────────┐  │
│  │ 文档解析       │  │ 分块器       │  │ 冲突检测/裁决  │  │ 检测编排(进度/门禁/依据/历史)     │  │
│  │ PDF(pypdf)   │  │ 段落+重叠    │  │ LLM/启发式    │  │ 滑窗 → VLM/CV → NMS → SAM/OpenCV│  │
│  │ DOCX(docx)   │  │ 图注合并     │  │              │  │                                │  │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └──────────────┬─────────────────┘  │
│         │                 │                 │                          │                    │
│  ┌──────▼─────────────────▼──────┐   ┌───────▼───────────┐   ┌────────▼─────────┐          │
│  │ LLM 标准化适配（BaseLLM）        │   │ RAG 检索           │   │ 视觉模型适配       │          │
│  │ 智谱GLM / 千问 / DeepSeek / Mock│   │ Chroma(余弦+阈值)  │   │ GLM-4V / 千问VL   │          │
│  └───────────────────────────────┘   └───────▲───────────┘   └──────────────────┘          │
│         │                                    │                                               │
│  ┌──────▼──────────┐               ┌─────────┴──────────┐                                    │
│  │ Embedding 抽象   │──────────────▶│ 向量库(ChromaDB)    │                                    │
│  │ 智谱/千问/本地哈希 │               │ HNSW + cosine       │                                    │
│  └─────────────────┘               └────────────────────┘                                    │
└───────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                            │
                    SQLite(app.db)：文档 / chunk原文 / 规则 / 冲突 / 检测任务
                    data/：上传文档、原图、标注图、向量库、SAM 权重
```

设计要点：

- **门面 / 依赖注入**：API 层只调用 service 单例，service 通过工厂拿到模型与向量库实例，模型可替换、可测试；
- **面向接口**：`BaseLLM`、`EmbeddingClient`、`BaseParser` 定义统一协议，新增模型/格式只加适配器；
- **在线/离线双轨**：所有外部 AI 能力都有确定性兜底，系统无外部服务也可演示与验证。

## 4. 快速开始（本地开发）

### 4.1 环境要求

- Python ≥ 3.10（开发验证：**Python 3.12**，Windows 11）
- Node.js ≥ 18（开发验证：Node 24）
- 无需外部数据库 / Docker（本地模式）

### 4.2 后端

Windows / Linux / macOS 通用（以下以 Windows PowerShell 为例）：

```powershell
cd defect-inspection-agent/backend

# 1) 创建并激活虚拟环境（任选其一）
conda create -y -p ./.conda-env python=3.12
conda activate ./.conda-env
#   或：python -m venv .venv ; .venv\Scripts\Activate.ps1

# 2) 安装依赖（如默认源慢可追加 -i https://pypi.tuna.tsinghua.edu.cn/simple）
pip install -r requirements.txt

# 3) （可选）配置大模型 API Key
copy .env.example .env   # Linux/macOS: cp .env.example .env
#   编辑 .env 填入 ZHIPU_API_KEY / QWEN_API_KEY / DEEPSEEK_API_KEY 任意一家

# 4) 启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 接口文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/health>

### 4.3 前端

```powershell
cd defect-inspection-agent/frontend
npm install
npm run dev
```

打开 <http://localhost:5173>。开发服务器会把 `/api` 代理到 `localhost:8000`。

### 4.4 快速验证流程

1. **规则文档**页上传 1~2 份 PDF/DOCX 检测标准（可先用企业自带标准文档试用）；
2. 若出现冲突 → **冲突中心**逐条裁决（全部解决前无法检测）；
3. **缺陷检测**页上传一张工业品图片，观察实时进度（规划窗数 → 逐窗检测 → 合并 → 分割）；
4. 查看框选结果、置信度与规则依据；
5. **检测历史**页回看；**模型状态**页确认当前是在线模型还是离线兜底。

## 5. Docker 容器化部署

一键启动（前端 Nginx :80，后端 :8000）：

```bash
cd defect-inspection-agent
# 可选：cp backend/.env.example backend/.env 并填入 Key
docker compose up -d --build
# 打开 http://localhost
```

- 后端数据（文档、SQLite、Chroma 向量库）持久化在命名卷 `backend-data`；
- 镜像基于 `python:3.12-slim` 与 `node:20-alpine / nginx:1.27-alpine`，支持 x86_64 / arm64；
- 想启用 SAM：以该 Dockerfile 为基础追加安装 `torch`、`segment-anything` 并挂载权重目录。

## 6. 使用指南

### 6.1 文档与冲突

- 仅接受 `.pdf` / `.docx`，服务端同时做扩展名白名单、**文件头魔数校验**与大小校验；
- 扫描件 PDF（无文字层）无法解析，系统会明确提示；
- 冲突卡片左/右分别展示两份文档的原文与页码，可「采纳 A」「采纳 B」或「自定义统一表述」；
  被淘汰规则状态变为 `superseded`，不再进入检测依据；自定义规则作为最终标准生效。

### 6.2 检测参数（backend/.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `WINDOW_SIZE` | 1024 | 固定方形滑窗边长 |
| `DET_CONFIDENCE_THRESHOLD` | 0.35 | 缺陷置信度过滤阈值 |
| `NMS_IOU_THRESHOLD` | 0.30 | 跨窗重复框 NMS 阈值 |
| `RETRIEVAL_TOP_K` | 4 | 每次规则检索返回片段数 |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | 0.30 | RAG 相似度下限（严格生效） |
| `IMAGE_MAX_PIXELS` | 800000000 | 大图像素安全阀，0 为不限制 |

### 6.3 启用 SAM 精确分割（可选）

```bash
pip install torch segment-anything
# 下载权重 sam_vit_b_01ec64.pth 放到 backend/data/sam_checkpoints/
# 重启后端，模型状态页将显示 “SAM 可用”
```

## 7. 核心算法思路

### 7.1 RAG 流水线

```
解析(段落/图注/表格) → 分块(480字/块, 80字重叠, 图注并入前文)
   → Embedding(在线 1024 维 / 离线字+bigram 特征哈希)
   → Chroma(HNSW, cosine) 持久化
检索: query embedding → HNSW 召回 → similarity=1-distance 显式阈值过滤
   → 同文档同位置去重 → 连同结构化规则组装成判定上下文
```

阈值过滤是**强制生效**的：低相似度片段不会进入模型上下文，避免“看似检索到了、其实不相关”的语义漂移。

### 7.2 固定方形滑窗（不重不漏）

- `max(W,H) ≤ window`：整图 resize 到 window×window **一次检测**，框按宽/高各自缩放比还原；
- 否则网格数 `n = ceil(dim / window)`，步长 `step = (dim - window)/(n - 1)`，
  末窗起点强制为 `dim - window`：所有窗完整、无 padding、并集恰为整图、边缘严格对齐；
- 框复原：网格模式 `(x+tx, y+ty)`；resize 模式 `(x·W/window, y·H/window)`，最后统一钳制边界。

### 7.3 检测、合并与分割

1. 逐窗把切片 JPEG + RAG 规则送多模态模型，返回 `[类型, 置信度, 框, 描述]` JSON（容错解析）；
2. 置信度阈值过滤 → 同类型 IoU-NMS 消除重叠窗重复框；
3. 分割：检测框作为 SAM 的 box prompt 输出像素掩膜；离线则 CLAHE + Otsu + 最大轮廓；
4. 离线 CV 兜底链路：中值滤波 → 自适应高斯阈值（暗斑）→ 形态学开运算
   → 轮廓面积/圆度/局部对比度过滤 → 近圆=气孔、细长=划痕。

### 7.4 大图内存策略

任意时刻内存中只有“原图 + 一个 1024² 切片”：切片按需裁剪、逐窗释放；
标注图保存时最长边超过 6000px 才等比缩小（前端坐标叠加独立于标注图）。
解码前设置 `PIL.Image.MAX_IMAGE_PIXELS` 安全阀，防止解压炸弹导致 OOM。

## 8. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/models` | 模型 / SAM 配置状态 |
| POST | `/api/documents/upload` | 上传规则文档（multipart） |
| GET | `/api/documents` | 文档列表（含 max_documents） |
| GET | `/api/documents/rules` | 结构化规则 |
| DELETE | `/api/documents/{id}` | 删除文档（联动清理） |
| GET | `/api/conflicts` | 冲突列表（pending/resolved） |
| POST | `/api/conflicts/{id}/resolve` | 裁决冲突（A/B/CUSTOM） |
| POST | `/api/inspection/detect` | 上传图片创建检测任务 |
| GET | `/api/inspection/jobs/{id}` | 轮询进度/获取结果 |
| GET | `/api/inspection/history` | 历史列表 |
| GET | `/api/inspection/image/{id}` | 原图 |
| GET | `/api/inspection/results/{name}` | 标注结果图 |

## 9. 目录结构

```
defect-inspection-agent/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                # FastAPI 入口（lifespan/CORS/异常/路由）
│  │  ├─ config.py              # 全部配置（.env 注入）
│  │  ├─ core/                  # 异常体系、文件安全（魔数/路径穿越）
│  │  ├─ db/database.py         # SQLite：文档/chunk/规则/冲突/任务
│  │  ├─ schemas/               # Pydantic 请求响应模型
│  │  └─ services/
│  │     ├─ parser/             # PDF / DOCX 解析器（可扩展）
│  │     ├─ chunker.py          # 语义分块
│  │     ├─ embeddings.py       # Embedding 抽象 + 在线/离线实现
│  │     ├─ vector_store.py     # Chroma 门面（阈值检索/重建）
│  │     ├─ llm/                # LLM 适配：基类/在线/Mock/工厂
│  │     ├─ conflict_service.py # 冲突检测与逐条裁决
│  │     ├─ document_service.py # 文档入库流水线
│  │     ├─ inspection_service.py # 检测任务编排/历史
│  │     └─ vision/             # 滑窗/NMS/CV/VLM/SAM/标注/门面
│  ├─ requirements.txt
│  ├─ Dockerfile  .env.example
├─ frontend/
│  ├─ src/
│  │  ├─ api/client.ts          # API 封装与统一错误
│  │  ├─ types.ts               # 与后端一致的 TS 类型
│  │  └─ components/            # 5 个功能面板 + 结果可视化
│  ├─ package.json  vite.config.ts  Dockerfile  nginx.conf
└─ docker-compose.yml
```

## 10. 学习要点

1. **RAG 工程闭环**：解析 → 分块 → 向量化 → 阈值检索 → 溯源，元数据在写入点固化（页码/文档名/序号），检索结果天然可解释；
2. **面向接口的模型抽象**：`BaseLLM` / `EmbeddingClient` 让“换模型”只改配置；JSON 容错解析、退避重试是 LLM 工程的必备护栏；
3. **滑窗检测坐标系统一**：所有框最终归一到原图坐标，NMS、分割、前端叠加都复用同一坐标系；
4. **大图与并发**：逐窗惰性裁剪、像素安全阀、CPU 任务 `asyncio.to_thread`、后台任务 + 轮询进度；
5. **安全基线**：魔数防伪、文件名消毒、`safe_join` 防穿越、Key 仅走环境变量、统一错误不泄露堆栈；
6. **可降级架构**：每条外部能力都有本地兜底（Mock LLM、哈希 Embedding、OpenCV 检测/分割），系统永远可跑通，便于离线演示与测试；
7. **跨平台交付**：纯 Python 解析库（无 poppler/libreoffice 系统依赖）+ 多架构容器 + 前后端分离，Windows/Linux/macOS 一致体验。

## 11. 常见问题（FAQ）

**Q：不配 API Key 能用吗？**
能。自动进入离线模式：规则抽取/冲突检测用内置启发式（对“含明确阈值数字”的规则文档效果最佳），图像检测用 OpenCV，适合验证流程；接入大模型后判定与泛化能力显著增强。

**Q：检测时报“仍有 N 条冲突未确认”？**
这是系统的质量门禁：到「冲突中心」逐条裁决后再检测，确保执行标准无矛盾。

**Q：大图检测很慢？**
窗数约为 `ceil(W/1024)×ceil(H/1024)`，在线模式下每窗一次视觉模型调用。可先在 `.env` 调大 `WINDOW_SIZE`，或部署多副本并发改造（检测器无状态）。

**Q：更换了 Embedding 提供方？**
重启时系统检测到向量空间不一致会自动清空集合，并依据 SQLite 中的 chunk 原文全量重建索引，无需重新上传文档。

**Q：扫描版 PDF 怎么办？**
需要先 OCR（本系统不内置 OCR 以保持跨平台零系统依赖），生成带文字层的 PDF/DOCX 后再上传。
