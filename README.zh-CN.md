# 多模态缺陷检测智能体（Multimodal Defect Inspection Agent）

> 基于 **RAG 规则向量知识库 + 本地深度视觉管线 + 可选云端 VLM 裁决** 的工业表面缺陷检测全栈系统。
> FastAPI（Python 3.12）后端 · React + TypeScript（Vite）前端 · SQLite + ChromaDB · Docker 跨平台部署。
> PatchCore + YOLO-World-S 模型文件已随项目内置（backend/models/*.onnx），CPU 推理，**无需下载权重**。

English documentation: [README.md](./README.md)

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
- 上传任意尺寸的工业品图片 → **PatchCore 全局哨兵** 锁定高危区域 →
  **YOLO-World-S 空间定位** 零样本输出缺陷框 → **云端 VLM 或离线启发式裁判** 按 RAG 规则裁决 →
  **OpenCV 像素轮廓** 精准落地 → 在原图标注缺陷坐标并给出规则依据；
- 全部检测记录可历史回溯，支持按图片名搜索与分页；
- **模型 API Key 可在前端一键配置并即时生效**，无需修改 `.env`、无需重启。

**关键特性：未配置任何大模型 API Key 时，系统仍可离线完整运行**
（规则启发式引擎 + PatchCore/YOLO-World + 启发式裁判），配 Key 后自动升级为 VLM 裁决能力。

## 2. 核心功能

| 模块 | 能力 |
| --- | --- |
| 文档处理 | PDF/DOCX 上传（最多 10 份、单文件 50MB）；文字段落 + 图片说明识别；表格抽取；按段落语义分块（带 80 字重叠窗口） |
| RAG 知识库 | 智谱 / 千问 Embedding（1024 维）+ 离线哈希向量兜底；Chroma 持久化向量库（HNSW / cosine）；相似度阈值显式过滤；片段带来源页码可溯源 |
| 规则理解 | 智谱 GLM / 千问 / DeepSeek 抽取结构化检测规则；JSON 容错解析；切换 Embedding 模型自动重建索引 |
| 冲突管理 | LLM 自动识别跨文档规则矛盾（阈值冲突等）；A / B / CUSTOM 逐条人工裁决；裁决结果直接生效为执行标准；未解决冲突时拒绝检测 |
| 本地视觉管线（CPU） | **PatchCore + ResNet18 哨兵** 全图异常热力图，过滤正常区域；**YOLO-World-S** 对高危区域动态切片，内联缺陷标签零样本定位；ONNX 模型随项目内置，无需下载 |
| 缺陷裁决 | 云端视觉大模型（GLM-4V / 千问 VL）按 RAG 规则审核候选框，过滤误报并纠正类型；未配置 Key 时自动降级为**启发式裁判**（对比度显著性 + 哨兵幅度双证据） |
| 像素落地 | OpenCV 大津法 + 形态学轮廓提取，仅在最终缺陷框 ROI 内工作，还原至原图坐标 |
| API Key 配置 | **前端一键保存**（SQLite 本地存储），保存即生效；脱敏展示（如 `sk-1****abcd`）；设为默认文本/视觉模型即时切换；Key 缺失时自动回退 |
| 结果展示 | 是否合格结论、缺陷框 + 半透明掩膜叠加、置信度、缺陷坐标、规则依据片段（含相似度/来源文档/页码）、文字结论；**SVG 叠加层随图片自适应缩放**，标注始终贴合真实位置 |
| 历史记录 | SQLite 持久化全部任务，缩略图列表 + 完整结果回看；**按图片名/摘要搜索 + 每页 20 条分页** |
| 工程质量 | 统一异常体系与友好错误；文件魔数校验 / 路径穿越防护；大图像素安全阀与逐窗内存控制；处理进度实时提示；CORS、健康检查、响应耗时头 |

## 3. 系统架构

```
┌──────────────────────────── 浏览器 (React + TypeScript + Vite) ───────────────────────────┐
│ 规则文档管理 │ 冲突中心(逐条裁决) │ 图片检测(进度轮询) │ SVG叠加可视化 │ 检测历史(搜索+分页) │ 模型状态(一键Key配置) │
└───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                        │ HTTP / multipart（开发经 Vite 代理；容器经 Nginx 反代）
┌───────────────────────────────────────▼──────────────────────── FastAPI ──────────────────┐
│ api 层     documents / conflicts / inspection / system（统一异常处理 / CORS / 参数校验）    │
│            + providers/{name}/key（前端一键 Key 管理）                                      │
│                                                                                             │
│ services 层                                                                                 │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────────┐  │
│  │ 文档解析       │  │ 分块器       │  │ 冲突检测/裁决  │  │ 检测编排(进度/门禁/依据/历史)     │  │
│  │ PDF(pypdf)   │  │ 段落+重叠    │  │ LLM/启发式    │  │ PatchCore哨兵→YOLO-World定位    │  │
│  │ DOCX(docx)   │  │ 图注合并     │  │              │  │ → VLM/启发式裁判 → OpenCV轮廓   │  │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └──────────────┬─────────────────┘  │
│         │                 │                 │                          │                    │
│  ┌──────▼─────────────────▼──────┐   ┌───────▼───────────┐   ┌────────▼─────────┐          │
│  │ LLM 标准化适配（BaseLLM）        │   │ RAG 检索           │   │ 本地视觉模型        │          │
│  │ 智谱GLM / 千问 / DeepSeek / Mock│   │ Chroma(余弦+阈值)  │   │ PatchCore(ResNet18)│          │
│  │ API Key: DB(前端保存) > .env   │   │                    │   │ YOLO-World-S(ONNX) │          │
│  └───────────────────────────────┘   └───────▲───────────┘   └──────────────────┘          │
│         │                                    │                                               │
│  ┌──────▼──────────┐               ┌─────────┴──────────┐                                    │
│  │ Embedding 抽象   │──────────────▶│ 向量库(ChromaDB)    │                                    │
│  │ 智谱/千问/本地哈希 │               │ HNSW + cosine       │                                    │
│  └─────────────────┘               └────────────────────┘                                    │
└───────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                            │
                    SQLite(app.db)：文档 / chunk原文 / 规则 / 冲突 / 检测任务 / API Key / 偏好
                    backend/models/：resnet18_features.onnx + yolov8s-worldv2.onnx（随项目分发）
                    data/：上传文档、原图、标注图、向量库
```

设计要点：

- **门面 / 依赖注入**：API 层只调用 service 单例，service 通过工厂拿到模型与向量库实例，模型可替换、可测试；
- **面向接口**：`BaseLLM`、`EmbeddingClient`、`BaseParser` 定义统一协议，新增模型/格式只加适配器；
- **在线/离线双轨**：所有外部 AI 能力都有确定性兜底，系统无外部服务也可演示与验证；
- **API Key 热配置**：前端保存后即时生效（`lru_cache` 清除 + embedding 提供方切换 + RAG 后台重建），无需重启；
- **视觉管线 CPU 友好**：PatchCore + YOLO-World-S 均为 ONNX，模型文件随项目内置，零额外下载。

## 4. 快速开始（本地开发）

### 4.1 环境要求

- Python ≥ 3.10（开发验证：**Python 3.12**，Windows 11）
- Node.js ≥ 18（开发验证：Node 24）
- 无需外部数据库 / Docker（本地模式）
- **本地视觉管线**：仅需 CPU，模型已随项目内置（backend/models/）

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

# 3) （可选）通过 .env 预配置大模型 API Key
copy .env.example .env   # Linux/macOS: cp .env.example .env
#   编辑 .env 填入 ZHIPU_API_KEY / QWEN_API_KEY / DEEPSEEK_API_KEY 任意一家
#   也可在前端「模型状态」页一键配置（推荐，无需重启）

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

1. **规则文档**页上传 1~2 份 PDF/DOCX 检测标准；
2. 若出现冲突 → **冲突中心**逐条裁决（全部解决前无法检测）；
3. **缺陷检测**页上传一张工业品图片，观察实时进度（哨兵扫描 → 高危区域 → 动态切片 → 定位 → 裁决 → 轮廓）；
4. 查看框选结果、置信度与规则依据；
5. **模型状态**页确认当前是离线启发式裁判还是 VLM 裁决，也可在此一键配置 API Key 启用 VLM；
6. **检测历史**页回看，支持按图片名搜索与分页。

### 4.5 启用 VLM（视觉大模型裁决）

两种方式任选其一：

**方式 A：前端一键配置（推荐，无需重启）**

1. 打开前端「模型状态」页；
2. 在智谱 GLM 或阿里千问行粘贴 API Key，点击「保存」；
3. 保存后立即生效，顶部横幅会自动切换为在线模式；
4. 可点击「设为视觉」把该提供方固定为 VLM 裁决模型。

**方式 B：.env 环境变量**

```bash
# 编辑 backend/.env，填入以下任意一家
ZHIPU_API_KEY=sk-xxxx
QWEN_API_KEY=sk-xxxx
DEEPSEEK_API_KEY=sk-xxxx
# 可选：固定提供方
LLM_PROVIDER=zhipu        # 或 qwen / deepseek / 留空自动选择
VISION_PROVIDER=zhipu     # 或 qwen / 留空自动选择
```

Key 优先级：**前端 SQLite 保存值 > .env 环境变量**。清除前端保存后会自动回退到 .env。

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
- 本地 ONNX 模型已随项目内置，容器启动即可使用 PatchCore + YOLO-World 管线。

## 6. 使用指南

### 6.1 文档与冲突

- 仅接受 `.pdf` / `.docx`，服务端同时做扩展名白名单、**文件头魔数校验**与大小校验；
- 扫描件 PDF（无文字层）无法解析，系统会明确提示；
- 冲突卡片左/右分别展示两份文档的原文与页码，可「采纳 A」「采纳 B」或「自定义统一表述」；
  被淘汰规则状态变为 `superseded`，不再进入检测依据；自定义规则作为最终标准生效。

### 6.2 本地视觉管线参数

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DNN_PIPELINE_ENABLED` | true | 本地深度视觉管线总开关 |
| `SENTINEL_INPUT_SIZE` | 448 | PatchCore 哨兵推理分辨率（32 倍数） |
| `SENTINEL_CORESET_RATIO` | 0.01 | PatchCore coreset 采样比例（控内存） |
| `YOLO_INPUT_SIZE` | 640 | YOLO-World 推理分辨率 |
| `YOLO_CONF_THRESHOLD` | 0.01 | 零样本定位置信度下限（低置信候选交裁判过滤） |
| `TILE_MAX_SIZE` | 1024 | 动态切片窗边长上限 |
| `JUDGE_HEURISTIC_CONTRAST` | 50 | 离线启发式裁判：显著性对比度阈值 |
| `JUDGE_SENTINEL_MARGIN_SPARSE` | 0.24 | 区域稀少时哨兵幅度判据阈值 |
| `JUDGE_SENTINEL_MARGIN_DENSE` | 0.40 | 区域饱和时哨兵幅度判据阈值 |

### 6.3 检测参数（滑窗 / 后处理）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `WINDOW_SIZE` | 1024 | 固定方形滑窗边长 |
| `DET_CONFIDENCE_THRESHOLD` | 0.35 | 缺陷置信度过滤阈值 |
| `NMS_IOU_THRESHOLD` | 0.30 | 跨窗重复框 NMS 阈值 |
| `RETRIEVAL_TOP_K` | 4 | 每次规则检索返回片段数 |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | 0.30 | RAG 相似度下限（严格生效） |
| `IMAGE_MAX_PIXELS` | 800000000 | 大图像素安全阀，0 为不限制 |

### 6.4 更新本地 ONNX 模型

```bash
pip install torch torchvision ultralytics onnx onnxscript onnxslim
python backend/scripts/prepare_models.py
# 输出 backend/models/resnet18_features.onnx + yolov8s-worldv2.onnx
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

阈值过滤是**强制生效**的：低相似度片段不会进入模型上下文，避免"看似检索到了、其实不相关"的语义漂移。

### 7.2 本地深度视觉管线（4 阶段）

```
原始图像
  │
  ├─① PatchCore + ResNet18 哨兵（ONNX, CPU）
  │    全图异常热力图扫描 → P99 阈值 + 绝对下限 + 区域合并
  │    → 锁定 N 个高危区域（最多 12 个，过滤正常区域）
  │
  ├─② YOLO-World-S 空间定位（ONNX, CPU）
  │    对高危区域动态切片（窗边长 = min(区域长边 × 3.0, 1024)）
  │    内联英文缺陷标签列表，零样本定位输出候选框
  │    过滤两边 > 85% tile 的近满帧退化框
  │
  ├─③ 裁决
  │    · 有 VLM → 裁剪候选框 ROI → 连同 RAG 规则送视觉大模型 → 审核 / 分类
  │    · 无 VLM → 启发式裁判：对比度显著性 ≥ 50 OR 哨兵幅度 ≥ 阈值（分档）
  │    → 置信度过滤 + 同类型 IoU-NMS 去重
  │
  └─④ OpenCV 像素落地
       在最终缺陷框 ROI 内：大津法 + 形态学开运算 + 最大轮廓
       还原至原图坐标，生成半透明掩膜
```

管线优势：
- **哨兵-定位-裁决** 三层级联，大幅减少 VLM 调用次数（仅高危区域送裁）；
- YOLO-World **内联缺陷标签**（如 "porosity, scratch, crack, crater, chipping, ..."），无需预训练特定缺陷类别；
- 离线启发式裁判用**哨兵幅度**（区域哨兵分 − 全图距离中位数）作为第二证据：干净长图 P99 区域饱和（12 个），缺陷图区域稀少且幅度大，天然具备判别力；
- ONNX 模型**随项目内置**（backend/models/*.onnx），用户零下载即可运行。

### 7.3 固定方形滑窗（不重不漏）

- `max(W,H) ≤ window`：整图 resize 到 window×window **一次检测**，框按宽/高各自缩放比还原；
- 否则网格数 `n = ceil(dim / window)`，步长 `step = (dim - window)/(n - 1)`，
  末窗起点强制为 `dim - window`：所有窗完整、无 padding、并集恰为整图、边缘严格对齐；
- 框复原：网格模式 `(x+tx, y+ty)`；resize 模式 `(x·W/window, y·H/window)`，最后统一钳制边界。

### 7.4 前端标注坐标映射

```
原始坐标（像素） ← viewBox="0 0 W H" → SVG overlay
         ↑                                        ↑
    后端 bbox 返回                           CSS width:100%, height:100%
    mask_polygon 点位                        preserveAspectRatio="xMidYMid meet"
                                              wrapper: display:inline-block, max-width:100%
                                                └─ 收缩到 <img> 实际渲染尺寸
                                                   （wrapper = img = overlay 宽高一致）
```

修复前的错位根因：wrapper 被强制 100% 面板宽度，但 img 保持原图宽高比；`preserveAspectRatio="none"` 把 SVG viewBox 拉伸到变形的坐标系，框自然飞出。**修复后三者精确对齐，SVG 自然贴合。**

## 8. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/models` | 模型 / 本地视觉管线 / 哨兵状态 |
| **PUT** | `/api/providers/{name}/key` | **保存 API Key（保存即生效，支持前端一键配置）** |
| **DELETE** | `/api/providers/{name}/key` | **清除已保存的 API Key（.env 不受影响）** |
| **PUT** | `/api/providers/{name}/select` | **设为默认文本/视觉模型（name=zhipu\|qwen\|deepseek\|mock, role=text\|vision）** |
| POST | `/api/documents/upload` | 上传规则文档（multipart） |
| GET | `/api/documents` | 文档列表（含 max_documents） |
| GET | `/api/documents/rules` | 结构化规则 |
| DELETE | `/api/documents/{id}` | 删除文档（联动清理） |
| GET | `/api/conflicts` | 冲突列表（pending/resolved） |
| POST | `/api/conflicts/{id}/resolve` | 裁决冲突（A/B/CUSTOM） |
| POST | `/api/inspection/detect` | 上传图片创建检测任务 |
| GET | `/api/inspection/jobs/{id}` | 轮询进度/获取结果 |
| GET | `/api/inspection/history` | 历史列表（全量，前端本地搜索 + 分页） |
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
│  │  ├─ db/database.py         # SQLite：docs/chunk/规则/冲突/任务 + app_meta(Key/偏好)
│  │  ├─ schemas/               # Pydantic 请求响应模型
│  │  └─ services/
│  │     ├─ parser/             # PDF / DOCX 解析器（可扩展）
│  │     ├─ chunker.py          # 语义分块
│  │     ├─ embeddings.py       # Embedding 抽象 + 在线/离线实现（Key 支持 DB 优先）
│  │     ├─ vector_store.py     # Chroma 门面（阈值检索/重建/热切换）
│  │     ├─ llm/                # LLM 适配：基类/在线/Mock/工厂（lru_cache + 热重载）
│  │     ├─ key_store.py        # API Key + 提供方偏好 SQLite 存储 / 脱敏
│  │     ├─ conflict_service.py # 冲突检测与逐条裁决
│  │     ├─ document_service.py # 文档入库流水线
│  │     ├─ inspection_service.py# 检测任务编排/历史
│  │     └─ vision/             # 哨兵/定位/裁判/滑窗/NMS/OpenCV/标注
│  ├─ models/                   # ONNX 模型（随项目内置）
│  │  ├─ resnet18_features.onnx  # PatchCore 哨兵 backbone
│  │  └─ yolov8s-worldv2.onnx    # YOLO-World-S 零样本定位器
│  ├─ scripts/                  # 自检测 / 模型导出
│  ├─ requirements.txt
│  ├─ Dockerfile  .env.example
├─ frontend/
│  ├─ src/
│  │  ├─ api/client.ts          # API 封装（含 providers/{name}/key/select）
│  │  ├─ types.ts               # 与后端一致的 TS 类型
│  │  └─ components/            # 5 个功能面板 + 结果可视化
│  │     ├─ ModelStatusPanel.tsx # 前端一键 API Key 配置 + 默认模型选择
│  │     ├─ HistoryPanel.tsx    # 历史记录搜索 + 分页
│  │     └─ ResultView.tsx      # SVG 叠加标注（自适应缩放，坐标精确贴合）
│  ├─ package.json  vite.config.ts  Dockerfile  nginx.conf
└─ docker-compose.yml
```

## 10. 学习要点

1. **本地深度视觉管线架构**：哨兵→定位→裁判三层级联，PatchCore 做"要不要看"、YOLO-World 做"看哪里"、VLM/启发式做"算不算缺陷"，大幅减少 VLM 调用且保持零样本泛化能力；
2. **RAG 工程闭环**：解析 → 分块 → 向量化 → 阈值检索 → 溯源，元数据在写入点固化（页码/文档名/序号），检索结果天然可解释；
3. **面向接口的模型抽象**：`BaseLLM` / `EmbeddingClient` 让"换模型"只改配置；JSON 容错解析、退避重试是 LLM 工程的必备护栏；
4. **API Key 热配置**：前端 SQLite 存 Key + `lru_cache` 失效 + embedding 提供方切换 + RAG 后台重建，实现"保存即生效"零重启；
5. **SVG 叠加坐标映射**：`wrapper=img=overlay` 三者等宽高 + `preserveAspectRatio="xMidYMid meet"`，保证任意宽高比下标注精确贴合；
6. **大图与并发**：逐窗惰性裁剪、像素安全阀、CPU 任务 `asyncio.to_thread`、后台任务 + 轮询进度；
7. **安全基线**：魔数防伪、文件名消毒、`safe_join` 防穿越、Key 默认仅走 .env 环境变量（前端可选保存时走 DB）、统一错误不泄露堆栈；
8. **可降级架构**：每条外部能力都有本地兜底（本地 ONNX 管线、哈希 Embedding、启发式裁判、Mock LLM、OpenCV 轮廓），系统永远可跑通，便于离线演示与测试；
9. **跨平台交付**：纯 Python 解析库（无 poppler/libreoffice 系统依赖）+ 多架构容器 + 前后端分离 + ONNX CPU 推理，Windows/Linux/macOS 一致体验。

## 11. 常见问题（FAQ）

**Q：不配 API Key 能用吗？**
完全可以。自动进入离线模式：规则抽取/冲突检测用内置启发式（对"含明确阈值数字"的规则文档效果最佳），图像检测用本地 PatchCore + YOLO-World + 启发式裁判，适合验证流程与批量跑图。配置 Key 后裁决能力显著增强（VLM + RAG 深度结合）。

**Q：如何配置 API Key？**
推荐两种方式：① 前端「模型状态」页粘贴保存（即时生效，无需重启，Key 存在 SQLite 本地）；② `.env` 环境变量。优先级：前端保存 > .env。

**Q：检测时报"仍有 N 条冲突未确认"？**
这是系统的质量门禁：到「冲突中心」逐条裁决后再检测，确保执行标准无矛盾。

**Q：大图检测很慢？**
当前哨兵+定位是全图扫描（CPU ONNX，4000×15000 长条图约 5~10 秒），裁出高危区域后再送 YOLO-World。可先在 `.env` 调小 `TILE_MAX_SIZE` 减少切片尺寸，或增加 CPU 核数。

**Q：更换了 Embedding 提供方 / 清除了前端 Key？**
系统检测到向量空间变化会自动清空集合，并依据 SQLite 中的 chunk 原文后台重建索引，无需重新上传文档。重建期间新检测会短暂使用离线哈希 Embedding。

**Q：扫描版 PDF 怎么办？**
需要先 OCR（本系统不内置 OCR 以保持跨平台零系统依赖），生成带文字层的 PDF/DOCX 后再上传。

**Q：标注框为什么飞到图片外面？**
这是旧版本的 CSS bug（wrapper 强制 100% 面板宽 + SVG `preserveAspectRatio="none"`）。当前版本已修复：wrapper 收缩到 img 实际尺寸 + SVG 自然贴合，坐标映射精确。刷新页面即可看到正确效果。

**Q：本地 ONNX 模型在哪里？能自己更新吗？**
在 `backend/models/` 下，已随项目分发。如需更新（YOLO-World 新版本、PatchCore 新 backbone），运行 `pip install torch torchvision ultralytics onnx onnxscript onnxslim && python backend/scripts/prepare_models.py` 即可重新导出。
