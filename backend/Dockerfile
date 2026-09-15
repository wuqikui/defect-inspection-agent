# =====================================================================
# 后端镜像：Python 3.12 slim + FastAPI
# 构建上下文：backend/
# =====================================================================
FROM python:3.12-slim

# 不生成 .pyc、日志不缓冲，便于容器内排障
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# opencv-python-headless 运行时仍依赖 libglib（无需 libGL，故用 headless）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 再拷贝业务代码
COPY app ./app

RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000

# 容器健康检查：命中 /api/health
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
