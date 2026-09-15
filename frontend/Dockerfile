# =====================================================================
# 前端镜像：两阶段构建（Node 编译 → Nginx 托管静态资源并反代后端）
# 构建上下文：frontend/
# =====================================================================

# ---- Stage 1: 构建静态资源 ----
FROM node:20-alpine AS builder
WORKDIR /build
COPY package.json package-lock.json* ./
# 有 lockfile 时用 npm ci；无 lockfile 时回退 install
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY . .
RUN npm run build

# ---- Stage 2: Nginx 运行时 ----
FROM nginx:1.27-alpine
COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
