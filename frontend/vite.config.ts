import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite 配置：
// 1. 开发服务器固定 5173 端口；
// 2. /api 与 /docs 代理到 FastAPI 后端（8000），前端无需处理跨域；
// 3. build 输出到 dist/，由 Dockerfile 构建为 Nginx 静态资源。
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
})
