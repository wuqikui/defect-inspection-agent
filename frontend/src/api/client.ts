/**
 * 后端 API 客户端
 * ---------------
 * - 开发环境通过 Vite 代理访问 /api（见 vite.config.ts）
 * - 生产环境前后端同源（Nginx 反代），直接使用相对路径
 * - 统一解析后端 {"error": {"code","message"}} 错误结构并抛出
 */
import type {
  ConflictListResponse,
  DocumentListResponse,
  HistoryItem,
  InspectionJob,
  InspectionRule,
  ModelStatus,
} from '../types'

const BASE = '/api'

class ApiError extends Error {
  code: string
  status: number
  constructor(message: string, code: string, status: number) {
    super(message)
    this.code = code
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, init)
  if (!resp.ok) {
    let message = `请求失败（HTTP ${resp.status}）`
    let code = 'http_error'
    try {
      const body = await resp.json()
      if (body?.error) {
        message = body.error.message
        code = body.error.code
      }
    } catch {
      /* 忽略非 JSON 错误体 */
    }
    throw new ApiError(message, code, resp.status)
  }
  return (await resp.json()) as T
}

function postFile(path: string, field: string, file: File): Promise<unknown> {
  const form = new FormData()
  form.append(field, file)
  return request(path, { method: 'POST', body: form })
}

export const api = {
  /** ---- 系统 ---- */
  health: () => request<{ status: string; version: string }>('/health'),
  modelStatus: () => request<ModelStatus>('/models'),

  /** ---- 规则文档 ---- */
  listDocuments: () => request<DocumentListResponse>('/documents'),
  uploadDocument: (file: File) =>
    postFile('/documents/upload', 'file', file) as Promise<unknown>,
  deleteDocument: (id: string) =>
    request<{ deleted: string }>(`/documents/${id}`, { method: 'DELETE' }),
  listRules: () =>
    request<{ items: InspectionRule[]; total: number }>('/documents/rules'),

  /** ---- 冲突 ---- */
  listConflicts: () => request<ConflictListResponse>('/conflicts'),
  resolveConflict: (
    id: string,
    payload: { choice: 'A' | 'B' | 'CUSTOM'; custom_text?: string },
  ) =>
    request(`/conflicts/${id}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  /** ---- 检测 ---- */
  startDetection: (file: File) =>
    postFile('/inspection/detect', 'file', file) as Promise<{
      job_id: string
      status: string
    }>,
  getJob: (id: string) => request<InspectionJob>(`/inspection/jobs/${id}`),
  history: () => request<HistoryItem[]>('/inspection/history'),

  /** ---- 图片资源 URL（直接用于 <img src>） ---- */
  originalImageUrl: (jobId: string) => `${BASE}/inspection/image/${jobId}`,
  resultImageUrl: (name: string) => `${BASE}/inspection/results/${name}`,
}

export { ApiError }
