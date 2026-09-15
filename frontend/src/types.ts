/**
 * 前后端共享的数据类型定义（与后端 Pydantic schema 一一对应）
 */

/** 规则文档 */
export interface RuleDocument {
  id: string
  original_name: string
  ext: string
  size_bytes: number
  status: string
  page_count: number
  chunk_count: number
  summary: string
  uploaded_at: string
}

export interface DocumentListResponse {
  items: RuleDocument[]
  total: number
  max_documents: number
}

/** 结构化检测规则 */
export interface InspectionRule {
  id: string
  doc_id: string
  doc_name: string
  defect_type: string
  title: string
  content: string
  severity: string
  source_location: string
  status: 'active' | 'superseded'
}

/** 规则冲突 */
export interface RuleConflict {
  id: string
  defect_type: string
  description: string
  doc_a_name: string
  doc_b_name: string
  content_a: string
  content_b: string
  source_a: string
  source_b: string
  status: 'pending' | 'resolved'
  resolution_choice: string
  resolution_text: string
  created_at: string
  resolved_at?: string
}

export interface ConflictListResponse {
  items: RuleConflict[]
  pending_count: number
  resolved_count: number
}

/** 检测结果 */
export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface RuleEvidence {
  doc_name: string
  source_location: string
  content: string
  similarity: number
}

export interface DefectResult {
  defect_type: string
  confidence: number
  bbox: BoundingBox
  segmentation_available: boolean
  mask_polygon?: [number, number][] | null
  description: string
  evidences: RuleEvidence[]
  tile_index: number
}

export type JobStatus = 'processing' | 'completed' | 'failed'

export interface InspectionJob {
  id: string
  image_name: string
  width: number
  height: number
  status: JobStatus
  progress: number
  stage: string
  has_defect: boolean
  summary: string
  defects: DefectResult[]
  error_message: string
  created_at: string
  completed_at?: string
  image_url: string
  annotated_url: string
}

/** 历史记录 */
export interface HistoryItem {
  id: string
  image_name: string
  width: number
  height: number
  has_defect: boolean
  defect_count: number
  summary: string
  created_at: string
  annotated_url: string
}

/** 模型状态 */
export interface ProviderStatus {
  name: string
  label: string
  configured: boolean
  supports_vision: boolean
  chat_model: string
  selected: boolean
}

export interface ModelStatus {
  active_text_provider: string
  active_vision_provider: string
  offline_mode: boolean
  sam_available: boolean
  sam_device: string
  providers: ProviderStatus[]
}
