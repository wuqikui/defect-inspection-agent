import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api } from '../api/client'
import type { InspectionRule, RuleDocument } from '../types'

interface Props {
  /** 文档变化（上传/删除）后通知父组件刷新冲突角标 */
  onChanged: () => void | Promise<void>
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`
  return `${(bytes / 1024).toFixed(1)} KB`
}

export default function DocumentPanel({ onChanged }: Props) {
  const [documents, setDocuments] = useState<RuleDocument[]>([])
  const [rules, setRules] = useState<InspectionRule[]>([])
  const [maxDocs, setMaxDocs] = useState(10)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [hint, setHint] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [showRules, setShowRules] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      const [docRes, ruleRes] = await Promise.all([
        api.listDocuments(),
        api.listRules(),
      ])
      setDocuments(docRes.items)
      setMaxDocs(docRes.max_documents)
      setRules(ruleRes.items)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载文档失败')
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const upload = useCallback(
    async (file: File) => {
      const ext = file.name.toLowerCase().split('.').pop() || ''
      if (!['pdf', 'docx'].includes(ext)) {
        setError('仅支持 PDF / DOCX 格式的规则文档')
        return
      }
      setLoading(true)
      setError('')
      setHint('')
      try {
        await api.uploadDocument(file)
        setHint(`《${file.name}》上传并解析成功，规则已入库`)
        await refresh()
        await onChanged()
      } catch (e) {
        setError(
          e instanceof ApiError
            ? e.message
            : '上传失败，请检查文件格式或后端服务状态',
        )
      } finally {
        setLoading(false)
        if (inputRef.current) inputRef.current.value = ''
      }
    },
    [refresh, onChanged],
  )

  const remove = useCallback(
    async (doc: RuleDocument) => {
      if (!window.confirm(`确认删除《${doc.original_name}》？相关向量、规则与冲突将一并清除。`))
        return
      try {
        await api.deleteDocument(doc.id)
        await refresh()
        await onChanged()
      } catch (e) {
        setError(e instanceof Error ? e.message : '删除失败')
      }
    },
    [refresh, onChanged],
  )

  const limitReached = documents.length >= maxDocs

  return (
    <>
      <section className="panel">
        <h2>规则文档库</h2>
        <p className="desc">
          上传 PDF / DOCX 格式的缺陷检测标准图文文档，系统将自动解析文字与图片说明、
          切分入库、构建语义向量并抽取结构化检测规则。最多 {maxDocs} 份文档，
          当前 {documents.length}/{maxDocs}。
        </p>

        {error && <div className="alert error">{error}</div>}
        {hint && <div className="alert success">{hint}</div>}

        <div
          className={`uploader ${dragOver ? 'dragover' : ''}`}
          onClick={() => !limitReached && !loading && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            const f = e.dataTransfer.files?.[0]
            if (f) upload(f)
          }}
          style={limitReached ? { cursor: 'not-allowed', opacity: 0.6 } : undefined}
        >
          <div className="big">
            {limitReached ? '已达文档数量上限' : '点击选择 或 拖拽文档到此处上传'}
          </div>
          <div>支持 .pdf / .docx · 单文件最大 50MB</div>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.docx"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) upload(f)
            }}
          />
        </div>
        {loading && (
          <div className="alert info" style={{ marginTop: 12 }}>
            正在解析文档、向量化并抽取规则，请稍候…
          </div>
        )}
      </section>

      <section className="panel">
        <div className="row">
          <h2 style={{ margin: 0 }}>已上传文档（{documents.length}）</h2>
          <span className="spacer" />
          <button className="btn sm" onClick={() => setShowRules((v) => !v)}>
            {showRules ? '隐藏规则' : `查看抽取规则（${rules.length}）`}
          </button>
        </div>

        {documents.length === 0 ? (
          <div className="empty">尚未上传任何规则文档</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>文档名称</th>
                <th>页数</th>
                <th>片段数</th>
                <th>大小</th>
                <th>内容摘要</th>
                <th>上传时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((d) => (
                <tr key={d.id}>
                  <td>
                    <strong>{d.original_name}</strong>
                    <div className="muted mono" style={{ fontSize: 12 }}>
                      {d.ext}
                    </div>
                  </td>
                  <td>{d.page_count}</td>
                  <td>{d.chunk_count}</td>
                  <td>{formatSize(d.size_bytes)}</td>
                  <td style={{ maxWidth: 320, color: 'var(--muted)' }}>
                    {d.summary || '—'}
                  </td>
                  <td className="mono" style={{ fontSize: 12 }}>
                    {new Date(d.uploaded_at).toLocaleString()}
                  </td>
                  <td>
                    <button className="btn sm danger" onClick={() => remove(d)}>
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {showRules && (
          <div style={{ marginTop: 18 }}>
            <h2 style={{ fontSize: 15 }}>结构化检测规则（{rules.length}）</h2>
            {rules.length === 0 ? (
              <div className="empty">暂未抽取到规则（离线模式下需文档中包含明确的判定性表述）</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>缺陷类型</th>
                    <th>规则内容</th>
                    <th>来源文档</th>
                    <th>溯源</th>
                    <th>状态</th>
                  </tr>
                </thead>
                <tbody>
                  {rules.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <span className="tag blue">{r.defect_type}</span>
                      </td>
                      <td style={{ maxWidth: 420 }}>{r.content}</td>
                      <td>{r.doc_name}</td>
                      <td className="muted">{r.source_location || '—'}</td>
                      <td>
                        <span className={`tag ${r.status === 'active' ? 'green' : 'gray'}`}>
                          {r.status === 'active' ? '生效中' : '冲突淘汰'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </section>
    </>
  )
}
