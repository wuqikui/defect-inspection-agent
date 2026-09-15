import { useState } from 'react'
import { api } from '../api/client'
import type { InspectionJob } from '../types'

/** 缺陷类型 → 展示颜色（与后端标注器配色近似） */
function colorOf(type: string): string {
  if (type.includes('划痕') || type.includes('划伤') || type.includes('刮伤'))
    return '#3b82f6'
  if (type.includes('起层') || type.includes('分层')) return '#a855f7'
  if (type.includes('裂纹')) return '#dc2626'
  if (type.includes('缺肉')) return '#eab308'
  return '#ef4444'
}

/**
 * 检测结果展示：结论横幅 + 原图框选/掩膜叠加 + 缺陷明细（含规则依据）
 * 叠加层使用与原图等宽高的 SVG viewBox，随图片自适应缩放，坐标无需手动换算。
 */
export default function ResultView({ job }: { job: InspectionJob }) {
  const [selected, setSelected] = useState<number | null>(null)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  const toggleExpand = (i: number) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })
  }

  return (
    <>
      <div className={`alert ${job.has_defect ? 'error' : 'success'}`}>
        <strong>
          {job.has_defect
            ? `检出 ${job.defects.length} 处缺陷，请复核`
            : '未检出缺陷，符合当前规则标准'}
        </strong>
        {job.summary && <div style={{ marginTop: 4 }}>{job.summary}</div>}
      </div>

      <div className="panel">
        <h2>缺陷位置标注（原始尺寸 {job.width} × {job.height}）</h2>
        <p className="desc">
          红色/蓝色框为检测边界框，半透明填充为分割掩膜；点击右侧缺陷项可高亮对应位置。
        </p>
        <div className="visual-stage">
          <img src={api.originalImageUrl(job.id)} alt={job.image_name} />
          <svg
            className="overlay"
            viewBox={`0 0 ${job.width} ${job.height}`}
            preserveAspectRatio="none"
          >
            {job.defects.map((d, i) => {
              const color = colorOf(d.defect_type)
              const active = selected === i
              const points = (d.mask_polygon || [])
                .map((p) => p.join(','))
                .join(' ')
              const { x, y, width, height } = d.bbox
              return (
                <g
                  key={i}
                  onMouseEnter={() => setSelected(i)}
                  onMouseLeave={() => setSelected(null)}
                  style={{ cursor: 'pointer' }}
                >
                  {d.segmentation_available && points && (
                    <polygon
                      points={points}
                      fill={color}
                      fillOpacity={active ? 0.35 : 0.16}
                      stroke={color}
                      strokeWidth={active ? 4 : 2}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  <rect
                    x={x}
                    y={y}
                    width={width}
                    height={height}
                    fill="none"
                    stroke={color}
                    strokeWidth={active ? 5 : 3}
                    vectorEffect="non-scaling-stroke"
                  />
                </g>
              )
            })}
          </svg>
        </div>
        <div className="legend">
          <span>图像坐标原点为左上角，单位为像素</span>
          {job.defects.some((d) => d.segmentation_available) && (
            <span>半透明区域 = 精确分割结果</span>
          )}
        </div>
      </div>

      {job.defects.length > 0 && (
        <div className="panel">
          <h2>缺陷明细与规则依据（{job.defects.length}）</h2>
          {job.defects.map((d, i) => {
            const open = expanded.has(i)
            const pct = Math.round(d.confidence * 100)
            return (
              <div key={i} className="defect-item">
                <div className="defect-head" onClick={() => toggleExpand(i)}>
                  <span className="tag red">#{i + 1} {d.defect_type}</span>
                  <span>
                    置信度 <strong>{pct}%</strong>
                  </span>
                  <span className="muted mono" style={{ fontSize: 12 }}>
                    bbox=({d.bbox.x}, {d.bbox.y}, {d.bbox.width}, {d.bbox.height})
                  </span>
                  <span className="spacer" />
                  <span className="muted" style={{ fontSize: 12 }}>
                    {open ? '收起 ▲' : '详情 / 规则依据 ▼'}
                  </span>
                </div>
                {open && (
                  <div className="defect-body">
                    {d.description && <p style={{ margin: '4px 0' }}>{d.description}</p>}
                    <div className="progress-wrap" style={{ maxWidth: 260 }}>
                      <div className="progress-bar" style={{ width: `${pct}%` }} />
                    </div>
                    <h3 style={{ fontSize: 13.5, margin: '12px 0 6px' }}>
                      判定依据（RAG 规则片段）
                    </h3>
                    {d.evidences.length === 0 && (
                      <div className="muted" style={{ fontSize: 12.5 }}>
                        未检索到高相似度规则片段，可能是规则库中暂无对应描述。
                      </div>
                    )}
                    {d.evidences.map((ev, j) => (
                      <div key={j} className="evidence">
                        <div className="meta">
                          📄 {ev.doc_name}
                          {ev.source_location ? ` · ${ev.source_location}` : ''} ·
                          相似度 <span className="sim">{ev.similarity.toFixed(3)}</span>
                        </div>
                        <div>{ev.content}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </>
  )
}
