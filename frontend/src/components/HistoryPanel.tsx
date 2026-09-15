import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { HistoryItem, InspectionJob } from '../types'
import ResultView from './ResultView'

/** 检测历史：列表 + 点击展开完整检测结果（复用 ResultView） */
export default function HistoryPanel() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [error, setError] = useState('')
  const [openJob, setOpenJob] = useState<InspectionJob | null>(null)
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setItems(await api.history())
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载历史失败')
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const openDetail = useCallback(async (id: string) => {
    if (openJob?.id === id) {
      setOpenJob(null)
      return
    }
    setLoadingId(id)
    setError('')
    try {
      setOpenJob(await api.getJob(id))
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载详情失败')
    } finally {
      setLoadingId(null)
    }
  }, [openJob])

  return (
    <>
      <section className="panel">
        <h2>检测历史</h2>
        <p className="desc">
          按时间倒序展示最近的检测记录，点击任一条可查看原图标注、缺陷坐标与规则依据。
        </p>
        {error && <div className="alert error">{error}</div>}
        {items.length === 0 ? (
          <div className="empty">暂无检测记录</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th style={{ width: 120 }}>缩略图</th>
                <th>图片名称</th>
                <th>原始尺寸</th>
                <th>结论</th>
                <th>摘要</th>
                <th>检测时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.id}>
                  <td>
                    {it.annotated_url ? (
                      <img
                        src={it.annotated_url}
                        alt=""
                        style={{
                          width: 104,
                          height: 70,
                          objectFit: 'cover',
                          borderRadius: 6,
                          border: '1px solid var(--border)',
                        }}
                      />
                    ) : (
                      <div
                        style={{
                          width: 104,
                          height: 70,
                          background: '#f0f2f5',
                          borderRadius: 6,
                        }}
                      />
                    )}
                  </td>
                  <td style={{ maxWidth: 200 }}>{it.image_name}</td>
                  <td className="mono" style={{ fontSize: 12 }}>
                    {it.width}×{it.height}
                  </td>
                  <td>
                    <span className={`tag ${it.has_defect ? 'red' : 'green'}`}>
                      {it.has_defect ? `缺陷 ×${it.defect_count}` : '合格'}
                    </span>
                  </td>
                  <td
                    className="muted"
                    style={{ maxWidth: 300, fontSize: 12.5 }}
                  >
                    {it.summary}
                  </td>
                  <td className="mono" style={{ fontSize: 12 }}>
                    {new Date(it.created_at).toLocaleString()}
                  </td>
                  <td>
                    <button
                      className="btn sm"
                      disabled={loadingId === it.id}
                      onClick={() => openDetail(it.id)}
                    >
                      {openJob?.id === it.id ? '收起' : '查看详情'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {openJob?.status === 'completed' && (
        <section className="panel">
          <div className="row" style={{ marginBottom: 10 }}>
            <h2 style={{ margin: 0 }}>{openJob.image_name}</h2>
            <span className="spacer" />
            <button className="btn sm" onClick={() => setOpenJob(null)}>
              关闭详情
            </button>
          </div>
          <ResultView job={openJob} />
        </section>
      )}
    </>
  )
}
