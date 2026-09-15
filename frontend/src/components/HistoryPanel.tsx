import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { HistoryItem, InspectionJob } from '../types'
import ResultView from './ResultView'

const PAGE_SIZE = 20

/** 检测历史：搜索 + 分页 + 点击展开完整检测结果（复用 ResultView） */
export default function HistoryPanel() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [error, setError] = useState('')
  const [openJob, setOpenJob] = useState<InspectionJob | null>(null)
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const [keyword, setKeyword] = useState('')
  const [page, setPage] = useState(1)

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

  /** 关键字过滤（图片名 / 摘要），大小写不敏感 */
  const filtered = useMemo(() => {
    const k = keyword.trim().toLowerCase()
    if (!k) return items
    return items.filter(
      (it) =>
        it.image_name.toLowerCase().includes(k) ||
        (it.summary && it.summary.toLowerCase().includes(k)),
    )
  }, [items, keyword])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const effectivePage = Math.min(page, totalPages)
  const pageItems = filtered.slice(
    (effectivePage - 1) * PAGE_SIZE,
    effectivePage * PAGE_SIZE,
  )

  // 搜索结果为空时自动回到第 1 页
  useEffect(() => {
    if (page > totalPages) setPage(1)
  }, [page, totalPages])

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
          按时间倒序展示全部检测记录，支持按图片名搜索、翻页查看，点击任一条可展开
          原始尺寸标注与缺陷规则依据。
        </p>

        {items.length > 0 && (
          <div
            className="row"
            style={{
              marginBottom: 12,
              flexWrap: 'nowrap',
              gap: 12,
            }}
          >
            <input
              type="text"
              placeholder="按图片名或摘要搜索（实时过滤）"
              value={keyword}
              onChange={(e) => {
                setKeyword(e.target.value)
                setPage(1)
              }}
              style={{ flex: 1, minWidth: 200 }}
            />
            <span className="muted mono" style={{ fontSize: 12.5, whiteSpace: 'nowrap' }}>
              {keyword
                ? `匹配 ${filtered.length} / ${items.length}`
                : `共 ${items.length} 条`}
            </span>
          </div>
        )}

        {error && <div className="alert error">{error}</div>}
        {items.length === 0 ? (
          <div className="empty">暂无检测记录</div>
        ) : filtered.length === 0 ? (
          <div className="empty">没有匹配“{keyword}”的记录</div>
        ) : (
          <>
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
                {pageItems.map((it) => (
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

            {totalPages > 1 && (
              <div
                className="row"
                style={{
                  marginTop: 14,
                  justifyContent: 'center',
                  gap: 6,
                }}
              >
                <button
                  className="btn sm"
                  disabled={effectivePage === 1}
                  onClick={() => setPage(1)}
                >
                  « 首页
                </button>
                <button
                  className="btn sm"
                  disabled={effectivePage === 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                >
                  ‹ 上一页
                </button>
                <span
                  className="muted mono"
                  style={{ fontSize: 13, minWidth: 120, textAlign: 'center' }}
                >
                  第 {effectivePage} / {totalPages} 页
                </span>
                <button
                  className="btn sm"
                  disabled={effectivePage === totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                >
                  下一页 ›
                </button>
                <button
                  className="btn sm"
                  disabled={effectivePage === totalPages}
                  onClick={() => setPage(totalPages)}
                >
                  末页 »
                </button>
              </div>
            )}
          </>
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
