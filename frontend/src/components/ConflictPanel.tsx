import { useCallback, useEffect, useState } from 'react'
import { ApiError, api } from '../api/client'
import type { RuleConflict } from '../types'

interface Props {
  /** 冲突状态变化后通知父组件刷新角标 */
  onChange: () => void | Promise<void>
}

export default function ConflictPanel({ onChange }: Props) {
  const [conflicts, setConflicts] = useState<RuleConflict[]>([])
  const [pending, setPending] = useState(0)
  const [error, setError] = useState('')
  /** 正在展开自定义输入的冲突 id */
  const [customId, setCustomId] = useState<string | null>(null)
  const [customText, setCustomText] = useState('')
  const [busy, setBusy] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const res = await api.listConflicts()
      setConflicts(res.items)
      setPending(res.pending_count)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载冲突失败')
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const resolve = useCallback(
    async (c: RuleConflict, choice: 'A' | 'B' | 'CUSTOM') => {
      if (choice === 'CUSTOM') {
        // 第一次点击：展开输入框；第二次（确认提交）才发请求
        if (customId !== c.id) {
          setCustomId(c.id)
          setCustomText('')
          return
        }
        if (!customText.trim()) {
          setError('请填写统一规则表述后再确认')
          return
        }
      }
      setBusy(c.id)
      setError('')
      try {
        await api.resolveConflict(c.id, {
          choice,
          custom_text: choice === 'CUSTOM' ? customText.trim() : undefined,
        })
        setCustomId(null)
        await refresh()
        await onChange()
      } catch (e) {
        setError(
          e instanceof ApiError
            ? e.message
            : '裁决失败，请稍后重试',
        )
      } finally {
        setBusy(null)
      }
    },
    [customId, customText, refresh, onChange],
  )

  return (
    <section className="panel">
      <h2>规则冲突中心</h2>
      <p className="desc">
        不同规则文档对同一缺陷的判定标准存在矛盾时，系统在此逐条列出。
        请对每条冲突选择以哪份文档为准，或填写统一的自定义表述；
        <strong> 所有冲突确认完毕后才能发起缺陷检测</strong>。
      </p>

      {error && <div className="alert error">{error}</div>}

      <div className="row" style={{ marginBottom: 14 }}>
        <span className="tag red">待确认 {pending}</span>
        <span className="tag green">已解决 {conflicts.length - pending}</span>
      </div>

      {pending === 0 && conflicts.length === 0 && (
        <div className="empty">
          暂无冲突。上传两份及以上规则文档后，系统会自动比对冲突；
          若所有规则一致则此处保持为空。
        </div>
      )}

      {conflicts.map((c) => {
        const resolved = c.status === 'resolved'
        return (
          <div key={c.id} className={`conflict-card ${resolved ? 'resolved' : ''}`}>
            <div className="conflict-head">
              <span className="tag amber">冲突类型：{c.defect_type || '通用'}</span>
              {resolved ? (
                <span className="tag green">
                  已解决（采纳 {c.resolution_choice === 'CUSTOM' ? '自定义' : c.resolution_choice}）
                </span>
              ) : (
                <span className="tag red">待确认</span>
              )}
              <span className="muted" style={{ fontSize: 12 }}>
                {new Date(c.created_at).toLocaleString()}
              </span>
            </div>

            <div>{c.description}</div>

            <div className="conflict-sides">
              <div className={`side-box ${resolved && c.resolution_choice === 'A' ? 'picked' : ''}`}>
                <div className="src">
                  📄 {c.doc_a_name}
                  {c.source_a ? ` · ${c.source_a}` : ''}
                </div>
                <div>【说法 A】{c.content_a}</div>
              </div>
              <div className={`side-box ${resolved && c.resolution_choice === 'B' ? 'picked' : ''}`}>
                <div className="src">
                  📄 {c.doc_b_name}
                  {c.source_b ? ` · ${c.source_b}` : ''}
                </div>
                <div>【说法 B】{c.content_b}</div>
              </div>
            </div>

            {resolved ? (
              <div className="alert success" style={{ margin: 0 }}>
                最终采用规则：{c.resolution_text}
              </div>
            ) : (
              <>
                {customId === c.id && (
                  <textarea
                    rows={3}
                    placeholder="请输入经确认后统一执行的规则表述（含具体阈值与单位）…"
                    value={customText}
                    onChange={(e) => setCustomText(e.target.value)}
                    style={{ marginBottom: 10 }}
                  />
                )}
                <div className="conflict-actions">
                  <button
                    className="btn primary sm"
                    disabled={busy === c.id}
                    onClick={() => resolve(c, 'A')}
                  >
                    采纳 A（以《{c.doc_a_name}》为准）
                  </button>
                  <button
                    className="btn primary sm"
                    disabled={busy === c.id}
                    onClick={() => resolve(c, 'B')}
                  >
                    采纳 B（以《{c.doc_b_name}》为准）
                  </button>
                  <button
                    className="btn sm"
                    disabled={busy === c.id}
                    onClick={() => resolve(c, 'CUSTOM')}
                  >
                    {customId === c.id ? '确认自定义表述' : '自定义统一表述'}
                  </button>
                </div>
              </>
            )}
          </div>
        )
      })}
    </section>
  )
}
