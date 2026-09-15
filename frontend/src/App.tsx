import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import ConflictPanel from './components/ConflictPanel'
import DocumentPanel from './components/DocumentPanel'
import HistoryPanel from './components/HistoryPanel'
import InspectionPanel from './components/InspectionPanel'
import ModelStatusPanel from './components/ModelStatusPanel'

/** 一级功能页签 */
type TabKey = 'documents' | 'conflicts' | 'inspection' | 'history' | 'models'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'documents', label: '规则文档' },
  { key: 'conflicts', label: '冲突中心' },
  { key: 'inspection', label: '缺陷检测' },
  { key: 'history', label: '检测历史' },
  { key: 'models', label: '模型状态' },
]

export default function App() {
  const [tab, setTab] = useState<TabKey>('documents')
  const [pendingConflicts, setPendingConflicts] = useState(0)
  const [online, setOnline] = useState<boolean | null>(null)

  /** 拉取待处理冲突数（角标），多个面板操作后都会刷新 */
  const refreshBadge = useCallback(async () => {
    try {
      const res = await api.listConflicts()
      setPendingConflicts(res.pending_count)
    } catch {
      /* 后端未启动时静默处理 */
    }
  }, [])

  useEffect(() => {
    const ping = async () => {
      try {
        await api.health()
        setOnline(true)
      } catch {
        setOnline(false)
      }
    }
    ping()
    const timer = setInterval(ping, 10000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    refreshBadge()
  }, [refreshBadge, tab])

  return (
    <>
      <header className="app-header">
        <div>
          <h1>多模态缺陷检测智能体</h1>
          <div className="subtitle">
            RAG 规则知识库 · 滑窗视觉检测 · 冲突人工确认 · 检测历史回溯
          </div>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`tab ${tab === t.key ? 'active' : ''}`}
              onClick={() => setTab(t.key)}
            >
              {t.label}
              {t.key === 'conflicts' && pendingConflicts > 0 && (
                <span className="badge">{pendingConflicts}</span>
              )}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {online === false && (
          <div className="alert error">
            无法连接后端服务（http://localhost:8000），请确认后端已启动。
          </div>
        )}

        {tab === 'documents' && (
          <DocumentPanel onChanged={refreshBadge} />
        )}
        {tab === 'conflicts' && (
          <ConflictPanel onChange={refreshBadge} />
        )}
        {tab === 'inspection' && (
          <InspectionPanel onConflictSolved={refreshBadge} />
        )}
        {tab === 'history' && <HistoryPanel />}
        {tab === 'models' && <ModelStatusPanel />}
      </main>
    </>
  )
}
