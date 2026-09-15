import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api } from '../api/client'
import type { InspectionJob } from '../types'
import ResultView from './ResultView'

interface Props {
  /** 冲突被解决后通知父组件刷新角标 */
  onConflictSolved: () => void | Promise<void>
}

export default function InspectionPanel({ onConflictSolved }: Props) {
  const [job, setJob] = useState<InspectionJob | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const timerRef = useRef<number | null>(null)

  /** 停止轮询 */
  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  useEffect(() => stopPolling, [stopPolling])

  /** 启动轮询：1.5s 拉取一次任务状态，直到 completed/failed */
  const startPolling = useCallback(
    (jobId: string) => {
      stopPolling()
      const tick = async () => {
        try {
          const j = await api.getJob(jobId)
          setJob(j)
          if (j.status !== 'processing') {
            stopPolling()
            onConflictSolved()
          }
        } catch {
          /* 单次轮询失败不中断，下次再试 */
        }
      }
      timerRef.current = window.setInterval(tick, 1500)
      tick()
    },
    [stopPolling, onConflictSolved],
  )

  const upload = useCallback(
    async (file: File) => {
      setUploading(true)
      setError('')
      setJob(null)
      try {
        // 限制前端预校验，减少无效上传
        const created = await api.startDetection(file)
        startPolling(created.job_id)
      } catch (e) {
        if (e instanceof ApiError) {
          setError(e.message)
        } else {
          setError('检测任务创建失败，请确认后端服务可用')
        }
      } finally {
        setUploading(false)
        if (inputRef.current) inputRef.current.value = ''
      }
    },
    [startPolling],
  )

  const processing = job?.status === 'processing'

  return (
    <>
      <section className="panel">
        <h2>工业图片缺陷检测</h2>
        <p className="desc">
          单次上传一张任意尺寸的工业产品图片（JPG/PNG/BMP/TIFF/WEBP，≤100MB）。
          系统使用固定方形滑窗（默认 1024px）扫描全图：小图整图缩放一次检测，
          大图自动计算步长不重不漏覆盖，检测框精确复原到原图坐标。
        </p>

        {error && <div className="alert error">{error}</div>}

        <div
          className={`uploader ${dragOver ? 'dragover' : ''}`}
          style={processing ? { pointerEvents: 'none', opacity: 0.6 } : undefined}
          onClick={() => !uploading && inputRef.current?.click()}
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
        >
          <div className="big">
            {processing ? '检测进行中…' : '点击选择 或 拖拽待测图片到此处'}
          </div>
          <div>单张图片 · 支持超大尺寸 · JPG / PNG / BMP / TIFF / WEBP</div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) upload(f)
            }}
          />
        </div>

        {job && processing && (
          <div style={{ marginTop: 16 }}>
            <div className="row">
              <strong>
                {job.image_name}（{job.width} × {job.height}）
              </strong>
              <span className="spacer" />
              <span className="mono">{job.progress}%</span>
            </div>
            <div className="progress-wrap">
              <div className="progress-bar" style={{ width: `${job.progress}%` }} />
            </div>
            <div className="progress-text">{job.stage}</div>
          </div>
        )}

        {job?.status === 'failed' && (
          <div className="alert error" style={{ marginTop: 14 }}>
            检测失败：{job.error_message || '未知错误'}
            <div style={{ marginTop: 6 }}>
              <button className="btn sm" onClick={() => setJob(null)}>
                重新上传
              </button>
            </div>
          </div>
        )}
      </section>

      {job?.status === 'completed' && <ResultView job={job} />}
    </>
  )
}
