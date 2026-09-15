import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { ModelStatus } from '../types'

/** 模型与推理后端状态总览 */
export default function ModelStatusPanel() {
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .modelStatus()
      .then(setStatus)
      .catch((e) => setError(e instanceof Error ? e.message : '获取模型状态失败'))
  }, [])

  if (error) return <div className="alert error">{error}</div>
  if (!status) return <div className="empty">加载中…</div>

  return (
    <>
      <section className="panel">
        <h2>模型配置状态</h2>
        <p className="desc">
          系统支持智谱 GLM、阿里千问、DeepSeek 三种在线大模型，通过后端环境变量
          （.env）配置 API Key 即可启用；未配置时自动运行在离线模式：
          文本理解使用规则启发式引擎，图像检测使用经典 OpenCV 视觉算法。
        </p>

        {status.offline_mode ? (
          <div className="alert warn">
            当前处于<strong>离线兜底模式</strong>：规则抽取 / 冲突检测使用内置启发式引擎，
            图像缺陷检测使用经典 CV 算法。配置任意一家 API Key 后重启后端即可切换为大模型能力。
          </div>
        ) : (
          <div className="alert success">
            文本模型：{status.active_text_provider}；
            视觉模型：{status.active_vision_provider || '未配置（视觉走 CV 兜底）'}
          </div>
        )}

        <table>
          <thead>
            <tr>
              <th>提供方</th>
              <th>配置状态</th>
              <th>视觉能力</th>
              <th>默认对话模型</th>
              <th>当前生效</th>
            </tr>
          </thead>
          <tbody>
            {status.providers.map((p) => (
              <tr key={p.name}>
                <td>
                  <strong>{p.label}</strong>
                  <div className="muted mono" style={{ fontSize: 12 }}>
                    {p.name}
                  </div>
                </td>
                <td>
                  <span className={`tag ${p.configured ? 'green' : 'gray'}`}>
                    {p.configured ? '已配置' : '未配置'}
                  </span>
                </td>
                <td>
                  <span className={`tag ${p.supports_vision ? 'blue' : 'gray'}`}>
                    {p.supports_vision ? '支持图像理解' : '仅文本'}
                  </span>
                </td>
                <td className="mono" style={{ fontSize: 12 }}>
                  {p.chat_model}
                </td>
                <td>
                  {p.selected && <span className="tag green">使用中</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>缺陷分割（SAM）</h2>
        <div className="kv-grid">
          <div>
            <div className="k">SAM 精确分割</div>
            <div>
              <span className={`tag ${status.sam_available ? 'green' : 'gray'}`}>
                {status.sam_available
                  ? `可用（device: ${status.sam_device}）`
                  : '未启用，使用 OpenCV 轮廓分割兜底'}
              </span>
            </div>
          </div>
          <div style={{ gridColumn: '1 / -1' }} className="muted">
            启用方式：安装可选依赖
            <code> pip install torch segment-anything</code>，
            并将 SAM 权重（如 sam_vit_b_01ec64.pth）放入
            backend/data/sam_checkpoints/ 后重启服务；检测框会作为 box prompt
            交给 SAM 输出像素级掩膜。
          </div>
        </div>
      </section>
    </>
  )
}
