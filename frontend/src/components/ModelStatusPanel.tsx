import { useCallback, useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { api } from '../api/client'
import type { ModelStatus, ProviderStatus } from '../types'

/** API Key 输入框内联样式（type=password 不在全局 input[type=text] 选择器内） */
const keyInputStyle: CSSProperties = {
  flex: 1,
  minWidth: 150,
  border: '1px solid var(--border)',
  borderRadius: 8,
  padding: '5px 10px',
  fontSize: 12.5,
  fontFamily: 'ui-monospace, Consolas, monospace',
}

/** 模型与推理后端状态总览（支持前端一键配置 API Key / 切换默认模型） */
export default function ModelStatusPanel() {
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState<{ type: 'success' | 'error'; text: string } | null>(
    null,
  )
  const [keyInputs, setKeyInputs] = useState<Record<string, string>>({})
  const [busyName, setBusyName] = useState('')

  const refresh = useCallback(() => {
    return api
      .modelStatus()
      .then(setStatus)
      .catch((e) => setError(e instanceof Error ? e.message : '获取模型状态失败'))
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  function flash(type: 'success' | 'error', text: string) {
    setNotice({ type, text })
    window.setTimeout(() => setNotice(null), 6000)
  }

  /** 统一执行写操作：成功后提示 + 刷新状态 */
  async function run(name: string, action: () => Promise<{ message: string }>) {
    setBusyName(name)
    try {
      const res = await action()
      flash('success', res.message)
      setKeyInputs((m) => ({ ...m, [name]: '' }))
      await refresh()
    } catch (e) {
      flash('error', e instanceof Error ? e.message : '操作失败，请稍后重试')
    } finally {
      setBusyName('')
    }
  }

  function saveKey(p: ProviderStatus) {
    const key = (keyInputs[p.name] || '').trim()
    if (key.length < 8) {
      flash('error', '请输入有效的 API Key（至少 8 个字符）。')
      return
    }
    run(p.name, () => api.saveProviderKey(p.name, key))
  }

  if (error) return <div className="alert error">{error}</div>
  if (!status) return <div className="empty">加载中…</div>

  return (
    <>
      <section className="panel">
        <h2>模型配置状态</h2>
        <p className="desc">
          系统支持智谱 GLM、阿里千问、DeepSeek 三种在线大模型：在下方表格直接粘贴
          API Key 并点击「保存」即可一键启用，保存后立即生效，
          <strong>无需修改 .env、无需重启后端</strong>。Key 保存在本地数据库
          （data/app.db，仅本机可读），界面只显示脱敏提示。已配置的模型还可一键设为
          默认文本模型 / 视觉裁决模型（VLM）。未配置任何 Key 时自动运行离线模式。
        </p>

        {notice && <div className={`alert ${notice.type}`}>{notice.text}</div>}

        {status.offline_mode ? (
          <div className="alert warn">
            当前处于<strong>离线兜底模式</strong>：规则抽取 / 冲突检测使用内置启发式引擎，
            图像缺陷检测使用本地深度视觉管线（离线裁判）。在下方任意一家提供方粘贴
            API Key 并保存，即可立即切换为大模型能力。
          </div>
        ) : (
          <div className="alert success">
            文本模型：{status.active_text_provider}；
            视觉模型：{status.active_vision_provider || '未配置（检测走本地启发式裁判）'}
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
              <th style={{ width: 300 }}>操作</th>
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
                  {p.key_hint && (
                    <div className="muted mono" style={{ fontSize: 11 }}>
                      Key：{p.key_hint}
                    </div>
                  )}
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
                  {p.name === 'mock' ? (
                    p.selected && <span className="tag green">使用中</span>
                  ) : (
                    <>
                      {p.selected && <span className="tag green">文本模型</span>}
                      {p.selected_vision && <span className="tag blue">VLM 裁决</span>}
                    </>
                  )}
                </td>
                <td>
                  {p.name === 'mock' ? (
                    <button
                      className="btn sm"
                      disabled={p.selected || busyName === p.name}
                      onClick={() =>
                        run(p.name, () => api.selectProvider(p.name, 'text'))
                      }
                    >
                      设为文本模型
                    </button>
                  ) : (
                    <>
                      <div className="row" style={{ flexWrap: 'nowrap' }}>
                        <input
                          type="password"
                          autoComplete="off"
                          placeholder={
                            p.configured ? '已配置，输入新 Key 可覆盖' : '粘贴 API Key'
                          }
                          value={keyInputs[p.name] || ''}
                          disabled={busyName === p.name}
                          onChange={(e) =>
                            setKeyInputs((m) => ({ ...m, [p.name]: e.target.value }))
                          }
                          onKeyDown={(e) => e.key === 'Enter' && saveKey(p)}
                          style={keyInputStyle}
                        />
                        <button
                          className="btn primary sm"
                          disabled={busyName === p.name}
                          onClick={() => saveKey(p)}
                        >
                          保存
                        </button>
                        {p.configured && (
                          <button
                            className="btn sm danger"
                            disabled={busyName === p.name}
                            onClick={() =>
                              run(p.name, () => api.clearProviderKey(p.name))
                            }
                          >
                            清除
                          </button>
                        )}
                      </div>
                      <div className="row" style={{ marginTop: 6 }}>
                        <button
                          className="btn sm"
                          disabled={!p.configured || p.selected || busyName === p.name}
                          onClick={() =>
                            run(p.name, () => api.selectProvider(p.name, 'text'))
                          }
                        >
                          设为文本
                        </button>
                        <button
                          className="btn sm"
                          disabled={
                            !p.configured ||
                            !p.supports_vision ||
                            p.selected_vision ||
                            busyName === p.name
                          }
                          onClick={() =>
                            run(p.name, () => api.selectProvider(p.name, 'vision'))
                          }
                        >
                          设为视觉
                        </button>
                      </div>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>缺陷检测管线（本地视觉模型）</h2>
        <p className="desc">
          图像缺陷检测由三级本地管线完成，模型文件已随项目内置
          （backend/models/*.onnx），纯 CPU 推理，无需安装依赖或下载权重：
        </p>
        <table>
          <thead>
            <tr>
              <th>级别</th>
              <th>模型 / 方法</th>
              <th>职责</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>① 全局哨兵</td>
              <td className="mono" style={{ fontSize: 12 }}>
                PatchCore + ResNet18（ONNX）
              </td>
              <td>全图异常热力图扫描，过滤正常区域，锁定高危区域</td>
              <td>
                <span
                  className={`tag ${status.dnn_pipeline_enabled && status.sentinel_ready ? 'green' : 'gray'}`}
                >
                  {status.dnn_pipeline_enabled
                    ? status.sentinel_ready
                      ? '就绪'
                      : '模型缺失'
                    : '已关闭'}
                </span>
              </td>
            </tr>
            <tr>
              <td>② 空间定位</td>
              <td className="mono" style={{ fontSize: 12 }}>
                YOLO-World-S（ONNX，内联缺陷标签）
              </td>
              <td>对高危区域动态切片，零样本定位输出缺陷矩形框</td>
              <td>
                <span
                  className={`tag ${status.dnn_pipeline_enabled && status.locator_ready ? 'green' : 'gray'}`}
                >
                  {status.dnn_pipeline_enabled
                    ? status.locator_ready
                      ? '就绪'
                      : '模型缺失'
                    : '已关闭'}
                </span>
              </td>
            </tr>
            <tr>
              <td>③ 裁决</td>
              <td>
                {status.active_vision_provider
                  ? '云端视觉大模型（VLM）'
                  : '对比度显著性 + 哨兵幅度（启发式）'}
              </td>
              <td>结合 RAG 检索规则审核候选框，过滤误报并纠正类型</td>
              <td>
                <span className={`tag ${status.active_vision_provider ? 'green' : 'blue'}`}>
                  {status.active_vision_provider ? 'VLM 裁决' : '离线启发式'}
                </span>
              </td>
            </tr>
            <tr>
              <td>④ 像素落地</td>
              <td className="mono" style={{ fontSize: 12 }}>
                OpenCV（大津法 + 形态学轮廓）
              </td>
              <td>仅在最终缺陷框 ROI 内提取像素级轮廓并还原至原图坐标</td>
              <td>
                <span className="tag green">就绪</span>
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </>
  )
}
