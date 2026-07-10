import { useEffect, useState } from 'react'
import { useStore } from '../store/store'
import { getHealth } from '../api/client'
import type { Health, ProviderKeys } from '../types'

const PROVIDERS: { key: keyof ProviderKeys; label: string; hint: string }[] = [
  { key: 'claude', label: 'Claude', hint: 'Anthropic（默认画面/分类模型）' },
  { key: 'openai', label: 'OpenAI', hint: 'GPT 兼容网关' },
  { key: 'glm', label: 'GLM', hint: '智谱 / 兼容网关' },
]

export function KeyPanel() {
  const { keys, setKey, remember, setRemember } = useStore()
  const [open, setOpen] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {})
  }, [])

  const filled = Object.values(keys).filter((v) => v.trim()).length

  return (
    <section className="card">
      <div className="card-head" onClick={() => setOpen((o) => !o)}>
        <span className="card-title">模型 Key 设置</span>
        <span className="card-meta">
          {filled > 0 ? `已填 ${filled} 个` : '未填（可用后端预置 Key）'} {open ? '▲' : '▼'}
        </span>
      </div>
      {open && (
        <div className="key-body">
          {PROVIDERS.map((p) => (
            <div className="key-row" key={p.key}>
              <label>{p.label}<small>{p.hint}</small></label>
              <input
                type="password"
                placeholder="sk-..."
                value={keys[p.key]}
                onChange={(e) => setKey(p.key, e.target.value)}
                autoComplete="off"
              />
            </div>
          ))}
          <label className="remember">
            <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
            记住 Key（仅保存在本机浏览器，清除浏览器数据即失效）
          </label>
          {health && (
            <p className="caps">
              能力：ffmpeg {health.ffmpeg ? '✓' : '✗'} · whisper 转写 {health.whisper ? '✓' : '✗（可换有字幕的视频）'}
            </p>
          )}
        </div>
      )}
    </section>
  )
}
