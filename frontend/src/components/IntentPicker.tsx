import type { IntentName } from '../types'

const OPTIONS: { value: IntentName; label: string; desc: string }[] = [
  { value: 'auto', label: '自动判断', desc: '根据提问自动选择（推荐）' },
  { value: 'SUMMARY', label: '理解摘要', desc: '结构化总结视频内容' },
  { value: 'LOCATE', label: '定位', desc: '目标最早出现的精确时间' },
  { value: 'MODERATE', label: '审核', desc: '高召回排查敏感内容' },
  { value: 'VISUAL_LOCATE', label: '以图搜', desc: '用参考图在视频里找' },
  { value: 'PLATE', label: '车牌/车辆', desc: '追踪车辆并高亮跟随，框内识别车牌' },
]

export function IntentPicker({ intent, setIntent }: { intent: IntentName; setIntent: (i: IntentName) => void }) {
  return (
    <section className="card">
      <label className="card-title">意图</label>
      <div className="intent-grid">
        {OPTIONS.map((o) => (
          <button
            key={o.value}
            className={`intent-chip ${intent === o.value ? 'active' : ''}`}
            onClick={() => setIntent(o.value)}
          >
            <b>{o.label}</b>
            <small>{o.desc}</small>
          </button>
        ))}
      </div>
    </section>
  )
}
