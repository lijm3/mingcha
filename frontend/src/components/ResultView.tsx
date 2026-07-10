import { useRef } from 'react'
import { useStore } from '../store/store'
import type { AnswerOut, EvidenceOut } from '../types'

export function ResultView() {
  const { answer } = useStore()
  const videoRef = useRef<HTMLVideoElement>(null)

  if (!answer) return null

  const seek = (t: number) => {
    const v = videoRef.current
    if (v) { v.currentTime = t; v.play().catch(() => {}) }
  }

  return (
    <section className="card result">
      <div className="result-head">
        <span className="intent-badge">{answer.intent}</span>
        {answer.confidence > 0 && (
          <span className="conf-badge">置信度 {(answer.confidence * 100).toFixed(0)}%</span>
        )}
      </div>

      {answer.query_image_url && (
        <div className="query-echo">
          <img src={answer.query_image_url} alt="参考图" />
          <span>参考图</span>
        </div>
      )}

      {/* SUMMARY 三段式；无结构化时优雅降级为整段 answer */}
      {answer.summary_detail ? (
        <SummaryCard detail={answer.summary_detail} fallback={answer.answer} />
      ) : (
        <p className="answer-text">{answer.answer}</p>
      )}

      {answer.video_url && (
        <video ref={videoRef} className="player" src={answer.video_url} controls preload="metadata" />
      )}

      {answer.evidence.length > 0 && (
        <EvidenceList evidence={answer.evidence} onSeek={seek} />
      )}

      {answer.caveats && (
        <div className="caveat-banner">
          <b>⚠ 采样局限 / 复核声明</b>
          <p>{answer.caveats}</p>
        </div>
      )}

      <DownloadJson answer={answer} />
    </section>
  )
}

function SummaryCard({ detail, fallback }: { detail: AnswerOut['summary_detail']; fallback: string }) {
  if (!detail) return <p className="answer-text">{fallback}</p>
  return (
    <div className="summary-card">
      {detail.topic && <h3 className="topic">{detail.topic}</h3>}
      {detail.segments.length > 0 && (
        <div className="sum-block">
          <h4>分段脉络</h4>
          <ol>{detail.segments.map((s, i) => <li key={i}>{s}</li>)}</ol>
        </div>
      )}
      {detail.key_points.length > 0 && (
        <div className="sum-block">
          <h4>关键要点</h4>
          <ul>{detail.key_points.map((k, i) => <li key={i}>{k}</li>)}</ul>
        </div>
      )}
      {!detail.topic && !detail.segments.length && !detail.key_points.length && (
        <p className="answer-text">{fallback}</p>
      )}
    </div>
  )
}

function EvidenceList({ evidence, onSeek }: { evidence: EvidenceOut[]; onSeek: (t: number) => void }) {
  return (
    <div className="evidence-list">
      <h4>证据（点击跳转到对应时间）</h4>
      <div className="evidence-grid">
        {evidence.map((ev, i) => (
          <div className="evidence-card" key={i} onClick={() => onSeek(ev.t)}>
            <img src={ev.frame_url} loading="lazy" alt={ev.hms} />
            <div className="ev-meta">
              <span className="badge">{ev.hms}</span>
              <span className="conf">置信 {(ev.confidence * 100).toFixed(0)}%</span>
              {ev.similarity != null && <span className="sim">相似 {(ev.similarity * 100).toFixed(0)}%</span>}
              {ev.verdict && (
                <span className={`verdict ${ev.verdict}`}>
                  {ev.verdict === 'same' ? '同一个体' : '同类外观'}
                </span>
              )}
            </div>
            {ev.note && <p className="note">{ev.note}</p>}
          </div>
        ))}
      </div>
    </div>
  )
}

function DownloadJson({ answer }: { answer: AnswerOut }) {
  const download = () => {
    const blob = new Blob([JSON.stringify(answer, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'answer.json'
    a.click()
  }
  return <button className="dl-btn" onClick={download}>下载 answer.json</button>
}
