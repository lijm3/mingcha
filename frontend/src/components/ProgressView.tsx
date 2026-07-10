import { useStore } from '../store/store'
import { useSSE } from '../api/sse'

const STAGE_LABEL: Record<string, string> = {
  queued: '排队中', downloading: '下载视频', extracting: '抽帧去重',
  transcribing: '语音转写', analyzing: '模型分析', assembling: '组装结果',
  done: '完成',
}

export function ProgressView() {
  const {
    phase, task, uploadProgress, status, logs, disconnected,
    onStatus, onLog, onDisconnected, onDone, onFail, cancel,
  } = useStore()

  // 订阅 SSE（task 就绪后）。上传段无 task，仅显示上传进度。
  useSSE(task?.id ?? null, task?.token ?? null, (e) => {
    if (e.type === 'state') { onStatus(e as any); onDisconnected(false) }
    else if (e.type === 'log') onLog(e.line)
    else if (e.type === 'done') onDone()
    else if (e.type === 'fail') onFail(e.error || '任务失败')
    else if (e.type === 'disconnected') onDisconnected(true)
  })

  // 两段式进度（§8.8）：上传段用 uploadProgress，分析段用 status.progress。
  const uploading = phase === 'uploading' && uploadProgress != null
  const frac = uploading ? uploadProgress! : status?.progress ?? 0
  const pct = Math.round(frac * 100)
  const stage = uploading ? '上传视频' : STAGE_LABEL[status?.state ?? 'queued'] ?? status?.state

  return (
    <section className="card progress">
      <div className="progress-head">
        <span className="stage">{stage}…</span>
        <span className="pct">{pct}%</span>
      </div>
      <div className="bar"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>

      {status?.stage_note && <p className="stage-note">{status.stage_note}</p>}
      {status?.intent && <p className="stage-note">意图：{status.intent}</p>}

      {disconnected && <p className="reconnect">连接中断，重连中…</p>}

      {logs.length > 0 && (
        <details className="logs">
          <summary>日志（{logs.length}）</summary>
          <pre>{logs.join('\n')}</pre>
        </details>
      )}

      <button className="cancel-btn" onClick={cancel}>取消任务</button>
    </section>
  )
}
