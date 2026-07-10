import { useEffect } from 'react'
import type { SSEEvent } from '../types'

// SSE 封装（§8.6）：
// 关键——原生 'error' = 连接层故障（无 data），交给浏览器自动重连，不 close；
// 业务失败走服务端自定义的 'fail' 事件（§5.4），收到才 close。
export function useSSE(
  taskId: string | null,
  token: string | null,
  onEvent: (e: SSEEvent) => void,
) {
  useEffect(() => {
    if (!taskId || !token) return
    const es = new EventSource(`/api/tasks/${taskId}/events?token=${token}`)
    const parse = (e: MessageEvent) => {
      try {
        return JSON.parse(e.data)
      } catch {
        return {}
      }
    }
    es.addEventListener('state', (e) => onEvent({ type: 'state', ...parse(e as MessageEvent) }))
    es.addEventListener('log', (e) => onEvent({ type: 'log', ...parse(e as MessageEvent) }))
    es.addEventListener('done', (e) => {
      onEvent({ type: 'done', ...parse(e as MessageEvent) })
      es.close()
    })
    es.addEventListener('fail', (e) => {
      onEvent({ type: 'fail', ...parse(e as MessageEvent) })
      es.close()
    })
    // 原生连接错误：e.data 为 undefined。不 close —— 浏览器带 Last-Event-ID 自动重连，
    // 仅把 UI 切到“连接中断，重连中…”。只有 'fail'/'done' 才真正结束流。
    es.onerror = () => onEvent({ type: 'disconnected' })
    return () => es.close()
  }, [taskId, token])
}
