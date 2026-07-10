import type { AnswerOut, CreateTaskResponse, Health, IntentName, ProviderKeys, TaskStatus } from '../types'

// Key 只走请求头 X-Provider-Keys（§7.1），绝不进 URL/query。
function keyHeader(keys: ProviderKeys): string {
  const filtered: Record<string, string> = {}
  for (const [k, v] of Object.entries(keys)) if (v?.trim()) filtered[k] = v.trim()
  return JSON.stringify(filtered)
}

export async function getHealth(): Promise<Health> {
  const r = await fetch('/api/health')
  return r.json()
}

export interface TaskInput {
  file?: File | null
  url?: string
  image?: File | null
  prompt: string
  intent: IntentName
  provider?: string
  vision_model?: string
}

// 大文件上传用 XHR 拿上传进度（§8.8）；URL 源用 fetch+JSON。
export function createTask(
  input: TaskInput, keys: ProviderKeys, onUpload?: (frac: number) => void,
): Promise<CreateTaskResponse> {
  if (input.file) {
    const form = new FormData()
    form.append('video', input.file)
    if (input.image) form.append('image', input.image)
    form.append('prompt', input.prompt)
    form.append('intent', input.intent)
    if (input.provider) form.append('provider', input.provider)
    if (input.vision_model) form.append('vision_model', input.vision_model)
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', '/api/tasks')
      xhr.setRequestHeader('X-Provider-Keys', keyHeader(keys))
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onUpload) onUpload(e.loaded / e.total)
      }
      xhr.onload = () =>
        xhr.status === 202
          ? resolve(JSON.parse(xhr.responseText))
          : reject(new Error(safeErr(xhr.responseText, xhr.status)))
      xhr.onerror = () => reject(new Error('网络错误，上传失败'))
      xhr.send(form)
    })
  }
  // URL 源
  return fetch('/api/tasks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Provider-Keys': keyHeader(keys) },
    body: JSON.stringify({
      url: input.url, prompt: input.prompt, intent: input.intent,
      override: { provider: input.provider || null, vision_model: input.vision_model || null },
    }),
  }).then(async (r) => {
    if (r.status !== 202) throw new Error(safeErr(await r.text(), r.status))
    return r.json()
  })
}

export async function getStatus(id: string, token: string): Promise<TaskStatus> {
  const r = await fetch(`/api/tasks/${id}?token=${token}`)
  return r.json()
}

export async function getAnswer(id: string, token: string): Promise<AnswerOut> {
  const r = await fetch(`/api/tasks/${id}/answer?token=${token}`)
  if (!r.ok) throw new Error(`获取结果失败: ${r.status}`)
  return r.json()
}

export async function cancelTask(id: string, token: string): Promise<void> {
  await fetch(`/api/tasks/${id}/cancel?token=${token}`, { method: 'POST' })
}

function safeErr(text: string, status: number): string {
  try {
    const d = JSON.parse(text)
    return d.detail || `请求失败 (${status})`
  } catch {
    return `请求失败 (${status})`
  }
}
