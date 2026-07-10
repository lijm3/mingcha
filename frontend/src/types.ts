// 前端类型：与后端 backend/app/schemas.py 对齐。
// 生产建议用 `npm run gen:types` 从 /openapi.json 自动生成到 src/types/api.ts（§8.9），
// 这里手写一份，保证未生成时也能开发/构建。

export type TaskState =
  | 'queued' | 'downloading' | 'extracting' | 'transcribing'
  | 'analyzing' | 'assembling' | 'done' | 'error' | 'cancelled'

export type IntentName = 'auto' | 'SUMMARY' | 'LOCATE' | 'MODERATE' | 'VISUAL_LOCATE'

export interface CreateTaskResponse {
  task_id: string
  task_token: string
}

export interface TaskStatus {
  task_id: string
  state: TaskState
  progress: number
  stage_note: string
  intent: string | null
  error: string | null
  created_at: number
  caveats: string
}

export interface EvidenceOut {
  frame: string
  frame_url: string
  t: number
  hms: string
  confidence: number
  similarity?: number | null
  verdict?: string | null
  note: string
}

export interface SummaryDetail {
  topic: string
  segments: string[]
  key_points: string[]
}

export interface AnswerOut {
  intent: string
  target: string | null
  answer: string
  summary_detail: SummaryDetail | null
  evidence: EvidenceOut[]
  confidence: number
  caveats: string
  video_url: string | null
  query_image_url: string | null
  grids: string[]
}

export interface Health {
  status: string
  ffmpeg: boolean
  whisper: boolean
  version: string
}

export type ProviderKeys = Record<'claude' | 'openai' | 'glm', string>

// SSE 事件（§5.4 / §8.6）
export type SSEEvent =
  | { type: 'state'; state?: TaskState; progress?: number; stage_note?: string; intent?: string | null }
  | { type: 'log'; line: string }
  | { type: 'done'; answer_url?: string }
  | { type: 'fail'; error: string; state?: string }
  | { type: 'disconnected' }
