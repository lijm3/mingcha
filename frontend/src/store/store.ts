import { create } from 'zustand'
import type { AnswerOut, IntentName, ProviderKeys, TaskStatus } from '../types'
import { cancelTask, createTask, getAnswer, TaskInput } from '../api/client'

const LS_KEY = 'mingcha_keys'

function loadKeys(): ProviderKeys {
  try {
    return { claude: '', openai: '', glm: '', ...JSON.parse(localStorage.getItem(LS_KEY) || '{}') }
  } catch {
    return { claude: '', openai: '', glm: '' }
  }
}

type Phase = 'idle' | 'uploading' | 'running' | 'done' | 'error'

interface AppState {
  keys: ProviderKeys
  remember: boolean
  setKey: (p: keyof ProviderKeys, v: string) => void
  setRemember: (v: boolean) => void

  phase: Phase
  task: { id: string; token: string } | null
  uploadProgress: number | null // 0~1（上传段），null=非上传态
  status: TaskStatus | null
  logs: string[]
  disconnected: boolean
  answer: AnswerOut | null
  errorMsg: string | null

  startTask: (input: TaskInput) => Promise<void>
  onStatus: (s: Partial<TaskStatus>) => void
  onLog: (line: string) => void
  onDisconnected: (v: boolean) => void
  onDone: () => Promise<void>
  onFail: (msg: string) => void
  cancel: () => Promise<void>
  reset: () => void
}

export const useStore = create<AppState>((set, get) => ({
  keys: loadKeys(),
  remember: !!localStorage.getItem(LS_KEY),
  setKey: (p, v) =>
    set((st) => {
      const keys = { ...st.keys, [p]: v }
      if (st.remember) localStorage.setItem(LS_KEY, JSON.stringify(keys))
      return { keys }
    }),
  setRemember: (v) =>
    set((st) => {
      if (v) localStorage.setItem(LS_KEY, JSON.stringify(st.keys))
      else localStorage.removeItem(LS_KEY)
      return { remember: v }
    }),

  phase: 'idle',
  task: null,
  uploadProgress: null,
  status: null,
  logs: [],
  disconnected: false,
  answer: null,
  errorMsg: null,

  startTask: async (input) => {
    set({
      phase: input.file ? 'uploading' : 'running',
      uploadProgress: input.file ? 0 : null,
      status: null, answer: null, errorMsg: null, logs: [], disconnected: false, task: null,
    })
    try {
      const resp = await createTask(input, get().keys, (f) => set({ uploadProgress: f }))
      set({
        task: { id: resp.task_id, token: resp.task_token },
        phase: 'running', uploadProgress: null,
      })
    } catch (e: any) {
      set({ phase: 'error', errorMsg: e?.message || '创建任务失败' })
    }
  },

  onStatus: (s) =>
    set((st) => ({
      status: { ...(st.status || ({} as TaskStatus)), ...s } as TaskStatus,
      disconnected: false,
    })),
  onLog: (line) => set((st) => ({ logs: [...st.logs, line].slice(-100) })),
  onDisconnected: (v) => set({ disconnected: v }),

  onDone: async () => {
    const t = get().task
    if (!t) return
    try {
      const ans = await getAnswer(t.id, t.token)
      set({ answer: ans, phase: 'done' })
    } catch (e: any) {
      set({ phase: 'error', errorMsg: e?.message || '获取结果失败' })
    }
  },
  onFail: (msg) => set({ phase: 'error', errorMsg: msg }),

  cancel: async () => {
    const t = get().task
    if (t) await cancelTask(t.id, t.token)
  },

  reset: () =>
    set({
      phase: 'idle', task: null, uploadProgress: null, status: null,
      logs: [], disconnected: false, answer: null, errorMsg: null,
    }),
}))

export type { IntentName }
