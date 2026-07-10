import { useState } from 'react'
import { useStore } from './store/store'
import { KeyPanel } from './components/KeyPanel'
import { SourceInput } from './components/SourceInput'
import { IntentPicker } from './components/IntentPicker'
import { ProgressView } from './components/ProgressView'
import { ResultView } from './components/ResultView'
import type { IntentName } from './types'

export function App() {
  const { phase, startTask, reset } = useStore()

  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [prompt, setPrompt] = useState('')
  const [intent, setIntent] = useState<IntentName>('auto')

  const canStart = (!!file || !!url.trim()) && (!!prompt.trim() || !!image)

  const onStart = () => {
    startTask({ file, url: url.trim(), image, prompt: prompt.trim(), intent })
  }

  const busy = phase === 'uploading' || phase === 'running'

  return (
    <div className="app">
      <header className="hero">
        <h1>明察 <span className="sub">MingCha</span></h1>
        <p className="tagline">看懂 · 看准 · 看住视频 —— 多模型视频分析智能体</p>
      </header>

      {phase === 'idle' || phase === 'error' ? (
        <div className="panel-stack">
          <KeyPanel />
          <SourceInput
            file={file} setFile={setFile} url={url} setUrl={setUrl}
            image={image} setImage={setImage}
          />
          <IntentPicker intent={intent} setIntent={setIntent} />
          <section className="card">
            <label className="card-title">提问</label>
            <textarea
              className="prompt"
              placeholder="例如：总结这个视频讲了什么 / 红色的车最早出现在什么时间 / 有没有暴力内容"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
            />
            {image && <p className="hint">已附参考图 → 将以「以图搜视频」模式运行（VISUAL_LOCATE）。</p>}
          </section>

          {phase === 'error' && <ErrorBar />}

          <button className="start-btn" disabled={!canStart} onClick={onStart}>
            开始分析
          </button>
          {!canStart && (
            <p className="hint center">请先选择视频（上传或粘贴 URL），并填写提问或附参考图。</p>
          )}
        </div>
      ) : (
        <div className="panel-stack">
          {busy && <ProgressView />}
          {phase === 'done' && (
            <>
              <ResultView />
              <button className="start-btn ghost" onClick={reset}>← 分析新的视频</button>
            </>
          )}
        </div>
      )}

      <footer className="foot">
        本地运行 · Key 仅存本机、不落盘 · 关键帧将发送到你所填的模型服务商分析
      </footer>
    </div>
  )
}

function ErrorBar() {
  const { errorMsg, reset } = useStore()
  return (
    <div className="error-bar">
      <span>✗ {errorMsg}</span>
      <button onClick={reset}>重试</button>
    </div>
  )
}
