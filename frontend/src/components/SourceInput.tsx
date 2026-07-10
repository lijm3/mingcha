import { useState } from 'react'

interface Props {
  file: File | null
  setFile: (f: File | null) => void
  url: string
  setUrl: (u: string) => void
  image: File | null
  setImage: (f: File | null) => void
}

export function SourceInput({ file, setFile, url, setUrl, image, setImage }: Props) {
  const [tab, setTab] = useState<'upload' | 'url'>('upload')
  const [drag, setDrag] = useState(false)

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files?.[0]
    if (f) setFile(f)
  }

  return (
    <section className="card">
      <div className="tabs">
        <button className={tab === 'upload' ? 'active' : ''} onClick={() => setTab('upload')}>
          上传视频
        </button>
        <button className={tab === 'url' ? 'active' : ''} onClick={() => setTab('url')}>
          粘贴链接
        </button>
      </div>

      {tab === 'upload' ? (
        <div
          className={`dropzone ${drag ? 'drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
        >
          <input
            id="video-file"
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            hidden
          />
          <label htmlFor="video-file">
            {file ? (
              <span className="filename">🎬 {file.name}（{(file.size / 1e6).toFixed(1)} MB）</span>
            ) : (
              <span>拖拽视频到此，或<u>点击选择文件</u></span>
            )}
          </label>
          {file && <button className="clear" onClick={() => setFile(null)}>清除</button>}
        </div>
      ) : (
        <input
          className="url-input"
          type="url"
          placeholder="https://... （支持 yt-dlp 可解析的链接）"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
      )}

      {/* 参考图（VISUAL_LOCATE 以图搜视频）*/}
      <div className="ref-image">
        <input
          id="ref-image"
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files?.[0] || null)}
          hidden
        />
        {image ? (
          <div className="ref-preview">
            <img src={URL.createObjectURL(image)} alt="参考图" />
            <span>参考图：{image.name}</span>
            <button className="clear" onClick={() => setImage(null)}>移除</button>
          </div>
        ) : (
          <label htmlFor="ref-image" className="ref-add">＋ 附参考图（可选，用于「以图搜视频」）</label>
        )}
      </div>
    </section>
  )
}
