import { useEffect, useId, useState } from 'react'
import { queryTracks, type TrackRow } from '../lib/trackProtocol'

/** Search box for picking one track (local full-data mode only). */
export function TrackSearch({ label, onPick }: { label: string; onPick: (t: TrackRow) => void }) {
  const id = useId()
  const [text, setText] = useState('')
  const [rows, setRows] = useState<TrackRow[]>([])
  const [total, setTotal] = useState(0)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const q = text.trim()
    if (q.length < 2) {
      setRows([])
      setTotal(0)
      return
    }
    let live = true
    setBusy(true)
    const t = setTimeout(() => {
      queryTracks(
        { text: q, genres: [], popMin: 0, popMax: 100, features: {}, sort: { key: 'popularity', dir: 'desc' }, page: 0, pageSize: 8 },
        { x: 'energy', y: 'danceability' },
      ).then((r) => {
        if (!live) return
        setRows(r.rows)
        setTotal(r.total)
        setBusy(false)
      })
    }, 150)
    return () => {
      live = false
      clearTimeout(t)
    }
  }, [text])

  return (
    <div className="field" style={{ position: 'relative', minWidth: 260, flex: 1 }}>
      <label htmlFor={id}>{label}</label>
      <input id={id} type="search" value={text} placeholder="Title, artist or album" onChange={(e) => setText(e.target.value)} autoComplete="off" aria-describedby={`${id}-status`} />
      <span id={`${id}-status`} className="small muted" aria-live="polite">
        {text.trim().length >= 2 ? (busy ? 'Searching…' : total ? `${total.toLocaleString('en-US')} matches, most popular first` : 'No tracks match') : 'Type at least 2 characters'}
      </span>
      {rows.length > 0 && (
        <ul className="result-list panel" style={{ padding: '0 12px', marginTop: 4 }} aria-label="Matching tracks">
          {rows.map((r) => (
            <li key={r.index} style={{ padding: '6px 0' }}>
              <button
                type="button"
                className="link"
                onClick={() => {
                  onPick(r)
                  setText('')
                }}
              >
                {r.name || 'Untitled'}
              </button>
              <div className="cell-sub">
                {r.artists} · {r.album} · popularity {r.popularity}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
