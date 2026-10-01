import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Comparison, Episode, InternalPrefix, ReadEvent, TraceFolder, WriteEvent } from './trace';
import { discoverTraces, loadComparisons, loadEpisode, referenceCondition, splitEpisodeKey } from './trace';
import type { LayerCalibration } from './calibration';
import { calibrateLayer, intensity, percentile } from './calibration';
import { TraceSidebar, conditionLabel } from './TraceSidebar';
import { InternalToken } from './InternalToken';

type Mode = 'writes' | 'reads';
type Selection = { kind: 'text'; position: number } | { kind: 'internal'; segment: number; index: number };

const number = (value: number | null | undefined, digits = 4) => value == null ? 'Not measured' : Number(value).toPrecision(digits);

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

function eventColor(value: number, p99: number, mode: Mode): CSSProperties {
  return { '--strength': intensity(value, p99), '--hue': mode === 'writes' ? '0 72% 53%' : '270 70% 58%' } as CSSProperties;
}

function App() {
  const [runs, setRuns] = useState<TraceFolder[]>([]);
  const [runId, setRunId] = useState('');
  const [discoveryErrors, setDiscoveryErrors] = useState<string[]>([]);
  const folder = runs.find(run => run.id === runId) ?? runs[0] ?? null;
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [key, setKey] = useState('');
  const [mode, setMode] = useState<Mode>('writes');
  const [layer, setLayer] = useState(0);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [expandedPrefixes, setExpandedPrefixes] = useState<Set<string>>(new Set());
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [condition, setCondition] = useState('');
  const [calibration, setCalibration] = useState<LayerCalibration | null>(null);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function scan() {
      try {
        const index = await discoverTraces();
        if (!cancelled) {
          setRuns(previous => JSON.stringify(previous) === JSON.stringify(index.runs) ? previous : index.runs.map(run => previous.find(old => old.id === run.id && old.revision === run.revision) ?? run));
          setDiscoveryErrors(index.errors);
        }
      } catch (cause) { if (!cancelled) setDiscoveryErrors([String(cause)]); }
      if (!cancelled) timer = setTimeout(scan, 3000);
    }
    void scan();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  const keys = useMemo(() => Object.keys(folder?.manifest.episodes ?? {}).sort((a, b) => a.localeCompare(b, undefined, { numeric: true })), [folder]);
  const currentCondition = key ? splitEpisodeKey(key).condition : '';
  const layers = useMemo(() => [...new Set([...(episode?.writes ?? []).map(item => item.layer), ...(episode?.reads ?? []).map(item => item.layer)])].sort((a, b) => a - b), [episode]);

  useEffect(() => {
    if (!folder) return;
    const preferred = referenceCondition(folder.manifest);
    setCondition(preferred);
    setKey(previous => folder.manifest.episodes[previous] ? previous : keys.find(entry => splitEpisodeKey(entry).condition === preferred) ?? keys[0]);
    setEpisode(null);
    setLayer(0);
    setSelection(null);
    setComparisons([]);
    setCalibration(null);
    setError('');
    setExpandedPrefixes(new Set());
  }, [folder?.id, folder?.revision]);

  useEffect(() => {
    if (!folder || !key || !folder.manifest.episodes[key]) return;
    let cancelled = false;
    setBusy(true);
    setEpisode(null);
    setComparisons([]);
    setExpandedPrefixes(new Set());
    setError('');
    loadEpisode(folder, key).then(loaded => {
      if (cancelled) return;
      setEpisode(loaded);
      const query = loaded.metadata.query_start_position;
      setSelection({ kind: 'text', position: typeof query === 'number' ? query : loaded.tokens.at(-1)?.position ?? 0 });
      if (![...new Set(loaded.writes.map(item => item.layer))].includes(layer)) setLayer(loaded.writes[0]?.layer ?? 0);
    }).catch(cause => { if (!cancelled) setError(String(cause)); })
      .finally(() => { if (!cancelled) setBusy(false); });
    const exampleId = splitEpisodeKey(key).exampleId;
    loadComparisons(folder, exampleId).then(loaded => { if (!cancelled) setComparisons(loaded); }).catch(cause => { if (!cancelled) setError(String(cause)); });
    return () => { cancelled = true; };
  }, [folder, key]);

  useEffect(() => {
    if (!folder || !condition || !episode) return;
    let cancelled = false;
    setCalibration(null);
    calibrateLayer(folder, layer, condition, episode.checkpoint)
      .then(result => { if (!cancelled) setCalibration(result); })
      .catch(cause => { if (!cancelled) setError(String(cause)); });
    return () => { cancelled = true; };
  }, [folder, layer, condition, episode?.checkpoint]);

  const writeAt = useMemo(() => new Map((episode?.writes ?? []).filter(item => item.layer === layer && item.position !== null).map(item => [item.position!, item])), [episode, layer]);
  const readAt = useMemo(() => new Map((episode?.reads ?? []).filter(item => item.layer === layer).map(item => [item.position, item])), [episode, layer]);
  const prefixesAt = useMemo(() => {
    const map = new Map<number, InternalPrefix[]>();
    for (const prefix of episode?.internal_prefixes ?? []) {
      if (prefix.layer === layer) map.set(prefix.before_position, [...(map.get(prefix.before_position) ?? []), prefix]);
    }
    return map;
  }, [episode, layer]);

  function selectTrace(run: string, trace: string) {
    setRunId(run);
    setKey(trace);
  }

  function jumpToSearch() {
    if (!episode || !search) return;
    const full = episode.tokens.map(token => token.text).join('');
    const found = full.toLocaleLowerCase().indexOf(search.toLocaleLowerCase());
    if (found < 0) { setError(`“${search}” was not found in this episode.`); return; }
    let cursor = 0;
    for (const token of episode.tokens) {
      cursor += token.text.length;
      if (found < cursor) {
        setSelection({ kind: 'text', position: token.position });
        document.getElementById(`token-${token.position}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
        setError('');
        return;
      }
    }
  }

  function togglePrefix(prefix: InternalPrefix) {
    const id = `${prefix.layer}:${prefix.segment_index}:${prefix.before_position}`;
    setExpandedPrefixes(previous => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  const selectedWrite: WriteEvent | undefined = selection?.kind === 'text' ? writeAt.get(selection.position) : episode?.writes.find(item => item.layer === layer && item.position === null && item.segment_index === selection?.segment && item.internal_index === selection.index);
  const selectedRead: ReadEvent | undefined = selection?.kind === 'text' ? readAt.get(selection.position) : undefined;
  const selectedToken = selection?.kind === 'text' ? episode?.tokens.find(token => token.position === selection.position) : undefined;
  const relevantComparisons = selection?.kind === 'text' ? comparisons.filter(item => item.scored_position === selection.position && (item.baseline_condition === currentCondition || item.variant_condition === currentCondition)) : [];
  const refValues = mode === 'writes' ? calibration?.writes : calibration?.reads;

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">M</span><div><strong>TaaL Memory Viewer</strong><small>Offline trace inspection</small></div></div>
      <TraceSidebar runs={runs} selectedRun={folder?.id ?? ''} selectedKey={key} onSelect={selectTrace} />
    </aside>

    <main className="main">
      <header className="topbar"><div><span className="eyebrow">TRAJECTORY INSPECTION</span><h1>{episode ? episode.example_id : 'Inspect a memory trace'}</h1><p>{episode ? `${conditionLabel(episode.condition_id)} · ${episode.tokens.length.toLocaleString()} visible tokens` : 'A token-level view of online writes and injected reads.'}</p></div><div className="topbar-meta">{episode?.checkpoint && <><span>CHECKPOINT</span><strong title={episode.checkpoint}>{episode.checkpoint.split('/').at(-1)}</strong></>}</div></header>
      {error && <div role="alert" className="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
      {discoveryErrors.length > 0 && <div role="alert" className="alert"><div><strong>Exports not loaded</strong>{discoveryErrors.map(message => <p key={message}>{message}</p>)}</div></div>}
      {!episode ? <section className="empty"><h2>{busy ? 'Reading episode…' : 'Checking local traces…'}</h2></section> : <>
        <section className="viewbar"><div className="view-controls"><div className="segmented"><button className={mode === 'writes' ? 'active' : ''} onClick={() => setMode('writes')}>Memory writes</button><button className={mode === 'reads' ? 'active' : ''} onClick={() => setMode('reads')}>Memory reads</button></div><label className="layer-control">Layer<select aria-label="Memory layer" value={layer} onChange={event => setLayer(Number(event.target.value))}>{layers.map(item => <option key={item} value={item}>{item}</option>)}</select></label></div><div className="search"><input aria-label="Find text" value={search} onChange={event => setSearch(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') jumpToSearch(); }} placeholder="Find text in episode" /><button onClick={jumpToSearch}>Find</button></div></section>
        <div className="content-grid">
          <section className="trajectory-panel"><div className="panel-heading"><div><span className="eyebrow">SEQUENCE</span><h2>Token trajectory</h2></div><span>Click a token to inspect its signal</span></div><div className="trajectory" aria-label="Token trajectory" aria-busy={!calibration}>
            {episode.tokens.map(token => {
              const write = writeAt.get(token.position);
              const read = readAt.get(token.position);
              let className = 'token';
              let strength = 0;
              if (mode === 'writes') {
                if (!write) className += ' missing';
                else if (!write.write_enabled || write.proposed_write_norm === 0) className += ' neutral';
                else if (calibration) strength = intensity(write.proposed_write_norm, calibration.writeP99);
              } else {
                if (!read) className += ' missing';
                else if (!read.enabled || !read.relative_injection) className += ' neutral';
                else if (calibration) strength = intensity(read.relative_injection, calibration.readP99);
              }
              if (selection?.kind === 'text' && selection.position === token.position) className += ' selected';
              return <span key={token.position} className="token-slot">
                {(prefixesAt.get(token.position) ?? []).map(prefix => {
                  const id = `${prefix.layer}:${prefix.segment_index}:${prefix.before_position}`;
                  const expanded = expandedPrefixes.has(id);
                  const internals = episode.writes.filter(item => item.layer === layer && item.position === null && item.segment_index === prefix.segment_index).sort((a, b) => (a.internal_index ?? 0) - (b.internal_index ?? 0));
                  return <span key={id} className="prefix-wrap"><button className="prefix-marker" onClick={() => togglePrefix(prefix)} aria-expanded={expanded}>{prefix.count} internal memory tokens <b>{expanded ? '▴' : '▾'}</b></button>{expanded && <span className="prefix-expanded">{Array.from({ length: prefix.count }, (_, index) => <InternalToken key={index} index={index} write={internals.find(item => item.internal_index === index)} mode={mode} writeP99={calibration?.writeP99 ?? null} selected={selection?.kind === 'internal' && selection.segment === prefix.segment_index && selection.index === index} onSelect={() => setSelection({ kind: 'internal', segment: prefix.segment_index, index })} />)}</span>}</span>;
                })}
                <button id={`token-${token.position}`} title={`Token ${token.position} · id ${token.token_id} · ${JSON.stringify(token.text)}`} aria-label={`Token ${token.position}: ${JSON.stringify(token.text)}`} className={className} style={eventColor(strength, 1, mode)} onClick={() => setSelection({ kind: 'text', position: token.position })}><span>{token.text.trim() || (token.text.includes('\n') ? '↵' : token.text.includes('\t') ? '⇥' : token.text ? '␠' : `⟨${token.token_id}⟩`)}</span></button>
              </span>;
            })}
          </div></section>
          <aside className="details-panel"><div className="panel-heading"><div><span className="eyebrow">INSPECTOR</span><h2>{selection?.kind === 'internal' ? `Internal token P${selection.index + 1}` : selectedToken ? `Token ${selectedToken.position}` : 'Select a token'}</h2></div></div>
            {selection && <>
              {selectedToken && <div className="selected-text"><span>TEXT FRAGMENT</span><strong>{JSON.stringify(selectedToken.text)}</strong><small>ID {selectedToken.token_id} · {selectedToken.phase}</small></div>}
              {mode === 'writes' ? selectedWrite ? <div className="metric-grid">{(selectedWrite.chunk_size ?? 1) > 1 && <p className="inspector-note">This token contributes to a {selectedWrite.chunk_size}-token gradient. Weights change only at the chunk boundary; the boundary's net change belongs to the whole chunk.</p>}<Metric label="NEW WRITE · ||−gₜ||" value={selectedWrite.write_enabled ? number(selectedWrite.proposed_write_norm) : 'Masked'} note="Includes learned write strength" /><Metric label="REFERENCE PERCENTILE" value={selectedWrite.write_enabled && calibration ? `${number(percentile(refValues ?? [], selectedWrite.proposed_write_norm), 3)}%` : 'Unavailable'} note="Rank among reference writes, not usefulness" /><Metric label="PREVIOUS ||W||" value={number(selectedWrite.previous_weight_norm)} /><Metric label="WRITE / PREVIOUS" value={number(selectedWrite.previous_weight_norm ? selectedWrite.proposed_write_norm / selectedWrite.previous_weight_norm : null)} /><Metric label={selectedWrite.chunk_boundary && (selectedWrite.chunk_size ?? 1) > 1 ? 'CHUNK NET CHANGE · ||ΔW||' : 'NET CHANGE · ||ΔWₜ||'} value={number(selectedWrite.net_weight_change_norm)} note={selectedWrite.chunk_boundary ? 'Includes momentum and forgetting' : 'No weight update before this chunk boundary'} /><Metric label="OTHER MOVEMENT · ||ΔWₜ + gₜ||" value={number(selectedWrite.other_movement_norm)} /><Metric label="WRITE ↔ NET ALIGNMENT" value={number(selectedWrite.write_to_net_alignment)} note="Unavailable for chunk-level updates" /><Metric label="WRITE STRENGTH θₜ" value={number(selectedWrite.write_strength)} /></div> : <p className="unavailable">This token has no write measurement at the selected layer.</p>
                : selection.kind === 'internal' ? <div className="metric-grid"><Metric label="READ STATUS" value="Read output discarded" note="Not injected into the transformer residual stream" /></div>
                : selectedRead ? <div className="metric-grid"><Metric label="READ STATUS" value={selectedRead.enabled ? 'Enabled' : 'Disabled'} note={selectedRead.state_timing === 'post-write' ? 'Read uses the newly committed state' : 'Read uses the previous completed chunk state'} /><Metric label="INJECTED ||δ||" value={number(selectedRead.injected_norm)} note="After gate and read scale" /><Metric label="INCOMING ||h||" value={number(selectedRead.incoming_norm)} /><Metric label="RELATIVE INJECTION" value={number(selectedRead.relative_injection)} /><Metric label="REFERENCE PERCENTILE" value={selectedRead.enabled && calibration && selectedRead.relative_injection != null ? `${number(percentile(refValues ?? [], selectedRead.relative_injection), 3)}%` : 'Unavailable'} /><Metric label="READ SCALE" value={number(selectedRead.read_scale)} /><Metric label="RESIDUAL GATE" value={number(selectedRead.residual_gate)} /></div> : <p className="unavailable">This token has no read measurement at the selected layer.</p>}
              {selection.kind === 'internal' && <p className="inspector-note">This is a learned memory-only token. Its read output is discarded; its write can still change the fast-weight state.</p>}
              {selection.kind === 'text' && (prefixesAt.get(selection.position) ?? []).map(prefix => <div key={prefix.segment_index} className="prefix-note">Internal prefix before this token: {prefix.count} tokens · group net state change {number(prefix.net_weight_change_norm)}</div>)}
              {mode === 'reads' && relevantComparisons.map(item => <div className="comparison" key={`${item.baseline_condition}/${item.variant_condition}`}><span className="eyebrow">PAIRED PREDICTION · WHOLE QUERY</span><h3>{item.baseline_condition} − {item.variant_condition}</h3><strong>{number(item.difference_log_probability)} Δ log p</strong><p>Scored next-token ID {item.scored_token_id}. Same text prefix: {item.identical_text_prefix ? 'yes' : 'no'} · same starting KV: {item.same_starting_kv ? 'yes' : 'no'}. This is a query-level intervention, not attribution to this one layer’s read.</p></div>)}
            </>}
            {episode.outcome && <details className="outcome"><summary>Saved evaluation outcome</summary><pre>{JSON.stringify(episode.outcome, null, 2)}</pre></details>}
          </aside>
        </div>
        <footer className="footer-note">A large write does not establish later retrieval. A large injected read does not establish a better answer. Missing measurements are never treated as zero.</footer>
      </>}
    </main>
  </div>;
}

export default App;
