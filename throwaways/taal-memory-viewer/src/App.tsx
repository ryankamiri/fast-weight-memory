import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Comparison, Episode, InternalPrefix, ReadEvent, TraceFolder, WriteEvent } from './trace';
import { discoverTraces, loadComparisons, loadEpisode, readTimingNotes, splitEpisodeKey } from './trace';
import type { SharedCalibration } from './calibration';
import { calibrateRuns, compatibleRuns, intensity, percentile } from './calibration';
import { layerSignal, summarizeActivity } from './activity';
import type { Mode } from './activity';
import { TraceSidebar, conditionLabel } from './TraceSidebar';
import { InternalToken } from './InternalToken';
import { LayerProfile } from './LayerProfile';
import { exampleName, expectedAnswer, trajectoryTokens } from './answer';
import './answer.css';
import './layer-profile.css';

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
  const [layer, setLayer] = useState<number | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [expandedPrefixes, setExpandedPrefixes] = useState<Set<string>>(new Set());
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [calibration, setCalibration] = useState<SharedCalibration | null>(null);
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
  const visibleTokens = useMemo(() => episode ? trajectoryTokens(episode) : [], [episode]);
  const predictingToken = useMemo(() => episode?.tokens.filter(token => token.phase === 'prompt').at(-1), [episode]);
  const referenceRuns = folder ? compatibleRuns(runs, folder) : [];
  const referenceKey = referenceRuns.map(run => `${run.id}@${run.revision}`).sort().join('|');

  useEffect(() => {
    if (!folder) return;
    const preferred = Object.keys(folder.manifest.episodes).some(entry => splitEpisodeKey(entry).condition === 'correct_full')
      ? 'correct_full' : splitEpisodeKey(keys[0]).condition;
    setKey(previous => folder.manifest.episodes[previous] ? previous : keys.find(entry => splitEpisodeKey(entry).condition === preferred) ?? keys[0]);
    setEpisode(null);
    setLayer(null);
    setSelection(null);
    setComparisons([]);
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
      setLayer(null);
    }).catch(cause => { if (!cancelled) setError(String(cause)); })
      .finally(() => { if (!cancelled) setBusy(false); });
    const exampleId = splitEpisodeKey(key).exampleId;
    loadComparisons(folder, exampleId).then(loaded => { if (!cancelled) setComparisons(loaded); }).catch(cause => { if (!cancelled) setError(String(cause)); });
    return () => { cancelled = true; };
  }, [folder, key]);

  useEffect(() => {
    if (!referenceKey) return;
    let cancelled = false;
    setCalibration(null);
    calibrateRuns(referenceRuns)
      .then(result => { if (!cancelled) setCalibration(result); })
      .catch(cause => { if (!cancelled) setError(String(cause)); });
    return () => { cancelled = true; };
  }, [referenceKey]);

  const writesAt = useMemo(() => {
    const positions = new Map<number, Map<number, WriteEvent>>();
    for (const write of episode?.writes ?? []) {
      if (write.position === null) continue;
      const byLayer = positions.get(write.position) ?? new Map<number, WriteEvent>();
      byLayer.set(write.layer, write);
      positions.set(write.position, byLayer);
    }
    return positions;
  }, [episode]);
  const readsAt = useMemo(() => {
    const positions = new Map<number, Map<number, ReadEvent>>();
    for (const read of episode?.reads ?? []) {
      const byLayer = positions.get(read.position) ?? new Map<number, ReadEvent>();
      byLayer.set(read.layer, read);
      positions.set(read.position, byLayer);
    }
    return positions;
  }, [episode]);
  const internalWrites = useMemo(() => {
    const groups = new Map<string, Map<number, WriteEvent>>();
    for (const write of episode?.writes ?? []) {
      if (write.internal_index === null) continue;
      const id = `${write.segment_index}:${write.internal_index}`;
      const byLayer = groups.get(id) ?? new Map<number, WriteEvent>();
      byLayer.set(write.layer, write);
      groups.set(id, byLayer);
    }
    return groups;
  }, [episode]);
  const prefixesAt = useMemo(() => {
    const map = new Map<number, InternalPrefix[]>();
    for (const prefix of episode?.internal_prefixes ?? []) {
      const existing = map.get(prefix.before_position) ?? [];
      if (!existing.some(item => item.segment_index === prefix.segment_index)) {
        map.set(prefix.before_position, [...existing, prefix]);
      }
    }
    return map;
  }, [episode]);

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
        setLayer(null);
        document.getElementById(`token-${token.position}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
        setError('');
        return;
      }
    }
  }

  function togglePrefix(prefix: InternalPrefix) {
    const id = `${prefix.segment_index}:${prefix.before_position}`;
    setExpandedPrefixes(previous => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  const selectedWrites = selection?.kind === 'text'
    ? writesAt.get(selection.position) ?? new Map<number, WriteEvent>()
    : selection?.kind === 'internal'
      ? internalWrites.get(`${selection.segment}:${selection.index}`) ?? new Map<number, WriteEvent>()
      : new Map<number, WriteEvent>();
  const selectedReads = selection?.kind === 'text'
    ? readsAt.get(selection.position) ?? new Map<number, ReadEvent>()
    : new Map<number, ReadEvent>();
  const selectedSignals = layers.map(item => layerSignal(item, mode, selectedWrites, selectedReads, calibration));
  const selectedActivity = summarizeActivity(selectedSignals);
  const selectedWrite = layer === null ? undefined : selectedWrites.get(layer);
  const selectedRead = layer === null ? undefined : selectedReads.get(layer);
  const selectedToken = selection?.kind === 'text' ? visibleTokens.find(token => token.position === selection.position) : undefined;
  const scoredPosition = selectedToken?.phase === 'generated' ? predictingToken?.position : selection?.kind === 'text' ? selection.position : undefined;
  const relevantComparisons = scoredPosition !== undefined ? comparisons.filter(item => item.scored_position === scoredPosition && (item.baseline_condition === currentCondition || item.variant_condition === currentCondition)) : [];
  const refValues = layer === null ? undefined : mode === 'writes'
    ? calibration?.layers.get(layer)?.writes
    : calibration?.layers.get(layer)?.reads;
  const answer = episode ? expectedAnswer(episode) : null;
  const candidateText = episode?.outcome && typeof episode.outcome.candidate_choice_token === 'string'
    ? episode.outcome.candidate_choice_token.trim()
    : null;
  const candidateId = episode?.outcome?.candidate_choice_token_id;
  const candidateChoice = candidateText || (typeof candidateId === 'number' ? `Token #${candidateId}` : null);
  const topToken = episode?.outcome?.vocabulary_top_token;
  const targetLogProbability = episode?.outcome?.target_log_probability;
  const candidateCorrect = episode?.outcome?.candidate_correct;

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">M</span><div><strong>TaaL Memory Viewer</strong><small>Offline trace inspection</small></div></div>
      <TraceSidebar runs={runs} selectedRun={folder?.id ?? ''} selectedKey={key} onSelect={selectTrace} />
    </aside>

    <main className="main">
      <header className="topbar"><div><span className="eyebrow">TRAJECTORY INSPECTION</span><h1>{episode ? exampleName(episode.example_id, folder?.exampleLabels ?? {}) : 'Inspect a memory trace'}</h1><p>{episode ? `${conditionLabel(episode.condition_id)} · ${episode.tokens.filter(token => token.phase === 'prompt').length.toLocaleString()} prompt tokens` : 'A token-level view of online writes and injected reads.'}</p></div><div className="topbar-meta">{episode?.checkpoint && <><span>CHECKPOINT</span><strong title={episode.checkpoint}>{episode.checkpoint.split('/').at(-1)}</strong></>}</div></header>
      {error && <div role="alert" className="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
      {discoveryErrors.length > 0 && <div role="alert" className="alert"><div><strong>Exports not loaded</strong>{discoveryErrors.map(message => <p key={message}>{message}</p>)}</div></div>}
      {!episode ? <section className="empty"><h2>{busy ? 'Reading episode…' : 'Checking local traces…'}</h2></section> : <>
        {episode.outcome && <section className="answer-summary" aria-label="Answer scoring">
          <div><span>EXPECTED NEXT TOKEN</span><strong>{answer ?? 'Not exported'}</strong></div>
          <div><span>BEST OF CANDIDATES</span><strong>{candidateChoice ?? 'Not recorded'}</strong><small>{candidateCorrect === true ? 'Correct' : candidateCorrect === false ? 'Incorrect' : 'Not scored'}</small></div>
          <div><span>GREEDY NEXT TOKEN</span><strong>{typeof topToken === 'string' ? JSON.stringify(topToken) : 'Not recorded'}</strong></div>
          <div><span>TARGET LOG PROBABILITY</span><strong>{typeof targetLogProbability === 'number' ? number(targetLogProbability) : 'Not recorded'}</strong></div>
        </section>}
        <section className="viewbar"><div className="view-controls"><div className="segmented"><button className={mode === 'writes' ? 'active' : ''} onClick={() => { setMode('writes'); setLayer(null); }}>Memory writes</button><button className={mode === 'reads' ? 'active' : ''} onClick={() => { setMode('reads'); setLayer(null); }}>Memory reads</button></div></div><div className="search"><input aria-label="Find text" value={search} onChange={event => setSearch(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') jumpToSearch(); }} placeholder="Find text in episode" /><button onClick={jumpToSearch}>Find</button></div></section>
        <div className="content-grid">
          <section className="trajectory-panel"><div className="panel-heading"><div><span className="eyebrow">SEQUENCE</span><h2>Token trajectory</h2></div><span>Click a token to inspect its signal</span></div><div className="trajectory" aria-label="Token trajectory" aria-busy={!calibration}>
            {visibleTokens.map(token => {
              const tokenWrites = writesAt.get(token.position) ?? new Map<number, WriteEvent>();
              const tokenReads = readsAt.get(token.position) ?? new Map<number, ReadEvent>();
              const activity = summarizeActivity(layers.map(item => layerSignal(item, mode, tokenWrites, tokenReads, calibration)));
              let className = 'token';
              if (activity.status === 'missing') className += ' missing';
              else if (activity.status === 'disabled' || activity.strength === 0) className += ' neutral';
              if (token.phase === 'generated') className += ' generated-token';
              if (selection?.kind === 'text' && selection.position === token.position) className += ' selected';
              return <span key={token.position} className="token-slot">
                {(prefixesAt.get(token.position) ?? []).map(prefix => {
                  const id = `${prefix.segment_index}:${prefix.before_position}`;
                  const expanded = expandedPrefixes.has(id);
                  return <span key={id} className="prefix-wrap"><button className="prefix-marker" onClick={() => togglePrefix(prefix)} aria-expanded={expanded}>{prefix.count} internal memory tokens <b>{expanded ? '▴' : '▾'}</b></button>{expanded && <span className="prefix-expanded">{Array.from({ length: prefix.count }, (_, index) => {
                    const tokenWrites = internalWrites.get(`${prefix.segment_index}:${index}`) ?? new Map<number, WriteEvent>();
                    const activity = summarizeActivity(layers.map(item => layerSignal(item, 'writes', tokenWrites, new Map<number, ReadEvent>(), calibration)));
                    return <InternalToken key={index} index={index} activity={activity} mode={mode} selected={selection?.kind === 'internal' && selection.segment === prefix.segment_index && selection.index === index} onSelect={() => { setSelection({ kind: 'internal', segment: prefix.segment_index, index }); setLayer(null); }} />;
                  })}</span>}</span>;
                })}
                <button id={`token-${token.position}`} title={`${token.phase === 'generated' ? 'Greedy prediction' : `Token ${token.position}`} · id ${token.token_id} · ${JSON.stringify(token.text)} · ${activity.measuredLayers} measured layers`} aria-label={`${token.phase === 'generated' ? 'Greedy prediction' : `Token ${token.position}`}: ${JSON.stringify(token.text)}`} className={className} style={eventColor(activity.strength, 1, mode)} onClick={() => { setSelection({ kind: 'text', position: token.position }); setLayer(null); }}><span>{token.text.trim() || (token.text.includes('\n') ? '↵' : token.text.includes('\t') ? '⇥' : token.text ? '␠' : `⟨${token.token_id}⟩`)}</span></button>
              </span>;
            })}
          </div></section>
          <aside className="details-panel"><div className="panel-heading"><div><span className="eyebrow">INSPECTOR</span><h2>{selection?.kind === 'internal' ? `Internal token P${selection.index + 1}` : selectedToken ? `Token ${selectedToken.position}` : 'Select a token'}</h2></div></div>
            {selection && <>
              {selectedToken?.phase === 'generated' && <div className="prediction-note">The preceding prompt token's read influenced this prediction. Reads and writes on the predicted token happen only after it is fed back.{predictingToken && <button onClick={() => { setSelection({ kind: 'text', position: predictingToken.position }); document.getElementById(`token-${predictingToken.position}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' }); }}>Inspect predicting token</button>}</div>}
              {selectedToken && <div className="selected-text"><span>TEXT FRAGMENT</span><strong>{JSON.stringify(selectedToken.text)}</strong><small>ID {selectedToken.token_id} · {selectedToken.phase}</small></div>}
              {!(selection.kind === 'internal' && mode === 'reads') && <>
                <p className="profile-summary">{selectedActivity.measuredLayers} of {layers.length} layers measured · {selectedActivity.unusuallyActiveLayers} above their 95th percentile</p>
                <LayerProfile mode={mode} signals={selectedSignals} selectedLayer={layer} onSelect={setLayer} />
              </>}
              {layer === null && !(selection.kind === 'internal' && mode === 'reads') && <p className="unavailable">Select a layer bar for its raw measurements.</p>}
              {mode === 'writes' && layer !== null && (selectedWrite
                ? <div className="metric-grid">
                    <h3 className="layer-detail-heading">Layer {layer} · write details</h3>
                    {(selectedWrite.chunk_size ?? 1) > 1 && <p className="inspector-note">This token contributes to a {selectedWrite.chunk_size}-token gradient. Weights change only at the chunk boundary; the boundary's net change belongs to the whole chunk.</p>}
                    <Metric label="NEW WRITE · ||−gₜ||" value={selectedWrite.write_enabled ? number(selectedWrite.proposed_write_norm) : 'Masked'} note="Includes learned write strength" />
                    <Metric label="REFERENCE PERCENTILE" value={selectedWrite.write_enabled && calibration ? `${number(percentile(refValues ?? [], selectedWrite.proposed_write_norm), 3)}%` : 'Unavailable'} note="Rank among reference writes, not usefulness" />
                    <Metric label="PREVIOUS ||W||" value={number(selectedWrite.previous_weight_norm)} />
                    <Metric label="WRITE / PREVIOUS" value={number(selectedWrite.previous_weight_norm ? selectedWrite.proposed_write_norm / selectedWrite.previous_weight_norm : null)} />
                    <Metric label={selectedWrite.chunk_boundary && (selectedWrite.chunk_size ?? 1) > 1 ? 'CHUNK NET CHANGE · ||ΔW||' : 'NET CHANGE · ||ΔWₜ||'} value={number(selectedWrite.net_weight_change_norm)} note={selectedWrite.chunk_boundary ? 'Includes momentum and forgetting' : 'No weight update before this chunk boundary'} />
                    <Metric label="OTHER MOVEMENT · ||ΔWₜ + gₜ||" value={number(selectedWrite.other_movement_norm)} />
                    <Metric label="WRITE ↔ NET ALIGNMENT" value={number(selectedWrite.write_to_net_alignment)} note="Unavailable for chunk-level updates" />
                    <Metric label="WRITE STRENGTH θₜ" value={number(selectedWrite.write_strength)} />
                  </div>
                : <p className="unavailable">No write measurement at layer {layer}.</p>)}
              {mode === 'reads' && selection.kind === 'internal' && <div className="metric-grid"><Metric label="READ STATUS" value="Read output discarded" note="Not injected into the transformer residual stream" /></div>}
              {mode === 'reads' && selection.kind === 'text' && layer !== null && (selectedRead
                ? <div className="metric-grid">
                    <h3 className="layer-detail-heading">Layer {layer} · read details</h3>
                    <Metric label="READ STATUS" value={selectedRead.enabled ? 'Enabled' : 'Disabled'} note={readTimingNotes[selectedRead.state_timing]} />
                    <Metric label="INJECTED ||δ||" value={number(selectedRead.injected_norm)} note="After gate and read scale" />
                    <Metric label="INCOMING ||h||" value={number(selectedRead.incoming_norm)} />
                    <Metric label="RELATIVE INJECTION" value={number(selectedRead.relative_injection)} />
                    <Metric label="REFERENCE PERCENTILE" value={selectedRead.enabled && calibration && selectedRead.relative_injection != null ? `${number(percentile(refValues ?? [], selectedRead.relative_injection), 3)}%` : 'Unavailable'} />
                    <Metric label="READ SCALE" value={number(selectedRead.read_scale)} />
                    <Metric label="RESIDUAL GATE" value={number(selectedRead.residual_gate)} />
                  </div>
                : <p className="unavailable">No read measurement at layer {layer}.</p>)}
              {selection.kind === 'internal' && <p className="inspector-note">This is a learned memory-only token. Its read output is discarded; its write can still change the fast-weight state.</p>}
              {selection.kind === 'text' && (prefixesAt.get(selection.position) ?? []).map(prefix => <div key={prefix.segment_index} className="prefix-note">Internal prefix before this token: {prefix.count} tokens{layer !== null && ` · layer ${layer} net state change ${number(episode.internal_prefixes.find(item => item.segment_index === prefix.segment_index && item.layer === layer)?.net_weight_change_norm)}`}</div>)}
              {mode === 'reads' && relevantComparisons.map(item => <div className="comparison" key={`${item.baseline_condition}/${item.variant_condition}`}><span className="eyebrow">PAIRED PREDICTION · {item.scope === 'prediction_token' ? 'PREDICTING TOKEN' : 'WHOLE QUERY'}</span><h3>{item.baseline_condition} − {item.variant_condition}</h3><strong>{number(item.difference_log_probability)} Δ log p</strong><p>Scored next-token ID {item.scored_token_id}. Same text prefix: {item.identical_text_prefix ? 'yes' : 'no'} · same starting KV: {item.same_starting_kv ? 'yes' : 'no'}. This score difference measures the recorded intervention; token colors show activity.</p></div>)}
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
