import type { LayerSignal, Mode } from './activity';

function raw(value: number | null): string {
  return value === null ? '—' : value.toPrecision(3);
}

export function LayerProfile({ mode, signals, selectedLayer, onSelect }: {
  mode: Mode;
  signals: LayerSignal[];
  selectedLayer: number | null;
  onSelect: (layer: number) => void;
}) {
  if (!signals.some(signal => signal.status !== 'missing')) {
    return <p className="unavailable">No per-layer {mode === 'writes' ? 'write' : 'read'} measurement for this token.</p>;
  }
  return <section className="layer-profile" aria-label="Layer activity at selected token">
    <div className="layer-profile-heading"><strong>Layer profile</strong><span>Signal relative to each layer’s shared P99</span></div>
    <div className="layer-profile-rows">
      {signals.map(signal => <button
        key={signal.layer}
        type="button"
        className={`layer-profile-row ${mode}${selectedLayer === signal.layer ? ' selected' : ''}`}
        onClick={() => onSelect(signal.layer)}
        aria-pressed={selectedLayer === signal.layer}
        aria-label={`Layer ${signal.layer}: ${signal.status === 'measured' ? `${raw(signal.value)} raw signal, ${Math.round(signal.strength * 100)} percent of shared P99` : signal.status}`}
        title={`Layer ${signal.layer} · ${signal.status === 'measured' ? `${raw(signal.value)} raw · ${(signal.referencePercentile ?? 0).toFixed(1)}th percentile` : signal.status}`}
      >
        <span className="layer-profile-index">{signal.layer}</span>
        <span className="layer-profile-track"><span style={{ width: `${Math.max(0, Math.min(signal.strength, 1)) * 100}%` }} /></span>
        <span className="layer-profile-value">{signal.status === 'measured' ? raw(signal.value) : signal.status === 'disabled' ? 'Off' : '—'}</span>
      </button>)}
    </div>
  </section>;
}
