import type { ReadEvent, WriteEvent } from './trace';
import type { SharedCalibration } from './calibration';
import { intensity, percentile } from './calibration';

export type Mode = 'writes' | 'reads';
export type LayerSignal = {
  layer: number;
  status: 'measured' | 'disabled' | 'missing';
  value: number | null;
  strength: number;
  referencePercentile: number | null;
};

export type ActivitySummary = {
  status: 'measured' | 'disabled' | 'missing';
  strength: number;
  measuredLayers: number;
  unusuallyActiveLayers: number;
};

export function layerSignal(
  layer: number,
  mode: Mode,
  writes: Map<number, WriteEvent>,
  reads: Map<number, ReadEvent>,
  calibration: SharedCalibration | null,
): LayerSignal {
  const event = mode === 'writes' ? writes.get(layer) : reads.get(layer);
  if (!event) return { layer, status: 'missing', value: null, strength: 0, referencePercentile: null };
  if (mode === 'writes' && !(event as WriteEvent).write_enabled) {
    return { layer, status: 'disabled', value: null, strength: 0, referencePercentile: null };
  }
  if (mode === 'reads' && !(event as ReadEvent).enabled) {
    return { layer, status: 'disabled', value: null, strength: 0, referencePercentile: null };
  }
  const value = mode === 'writes'
    ? (event as WriteEvent).proposed_write_norm
    : (event as ReadEvent).relative_injection;
  if (value === null || !Number.isFinite(value)) {
    return { layer, status: 'missing', value: null, strength: 0, referencePercentile: null };
  }
  const reference = calibration?.layers.get(layer);
  const distribution = mode === 'writes' ? reference?.writes : reference?.reads;
  const p99 = mode === 'writes' ? reference?.writeP99 : reference?.readP99;
  return {
    layer,
    status: 'measured',
    value,
    strength: p99 === undefined ? 0 : intensity(value, p99),
    referencePercentile: distribution ? percentile(distribution, value) : null,
  };
}

export function summarizeActivity(signals: LayerSignal[]): ActivitySummary {
  const measured = signals.filter(signal => signal.status === 'measured');
  if (measured.length) {
    const strongest = measured.map(signal => signal.strength).sort((a, b) => b - a).slice(0, 3);
    return {
      status: 'measured',
      strength: strongest.reduce((sum, value) => sum + value, 0) / strongest.length,
      measuredLayers: measured.length,
      unusuallyActiveLayers: measured.filter(signal => (signal.referencePercentile ?? 0) >= 95).length,
    };
  }
  return {
    status: signals.some(signal => signal.status === 'missing') ? 'missing' : 'disabled',
    strength: 0,
    measuredLayers: 0,
    unusuallyActiveLayers: 0,
  };
}
