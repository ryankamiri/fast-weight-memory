import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { layerSignal, summarizeActivity } from './activity';
import { LayerProfile } from './LayerProfile';
import type { SharedCalibration } from './calibration';
import { demoEpisode } from './demo';

const calibration: SharedCalibration = {
  referenceEpisodes: 1,
  referenceRuns: 1,
  layers: new Map([
    [0, { layer: 0, writes: [1, 2, 4], reads: [0.1, 0.2], writeP99: 4, readP99: 0.2 }],
    [1, { layer: 1, writes: [10, 20], reads: [1, 2], writeP99: 20, readP99: 2 }],
  ]),
};

describe('all-layer token activity', () => {
  it('normalizes each layer before taking the strongest three', () => {
    const signals = [
      { layer: 0, status: 'measured' as const, value: 1, strength: 0.25, referencePercentile: 50 },
      { layer: 1, status: 'measured' as const, value: 2, strength: 0.5, referencePercentile: 70 },
      { layer: 2, status: 'measured' as const, value: 3, strength: 0.75, referencePercentile: 95 },
      { layer: 3, status: 'measured' as const, value: 4, strength: 1, referencePercentile: 99 },
    ];
    expect(summarizeActivity(signals)).toEqual({ status: 'measured', strength: 0.75, measuredLayers: 4, unusuallyActiveLayers: 2 });
  });

  it('keeps a disabled read separate from an absent measurement', () => {
    const read = demoEpisode.reads[0];
    const disabled = layerSignal(0, 'reads', new Map(), new Map([[0, { ...read, enabled: false }]]), calibration);
    const missing = layerSignal(1, 'reads', new Map(), new Map(), calibration);
    expect(disabled.status).toBe('disabled');
    expect(missing.status).toBe('missing');
    expect(summarizeActivity([disabled]).status).toBe('disabled');
    expect(summarizeActivity([missing]).status).toBe('missing');
  });

  it('renders ordered layer bars with separate disabled and missing labels', () => {
    const signals = [
      { layer: 0, status: 'measured' as const, value: 2, strength: 0.5, referencePercentile: 80 },
      { layer: 1, status: 'disabled' as const, value: null, strength: 0, referencePercentile: null },
      { layer: 2, status: 'missing' as const, value: null, strength: 0, referencePercentile: null },
    ];
    const html = renderToStaticMarkup(<LayerProfile mode="writes" signals={signals} selectedLayer={0} onSelect={() => {}} />);
    expect(html).toContain('width:50%');
    expect(html).toContain('Layer 1: disabled');
    expect(html).toContain('Layer 2: missing');
    expect(html).toContain('aria-pressed="true"');
  });
});
