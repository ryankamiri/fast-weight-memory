import type { Episode, TraceFolder } from './trace';
import { loadEpisode, referenceCondition, splitEpisodeKey } from './trace';

export interface LayerCalibration {
  layer: number;
  writes: number[];
  reads: number[];
  writeP99: number;
  readP99: number;
}

export interface SharedCalibration {
  layers: Map<number, LayerCalibration>;
  referenceEpisodes: number;
  referenceRuns: number;
}

export function percentile(sorted: number[], value: number): number | null {
  if (sorted.length === 0 || !Number.isFinite(value)) return null;
  let left = 0;
  let right = sorted.length;
  while (left < right) {
    const middle = (left + right) >>> 1;
    if (sorted[middle] <= value) left = middle + 1;
    else right = middle;
  }
  return (100 * left) / sorted.length;
}

function quantile(sorted: number[], fraction: number): number {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
}

export function compatibleRuns(runs: TraceFolder[], selected: TraceFolder): TraceFolder[] {
  if (selected.synthetic) return [selected];
  const tokenizer = selected.manifest.run_metadata.tokenizer;
  const layers = selected.manifest.run_metadata.layers;
  if (typeof tokenizer !== 'string' || !Array.isArray(layers)) return [selected];
  return runs.filter(run => !run.synthetic
    && run.manifest.schema === selected.manifest.schema
    && run.manifest.run_metadata.tokenizer === tokenizer
    && JSON.stringify(run.manifest.run_metadata.layers) === JSON.stringify(layers));
}

function addEpisode(
  episode: Episode,
  values: Map<number, { writes: number[]; reads: number[] }>,
) {
  for (const write of episode.writes) {
    if (write.position === null || !write.write_enabled || write.proposed_write_norm <= 0) continue;
    const layer = values.get(write.layer) ?? { writes: [], reads: [] };
    layer.writes.push(write.proposed_write_norm);
    values.set(write.layer, layer);
  }
  for (const read of episode.reads) {
    if (!read.enabled || read.relative_injection === null || read.relative_injection <= 0) continue;
    const layer = values.get(read.layer) ?? { writes: [], reads: [] };
    layer.reads.push(read.relative_injection);
    values.set(read.layer, layer);
  }
}

// Load each reference episode once and derive every layer's scale together.
// Compatible runs share these scales, so their token colors are comparable.
export async function calibrateRuns(folders: TraceFolder[]): Promise<SharedCalibration> {
  const values = new Map<number, { writes: number[]; reads: number[] }>();
  let referenceEpisodes = 0;
  for (const folder of folders) {
    const condition = referenceCondition(folder.manifest);
    const checkpoint = folder.manifest.run_metadata.checkpoint;
    const keys = Object.keys(folder.manifest.episodes)
      .filter(key => splitEpisodeKey(key).condition === condition);
    for (const key of keys) {
      const episode = await loadEpisode(folder, key);
      if (typeof checkpoint === 'string' && episode.checkpoint !== checkpoint) continue;
      addEpisode(episode, values);
      referenceEpisodes += 1;
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
  }
  const layers = new Map<number, LayerCalibration>();
  for (const [layer, signals] of values) {
    signals.writes.sort((a, b) => a - b);
    signals.reads.sort((a, b) => a - b);
    layers.set(layer, {
      layer,
      ...signals,
      writeP99: quantile(signals.writes, 0.99),
      readP99: quantile(signals.reads, 0.99),
    });
  }
  return { layers, referenceEpisodes, referenceRuns: folders.length };
}

export function intensity(value: number, p99: number): number {
  if (!Number.isFinite(value) || value <= 0 || p99 <= 0) return 0;
  return Math.min(value / p99, 1);
}
