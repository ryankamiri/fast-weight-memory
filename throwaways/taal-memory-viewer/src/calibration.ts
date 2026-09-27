import type { Episode, TraceFolder } from './trace';
import { loadEpisode, splitEpisodeKey } from './trace';

export interface LayerCalibration {
  layer: number;
  referenceCondition: string;
  referenceEpisodes: number;
  writes: number[];
  reads: number[];
  writeP99: number;
  readP99: number;
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

export function valuesForLayer(episode: Episode, layer: number): { writes: number[]; reads: number[] } {
  return {
    writes: episode.writes
      .filter(write => write.layer === layer && write.position !== null && write.write_enabled && write.proposed_write_norm > 0)
      .map(write => write.proposed_write_norm),
    reads: episode.reads
      .filter(read => read.layer === layer && read.enabled && read.relative_injection !== null && read.relative_injection > 0)
      .map(read => read.relative_injection!),
  };
}

export async function calibrateLayer(
  folder: TraceFolder,
  layer: number,
  condition: string,
  checkpoint: string,
  onProgress?: (done: number, total: number) => void,
): Promise<LayerCalibration> {
  // One fixed reference condition prevents intervention copies of the same
  // episode from silently overweighting the scale.
  const keys = Object.keys(folder.manifest.episodes)
    .filter(key => splitEpisodeKey(key).condition === condition);
  const writes: number[] = [];
  const reads: number[] = [];
  let referenceEpisodes = 0;
  for (const [index, key] of keys.entries()) {
    const episode = await loadEpisode(folder, key);
    if (episode.checkpoint === checkpoint) {
      const values = valuesForLayer(episode, layer);
      writes.push(...values.writes);
      reads.push(...values.reads);
      referenceEpisodes += 1;
    }
    onProgress?.(index + 1, keys.length);
    // Permit the browser to paint progress during long local imports.
    await new Promise<void>(resolve => setTimeout(resolve, 0));
  }
  writes.sort((a, b) => a - b);
  reads.sort((a, b) => a - b);
  return {
    layer,
    referenceCondition: condition,
    referenceEpisodes,
    writes,
    reads,
    writeP99: quantile(writes, 0.99),
    readP99: quantile(reads, 0.99),
  };
}

export function intensity(value: number, p99: number): number {
  if (!Number.isFinite(value) || value <= 0 || p99 <= 0) return 0;
  return Math.min(value / p99, 1);
}
