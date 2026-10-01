import { strFromU8, gunzipSync } from 'fflate';

export const SCHEMA = 'taal-memory-trace/v1';

export interface TraceToken {
  position: number;
  token_id: number;
  text: string;
  phase: 'prompt' | 'generated';
}

export interface WriteEvent {
  layer: number;
  position: number | null;
  segment_index: number;
  internal_index: number | null;
  write_enabled: boolean;
  write_strength: number;
  proposed_write_norm: number;
  previous_weight_norm: number;
  net_weight_change_norm: number;
  other_movement_norm: number | null;
  write_to_net_alignment: number | null;
  chunk_size?: number;
  chunk_boundary?: boolean;
}

export interface ReadEvent {
  layer: number;
  position: number;
  segment_index: number;
  enabled: boolean;
  read_scale: number;
  residual_gate: number;
  injected_norm: number;
  incoming_norm: number;
  relative_injection: number | null;
  state_timing: 'pre-write' | 'post-write';
}

export interface InternalPrefix {
  layer: number;
  segment_index: number;
  before_position: number;
  count: number;
  net_weight_change_norm: number;
}

export interface Episode {
  schema: string;
  run_id: string;
  example_id: string;
  condition_id: string;
  checkpoint: string;
  tokenizer_id: string;
  tokens: TraceToken[];
  writes: WriteEvent[];
  reads: ReadEvent[];
  internal_prefixes: InternalPrefix[];
  outcome: Record<string, unknown> | null;
  metadata: Record<string, unknown>;
}

export interface Comparison {
  schema: string;
  example_id: string;
  baseline_condition: string;
  variant_condition: string;
  intervention: 'read_scale' | 'memory_state';
  scope: 'whole_query';
  identical_text_prefix: boolean;
  same_starting_kv: boolean;
  same_starting_memory: boolean;
  scored_position: number;
  scored_token_id: number;
  baseline_log_probability: number;
  variant_log_probability: number;
  difference_log_probability: number;
}

export interface Manifest {
  schema: string;
  run_metadata: Record<string, unknown>;
  episodes: Record<string, string>;
  comparisons: Record<string, string>;
}

export interface TraceFolder {
  manifest: Manifest;
  id: string;
  revision: string;
  synthetic: boolean;
  rootName: string;
}

export interface TraceIndex { runs: TraceFolder[]; errors: string[] }

export async function discoverTraces(): Promise<TraceIndex> {
  const response = await fetch('/api/trace-runs', { cache: 'no-store' });
  if (!response.ok) throw new Error('Unable to check local data directory');
  return response.json();
}

async function readGzipJson<T>(folder: TraceFolder, path: string): Promise<T> {
  const params = new URLSearchParams({ run: folder.id, path, revision: folder.revision });
  const response = await fetch(`/api/trace-file?${params}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Trace file unavailable: ${path}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  return JSON.parse(strFromU8(gunzipSync(bytes))) as T;
}

export async function loadEpisode(folder: TraceFolder, key: string): Promise<Episode> {
  const relative = folder.manifest.episodes[key];
  if (!relative) throw new Error(`Episode file is missing: ${key}`);
  const episode = await readGzipJson<Episode>(folder, relative);
  if (episode.schema !== SCHEMA || !Array.isArray(episode.tokens) || !Array.isArray(episode.writes) || !Array.isArray(episode.reads)) {
    throw new Error(`Episode has an unsupported schema or missing event arrays: ${key}`);
  }
  return episode;
}

export async function loadComparisons(folder: TraceFolder, exampleId: string): Promise<Comparison[]> {
  const entries = Object.entries(folder.manifest.comparisons)
    .filter(([key]) => key.startsWith(`${exampleId}/`));
  const comparisons: Comparison[] = [];
  for (const [, relative] of entries) {
    const comparison = await readGzipJson<Comparison>(folder, relative);
    if (comparison.schema === SCHEMA) comparisons.push(comparison);
  }
  return comparisons;
}

export function splitEpisodeKey(key: string): { exampleId: string; condition: string } {
  const split = key.lastIndexOf('/');
  if (split < 0) throw new Error(`Invalid episode key: ${key}`);
  return { exampleId: key.slice(0, split), condition: key.slice(split + 1) };
}

export function referenceCondition(manifest: Manifest): string {
  const conditions = new Set(Object.keys(manifest.episodes).map(key => splitEpisodeKey(key).condition));
  return conditions.has('correct_full') ? 'correct_full' : [...conditions].sort()[0] ?? '';
}
