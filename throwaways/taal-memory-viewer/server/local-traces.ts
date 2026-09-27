import { mkdir, readdir, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { resolve, relative, sep } from 'node:path';
import { gzipSync, gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import type { Plugin } from 'vite';
import { demoEpisode, demoManifest } from '../src/demo.ts';
import { SCHEMA, type Manifest, type TraceIndex } from '../src/trace.ts';

export async function seedSynthetic(root: string) {
  const directory = resolve(root, 'synthetic-example/memory_traces');
  await mkdir(resolve(directory, 'episodes'), { recursive: true });
  await writeFile(resolve(directory, 'episodes/demo.json.gz'), gzipSync(JSON.stringify(demoEpisode)));
  await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(demoManifest, null, 2));
}

async function safeFile(root: string, path: string) {
  const base = await realpath(root);
  const file = await realpath(resolve(base, path));
  const local = relative(base, file);
  if (local === '..' || local.startsWith(`..${sep}`) || local.startsWith(sep)) throw new Error('Path escapes trace directory');
  return file;
}

const validated = new Set<string>();

function checkFields(record: unknown, fields: Record<string, 'number' | 'string' | 'boolean' | 'nullable'>) {
  if (!record || typeof record !== 'object') throw new Error('Invalid event record');
  const value = record as Record<string, unknown>;
  for (const [field, type] of Object.entries(fields)) {
    const item = value[field];
    const valid = type === 'nullable' ? item === null || typeof item === 'number' && Number.isFinite(item)
      : typeof item === type && (type !== 'number' || Number.isFinite(item));
    if (!valid) throw new Error(`Invalid or missing ${field}`);
  }
}

function validateEvents(payload: Record<string, unknown>) {
  for (const token of payload.tokens as unknown[]) {
    checkFields(token, { position: 'number', token_id: 'number', text: 'string', phase: 'string' });
    if (!['prompt', 'generated'].includes((token as { phase: string }).phase)) throw new Error('Invalid token phase');
  }
  for (const event of payload.writes as unknown[]) checkFields(event, {
    layer: 'number', position: 'nullable', segment_index: 'number', internal_index: 'nullable',
    write_enabled: 'boolean', write_strength: 'number', proposed_write_norm: 'number',
    previous_weight_norm: 'number', net_weight_change_norm: 'number', other_movement_norm: 'number', write_to_net_alignment: 'nullable',
  });
  for (const event of payload.reads as unknown[]) {
    checkFields(event, { layer: 'number', position: 'number', segment_index: 'number', enabled: 'boolean', read_scale: 'number', residual_gate: 'number', injected_norm: 'number', incoming_norm: 'number', relative_injection: 'nullable' });
    if ((event as { state_timing: string }).state_timing !== 'post-write') throw new Error('Unsupported read timing');
  }
  for (const event of payload.internal_prefixes as unknown[]) checkFields(event, { layer: 'number', segment_index: 'number', before_position: 'number', count: 'number', net_weight_change_norm: 'number' });
}

export async function scanTraces(root: string): Promise<TraceIndex> {
  const index: TraceIndex = { runs: [], errors: [] };
  async function visit(directory: string) {
    const entries = await readdir(directory, { withFileTypes: true });
    if (entries.some(entry => entry.isFile() && entry.name === 'manifest.json')) {
      const id = relative(root, directory).split(sep).join('/');
      try {
        const manifest = JSON.parse(await readFile(resolve(directory, 'manifest.json'), 'utf8')) as Manifest;
        if (manifest.schema !== SCHEMA) throw new Error(`Unsupported schema: ${manifest.schema}`);
        for (const field of ['run_metadata', 'episodes', 'comparisons'] as const) {
          if (!manifest[field] || typeof manifest[field] !== 'object' || Array.isArray(manifest[field])) throw new Error(`Missing ${field} mapping`);
        }
        if (!Object.keys(manifest.episodes).length) throw new Error('No episodes indexed');
        const revisions = [(await stat(resolve(directory, 'manifest.json'))).mtimeMs];
        for (const [kind, mapping] of [['episodes', manifest.episodes], ['comparisons', manifest.comparisons]] as const) {
          for (const [key, path] of Object.entries(mapping)) {
            if (!key.includes('/') || typeof path !== 'string' || !new RegExp(`^${kind}/[^/]+\\.json\\.gz$`).test(path)) throw new Error(`Invalid ${kind} entry: ${key}`);
            const file = await safeFile(directory, path);
            const info = await stat(file);
            const fingerprint = `${file}:${key}:${info.mtimeMs}:${info.size}`;
            // Recheck changed payloads, without decompressing every export each poll.
            if (!validated.has(fingerprint)) {
              const payload = JSON.parse(gunzipSync(await readFile(file)).toString());
              if (payload.schema !== SCHEMA) throw new Error(`Unsupported payload schema: ${key}`);
              if (kind === 'episodes') {
                if (`${payload.example_id}/${payload.condition_id}` !== key) throw new Error(`Episode identity mismatch: ${key}`);
                for (const field of ['run_id', 'example_id', 'condition_id', 'checkpoint', 'tokenizer_id']) if (typeof payload[field] !== 'string') throw new Error(`Missing ${field}: ${key}`);
                for (const field of ['tokens', 'writes', 'reads', 'internal_prefixes']) if (!Array.isArray(payload[field])) throw new Error(`Missing ${field}: ${key}`);
                if (!payload.metadata || typeof payload.metadata !== 'object') throw new Error(`Missing metadata: ${key}`);
                validateEvents(payload);
              } else {
                if (payload.scope !== 'whole_query' || !['read_scale', 'memory_state'].includes(payload.intervention)) throw new Error(`Invalid comparison: ${key}`);
                checkFields(payload, { example_id: 'string', baseline_condition: 'string', variant_condition: 'string', identical_text_prefix: 'boolean', same_starting_kv: 'boolean', same_starting_memory: 'boolean', scored_position: 'number', scored_token_id: 'number', baseline_log_probability: 'number', variant_log_probability: 'number', difference_log_probability: 'number' });
                if (`${payload.example_id}/${payload.baseline_condition}/${payload.variant_condition}` !== key) throw new Error(`Comparison identity mismatch: ${key}`);
              }
              validated.add(fingerprint);
            }
            revisions.push(info.mtimeMs, info.size);
          }
        }
        index.runs.push({ id, rootName: id.replace(/\/memory_traces$/, ''), synthetic: id === 'synthetic-example/memory_traces', manifest, revision: createHash('sha256').update(revisions.join(':')).digest('hex') });
      } catch (error) { index.errors.push(`${id}: ${error instanceof Error ? error.message : String(error)}`); }
      return;
    }
    for (const entry of entries) if (entry.isDirectory()) await visit(resolve(directory, entry.name));
  }
  await mkdir(root, { recursive: true });
  await visit(root);
  index.runs.sort((a, b) => Number(a.synthetic) - Number(b.synthetic) || a.id.localeCompare(b.id));
  return index;
}

export function localTraces(root: string): Plugin {
  return {
    name: 'local-taal-traces',
    async configureServer(server) {
      await seedSynthetic(root);
      server.middlewares.use(async (request, response, next) => {
        const url = new URL(request.url ?? '/', 'http://localhost');
        if (!['/api/trace-runs', '/api/trace-file'].includes(url.pathname)) return next();
        response.setHeader('Cache-Control', 'no-store');
        try {
          if (request.method !== 'GET') throw new Error('Read-only endpoint');
          const index = await scanTraces(root);
          if (url.pathname === '/api/trace-runs') {
            response.setHeader('Content-Type', 'application/json');
            response.end(JSON.stringify(index));
          } else {
            const run = index.runs.find(item => item.id === url.searchParams.get('run'));
            const path = url.searchParams.get('path') ?? '';
            if (!run || ![...Object.values(run.manifest.episodes), ...Object.values(run.manifest.comparisons)].includes(path)) throw new Error('File not indexed by a valid manifest');
            response.setHeader('Content-Type', 'application/octet-stream');
            response.end(await readFile(await safeFile(resolve(root, run.id), path)));
          }
        } catch (error) {
          response.statusCode = 400;
          response.end(error instanceof Error ? error.message : String(error));
        }
      });
    },
  };
}
