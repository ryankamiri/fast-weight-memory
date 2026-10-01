import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { scanTraces, seedSynthetic } from '../server/local-traces';
import { calibrateLayer, percentile } from './calibration';
import { demoEpisode, demoManifest } from './demo';
import { loadEpisode, splitEpisodeKey } from './trace';
import { exampleName, expectedAnswer, trajectoryTokens } from './answer';

const roots: string[] = [];
afterEach(async () => {
  vi.unstubAllGlobals();
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});
async function root() {
  const directory = await mkdtemp(resolve(tmpdir(), 'taal-viewer-test-'));
  roots.push(directory);
  return directory;
}
async function addExport(directory: string, name: string, manifest = demoManifest, payload: unknown = demoEpisode) {
  const run = resolve(directory, name, 'memory_traces');
  await mkdir(resolve(run, 'episodes'), { recursive: true });
  await writeFile(resolve(run, 'episodes/demo.json.gz'), gzipSync(JSON.stringify(payload)));
  await writeFile(resolve(run, 'manifest.json'), JSON.stringify(manifest));
}

describe('fixed local trace directory', () => {
  it('seeds one synthetic export and loads/calibrates through normal gzip fetches', async () => {
    const directory = await root();
    await seedSynthetic(directory);
    await seedSynthetic(directory);
    const index = await scanTraces(directory);
    expect(index.errors).toEqual([]);
    expect(index.runs).toHaveLength(1);
    expect(index.runs[0].synthetic).toBe(true);
    expect(index.runs[0].exampleLabels).toEqual({});
    vi.stubGlobal('fetch', vi.fn(async (input: string) => {
      const url = new URL(input, 'http://localhost');
      return new Response(await readFile(resolve(directory, url.searchParams.get('run')!, url.searchParams.get('path')!)));
    }));
    const folder = index.runs[0];
    expect((await loadEpisode(folder, 'example-01/correct_full')).tokens).toHaveLength(15);
    const calibration = await calibrateLayer(folder, 0, 'correct_full', 'synthetic-checkpoint');
    expect(calibration.referenceEpisodes).toBe(1);
    expect(percentile(calibration.writes, 0.81)).toBe(100);
  });
  it('discovers new exports and rejects unsupported, missing, malformed or escaping files', async () => {
    const directory = await root();
    await seedSynthetic(directory);
    await addExport(directory, 'real-run');
    expect((await scanTraces(directory)).runs.map(run => run.rootName)).toEqual(['real-run', 'synthetic-example']);
    await addExport(directory, 'wrong-schema', { ...demoManifest, schema: 'other' });
    await addExport(directory, 'missing', { ...demoManifest, episodes: { 'example-01/correct_full': 'episodes/missing.json.gz' } });
    await addExport(directory, 'malformed', demoManifest, { ...demoEpisode, reads: null });
    await addExport(directory, 'escaping', { ...demoManifest, episodes: { 'example-01/correct_full': '../other.json.gz' } });
    const index = await scanTraces(directory);
    expect(index.runs).toHaveLength(2);
    expect(index.errors).toHaveLength(4);
  });
  it('revalidates changed files and detects corrupt gzip', async () => {
    const directory = await root();
    await seedSynthetic(directory);
    expect((await scanTraces(directory)).runs).toHaveLength(1);
    await writeFile(resolve(directory, 'synthetic-example/memory_traces/episodes/demo.json.gz'), 'not gzip');
    const index = await scanTraces(directory);
    expect(index.runs).toHaveLength(0);
    expect(index.errors).toHaveLength(1);
  });
  it('accepts unavailable chunk movement and keeps compound conditions together', async () => {
    const directory = await root();
    const episode = {
      ...demoEpisode,
      condition_id: '1536/episode_off',
      writes: demoEpisode.writes.map(write => ({ ...write, other_movement_norm: null })),
      reads: demoEpisode.reads.map(read => ({ ...read, state_timing: 'pre-write' })),
    };
    const manifest = { ...demoManifest, episodes: { 'example-01/1536/episode_off': 'episodes/demo.json.gz' } };
    await addExport(directory, 'chunked-run', manifest, episode);
    const index = await scanTraces(directory);
    expect(index.errors).toEqual([]);
    expect(index.runs).toHaveLength(1);
    expect(splitEpisodeKey('example-01/1536/episode_off')).toEqual({ exampleId: 'example-01', condition: '1536/episode_off' });
  });
});

describe('answer labels', () => {
  it('prefers the exported answer and falls back to a known fact position', () => {
    const episode = {
      ...demoEpisode,
      metadata: { expected_answer: 'amber', fact_position: 4 },
    };
    expect(expectedAnswer(episode)).toBe('amber');
    expect(expectedAnswer({ ...episode, metadata: { fact_position: 4 } })).toBe('amber');
  });

  it('recovers the answer from an older TTCD trace without inventing one for unrelated text', () => {
    const text = "User: Remember that Record R0001's archive label is anchor.\nAssistant: Understood.\n\n";
    const episode = {
      ...demoEpisode,
      example_id: 'fact-0001-no_bridge-exact',
      tokens: [{ position: 0, token_id: 1, text, phase: 'prompt' as const }],
      metadata: {},
    };
    expect(expectedAnswer(episode)).toBe('anchor');
    expect(expectedAnswer({ ...episode, example_id: 'unrelated-example' })).toBeNull();
    expect(exampleName(episode.example_id, { [episode.example_id]: 'anchor' })).toBe('anchor');
  });

  it('shows a clickable greedy prediction for old exports without inventing trace events', () => {
    const older = {
      ...demoEpisode,
      outcome: { vocabulary_top_token_id: 7, vocabulary_top_token: ' anchor' },
    };
    const visible = trajectoryTokens(older);
    expect(visible.at(-1)).toEqual({
      position: demoEpisode.tokens.length,
      token_id: 7,
      text: ' anchor',
      phase: 'generated',
    });
    expect(older.writes.some(write => write.position === visible.at(-1)?.position)).toBe(false);
    const traced = { ...older, tokens: visible };
    expect(trajectoryTokens(traced)).toBe(visible);
  });
});
