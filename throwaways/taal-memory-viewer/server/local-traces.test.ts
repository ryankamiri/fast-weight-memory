import { afterEach, expect, it } from 'vitest';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { demoEpisode, demoManifest } from '../src/demo';
import { readTimingNotes } from '../src/trace';
import { scanTraces } from './local-traces';

const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

it.each(['pre-write', 'post-write', 'substituted', 'invalid'])('validates read timing %s', async timing => {
  const root = await mkdtemp(resolve(tmpdir(), 'taal-timing-test-'));
  roots.push(root);
  const run = resolve(root, 'test/memory_traces');
  await mkdir(resolve(run, 'episodes'), { recursive: true });
  const episode = {
    ...demoEpisode,
    reads: demoEpisode.reads.map(read => ({ ...read, state_timing: timing })),
  };
  await writeFile(resolve(run, 'manifest.json'), JSON.stringify(demoManifest));
  await writeFile(resolve(run, 'episodes/demo.json.gz'), gzipSync(JSON.stringify(episode)));
  const index = await scanTraces(root);
  expect(index.runs).toHaveLength(timing === 'invalid' ? 0 : 1);
  expect(index.errors).toHaveLength(timing === 'invalid' ? 1 : 0);
  expect(readTimingNotes.substituted).toContain('unchanged background');
});

it.each([
  ['whole_query', 'read_scale', true],
  ['whole_query', 'memory_state', true],
  ['whole_query', 'memory_read', true],
  ['prediction_token', 'memory_read', true],
  ['unknown_scope', 'memory_read', false],
  ['whole_query', 'unknown_intervention', false],
])('validates comparison scope %s and intervention %s', async (scope, intervention, valid) => {
  const root = await mkdtemp(resolve(tmpdir(), 'taal-comparison-test-'));
  roots.push(root);
  const run = resolve(root, 'test/memory_traces');
  await mkdir(resolve(run, 'episodes'), { recursive: true });
  await mkdir(resolve(run, 'comparisons'));
  const comparison = {
    schema: demoManifest.schema,
    example_id: 'example-01', baseline_condition: 'correct_full', variant_condition: 'crossed',
    scope, intervention, identical_text_prefix: true, same_starting_kv: true,
    same_starting_memory: true, scored_position: 14, scored_token_id: 1,
    baseline_log_probability: -0.1, variant_log_probability: -1.1, difference_log_probability: 1,
  };
  const manifest = {
    ...demoManifest,
    comparisons: { 'example-01/correct_full/crossed': 'comparisons/crossed.json.gz' },
  };
  await writeFile(resolve(run, 'manifest.json'), JSON.stringify(manifest));
  await writeFile(resolve(run, 'episodes/demo.json.gz'), gzipSync(JSON.stringify(demoEpisode)));
  await writeFile(resolve(run, 'comparisons/crossed.json.gz'), gzipSync(JSON.stringify(comparison)));
  const index = await scanTraces(root);
  expect(index.runs).toHaveLength(valid ? 1 : 0);
  expect(index.errors).toHaveLength(valid ? 0 : 1);
});
