import type { Episode, Manifest, TraceToken, WriteEvent, ReadEvent } from './trace.ts';
import { SCHEMA } from './trace.ts';

const fragments = ['The', ' code', ' word', ' is', ' amber', '.', ' ', 'Later', ',', ' what', ' was', ' the', ' code', ' word', '?'];
const tokens: TraceToken[] = fragments.map((text, position) => ({ position, token_id: 1000 + position, text, phase: 'prompt' }));
const writes: WriteEvent[] = [];
const reads: ReadEvent[] = [];

for (const layer of [0, 1]) {
  for (const token of tokens) {
    const proposed = token.position === 4 ? 0.81 + layer * 0.12 : token.position === 2 ? 0.24 : 0.015 + (token.position % 3) * 0.008;
    writes.push({
      layer, position: token.position, segment_index: 0, internal_index: null,
      write_enabled: true, write_strength: 0.5, proposed_write_norm: proposed,
      previous_weight_norm: 12.4, net_weight_change_norm: proposed * 0.77,
      other_movement_norm: proposed * 0.24, write_to_net_alignment: 0.91,
    });
    reads.push({
      layer, position: token.position, segment_index: 0, enabled: true,
      read_scale: 1, residual_gate: 0.16, injected_norm: token.position >= 9 ? 0.14 + layer * 0.03 : 0.018,
      incoming_norm: 5.2, relative_injection: token.position >= 9 ? (0.14 + layer * 0.03) / 5.2 : 0.018 / 5.2,
      state_timing: 'post-write',
    });
  }
  for (let internal_index = 0; internal_index < 3; internal_index++) {
    writes.push({
      layer, position: null, segment_index: 0, internal_index,
      write_enabled: true, write_strength: 0.5, proposed_write_norm: 0.09 + internal_index * 0.04,
      previous_weight_norm: 12.3, net_weight_change_norm: 0.08 + internal_index * 0.03,
      other_movement_norm: 0.02, write_to_net_alignment: 0.88,
    });
  }
}

export const demoEpisode: Episode = {
  schema: SCHEMA, run_id: 'synthetic-demo', example_id: 'example-01', condition_id: 'correct_full',
  checkpoint: 'synthetic-checkpoint', tokenizer_id: 'synthetic-tokenizer', tokens, writes, reads,
  internal_prefixes: [0, 1].map(layer => ({ layer, segment_index: 0, before_position: 0, count: 3, net_weight_change_norm: 0.19 })),
  outcome: null,
  metadata: { note: 'Synthetic UI example. No model was run.' },
};

export const demoManifest: Manifest = {
  schema: SCHEMA,
  run_metadata: { source: 'synthetic UI example' },
  episodes: { 'example-01/correct_full': 'episodes/demo.json.gz' },
  comparisons: {},
};
