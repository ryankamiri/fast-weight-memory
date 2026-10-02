# TaaL state attribution audits

`state_audit.py` evaluates saved pre-query state while retaining each recipient's
own bounded KV cache. The default audit keeps its six existing controls. Additive
configuration settings support component attribution without changing training:

- `audit_suite: standard`: correct/full, correct/half, query reads-off, learned
  initial state, zeroed state, and complete donor-state swap.
- `audit_suite: weights_vs_rest`: correct/full, reads-off, donor weights only,
  donor momentum plus convolution histories, and complete donor-state swap.
- `audit_suite: factorial`: every own/donor combination of weights, momentum,
  and q/k/v convolution history, plus query reads-off (nine conditions).
- `persistent_writes_enabled: false`: mask only fresh persistent-token writes
  during prefix construction. Retain their inputs, convolution processing,
  forgetting/momentum transitions, and all text-token writes and reads.
- `query_updates_enabled: false`: keep weights and momentum fixed throughout
  the query and generated-token trace. Reads and convolution processing continue.
  Traces retain proposed gradients, mark `updates_enabled: false`, and record
  zero committed movement. This is not equivalent to masking fresh gradients.

All supplied attribution configurations use `save_traces: true`. Component and
fixed-weight controls require one-token memory chunks and an empty pending-write
buffer at the intervention boundary. Each experiment builds its own prefix bank;
snapshots are fingerprinted by checkpoint contents and evaluation configuration.
Within an experiment, partial states clone their selected components and always
retain the recipient's KV, positions, and query. Results and trace metadata record
component provenance, prefix policy, and query-update policy. Summaries include
recipient-versus-donor label margins and paired record-group shifts.

## Explorer execution

Use the live interactive cmux Explorer login shell, not non-interactive SSH
submission. Verify a clean synchronized checkout, the intended checkpoint and
16-example dataset, live partition limits, and cache/proxy variables first.
The launcher checks the exact submitted commit and performs an allocated-node
preflight before loading the model. Prepare the dataset before submitting jobs,
not concurrently from several workers.

```bash
sbatch evaluation/sbatch/taal/fs_qwen_conflict_carriers_short.sbatch \
  evaluation/configs/taal/conflict_component_split.yaml \
  CHECKPOINT_DIRECTORY UNIQUE_OUTPUT_DIRECTORY FULL_GIT_COMMIT
```

The launcher requests one H200 on `gpu-short`, with a two-hour limit. Four configs
cover persistent writes off, the broad component split, the full component
factorial, and the split with fixed query weights. These are checkpoint audits;
no new training is required. Verify scheduler acceptance, startup preflight/model
loading, and final score/trace counts separately. A submitted job is not a result.

Partial swaps can produce incompatible states. A coherent bidirectional shift
toward the donor's label is more informative than nonspecific degradation. These
controls identify causal dependencies, not uniquely decoded memory contents.

## Acquisition, retention, and query-transition audits

Three additional suites reuse the same state-audit runner and compact trace format:

- `encoding`: snapshots immediately after the label, after the complete fact,
  and before the question. Fixed-query own/donor contrasts test whether an
  episode-dependent label preference is already accessible. An initial-memory
  control and evolving-query early-state controls distinguish shared task bias
  from state-dependent behavior.
- `retention`: compare after-fact, mid-gap, and pre-query snapshots, plus a
  trajectory that freezes weights and momentum through the neutral gap. Read
  each snapshot under both own/donor controls; query updates are fixed for the
  main time-course comparison. The frozen-gap snapshot is also tested with
  normal query updates.
- `query_transitions`: all eight enabled/disabled combinations of fresh query
  writes, carried momentum, and forgetting, each with own and matched donor
  state, plus reads-off and fully fixed-state controls. Fresh-write masking
  zeros the write loss; disabling momentum removes only `eta * S_previous`;
  disabling forgetting sets `alpha = 0`. New gradients still form momentum when
  carried momentum is disabled. The checkpoint parameters are never changed.

All staged-memory comparisons retain the normal recipient's final KV, absolute
positions, final convolution histories, and exact question. Only weights and
momentum come from the earlier/donor snapshot. This avoids visible-fact KV and
old convolution-history confounds. The normal/full-swap endpoints remain full
state controls. Lifecycle prefix execution splits at recorded boundaries, so
check their normal endpoints against the prior baseline for numerical drift.
Persistent inputs are prepended once, not once per snapshot.

The exported prefix is the normal recipient reference trajectory. At the query
boundary it may be replaced by an earlier, donor, or frozen-gap state; metadata
explicitly records that discontinuity. No earlier state is falsely presented as
the uninterrupted result of the reference prefix. Scalar proposed writes still
do not establish faithful semantic storage. Negative early-state retrieval does
not exclude information that this checkpoint's reader cannot access. Retention
contrasts identify changes in accessible signal, not a unique storage capacity.

Use `fs_qwen_conflict_lifecycle_short.sbatch` with one of
`conflict_encoding.yaml`, `conflict_retention.yaml`, or
`conflict_query_transitions.yaml`. The launcher requests one gpu-short H200 and
the live two-hour partition maximum, checks the submitted commit, and runs the
allocated-node preflight. All three configurations save every example's traces.
No retraining is required for these interventions.
