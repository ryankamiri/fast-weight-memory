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
