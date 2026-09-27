# TaaL Memory Viewer

Local React prototype for `taal-memory-trace/v1`. It shows per-token proposed writes and injected reads from an exported TaaL evaluation. Nothing is uploaded.

```bash
cd throwaways/taal-memory-viewer
npm install
npm run dev
```

Open the local URL. The viewer automatically scans the fixed `data/` directory every three seconds. Place exports in `data/<run>/memory_traces/`, containing `manifest.json`, `episodes/*.json.gz`, and optionally `comparisons/*.json.gz`. There is no folder picker. Invalid or incomplete exports are reported instead of loaded, and changed files are revalidated. Color scales are derived from every episode in the reference condition (normally `correct_full`). For large exports, initial validation and calibration may take a while.

The dev server automatically creates one synthetic export in `data/synthetic-example/memory_traces/`. It passes through the same discovery and contract checks as real exports. It is test data, not a model result. This prototype does not display full fast-weight tensors or claim that a large read improved an answer. Comparison cards appear only at the scored prediction position, when a paired record exists.

Local trace data belongs in `data/`, which is ignored by Git. The `explorer-taal-traces` skill can copy a completed Explorer export here.

The sidebar groups clickable traces by run. Each trace is one episode evaluated under one condition, such as correct memory with full reads or reads disabled. The same episode can therefore appear several times for comparison. Search filters these entries; run headings collapse their lists. Memory layer selection stays in the main view because layers are different signals within the same trace, not separate evaluations.
