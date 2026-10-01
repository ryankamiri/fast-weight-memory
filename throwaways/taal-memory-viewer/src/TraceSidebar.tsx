import { useMemo, useState } from 'react';
import type { TraceFolder } from './trace';
import { splitEpisodeKey } from './trace';
import { exampleName } from './answer';

export function conditionLabel(condition: string): string {
  const labels: Record<string, string> = {
    correct_full: 'Correct memory · full reads',
    correct_half: 'Correct memory · half reads',
    reads_off: 'Memory reads off',
    reads_disabled: 'Memory reads off',
    reset_initial: 'Reset to initial memory',
    zeroed: 'Zeroed memory',
    swapped: 'Swapped memory',
  };
  return labels[condition] ?? condition.replaceAll('_', ' ');
}

export function TraceSidebar({ runs, selectedRun, selectedKey, onSelect }: {
  runs: TraceFolder[];
  selectedRun: string;
  selectedKey: string;
  onSelect: (run: string, key: string) => void;
}) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState('');
  const groups = useMemo(() => runs.map(run => ({
    run,
    keys: Object.keys(run.manifest.episodes)
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
      .filter(key => `${run.rootName} ${key} ${exampleName(splitEpisodeKey(key).exampleId, run.exampleLabels)} ${conditionLabel(splitEpisodeKey(key).condition)}`.toLowerCase().includes(search.toLowerCase())),
  })).filter(group => group.keys.length > 0), [runs, search]);

  function toggle(id: string) {
    setCollapsed(previous => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  return <nav className="trace-navigation" aria-label="Traces">
    <input className="trace-search" aria-label="Search traces" placeholder="Search traces" value={search} onChange={event => setSearch(event.target.value)} />
    <div className="trace-nav-heading">Traces</div>
    {groups.map(({ run, keys }) => <section className="trace-run" key={run.id}>
      <button className="trace-run-heading" onClick={() => toggle(run.id)} aria-expanded={search !== '' || !collapsed.has(run.id)}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" /></svg>
        <span>{run.synthetic ? 'Synthetic example' : run.rootName}</span>
        <span className="trace-chevron" aria-hidden="true">{search !== '' || !collapsed.has(run.id) ? '⌄' : '›'}</span>
      </button>
      {(search !== '' || !collapsed.has(run.id)) && <div className="trace-entries">{keys.map(key => {
        const { exampleId, condition } = splitEpisodeKey(key);
        const active = run.id === selectedRun && key === selectedKey;
        return <button key={key} className={`trace-entry${active ? ' active' : ''}`} aria-current={active ? 'page' : undefined} onClick={() => onSelect(run.id, key)} title={`${exampleId} / ${condition}`}>
          <span className="trace-entry-title">{exampleName(exampleId, run.exampleLabels)}</span>
          <span className="trace-entry-condition">{conditionLabel(condition)}</span>
        </button>;
      })}</div>}
    </section>)}
    {groups.length === 0 && <p className="sidebar-hint">{runs.length ? 'No matching traces.' : 'Checking local traces…'}</p>}
    <p className="sidebar-hint trace-nav-note">Auto-sync from <code>data/</code>{runs.some(run => run.synthetic) && <><br />Synthetic example is test data.</>}</p>
  </nav>;
}
