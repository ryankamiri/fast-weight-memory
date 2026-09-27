import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { conditionLabel, TraceSidebar } from './TraceSidebar';
import { demoManifest } from './demo';

describe('trace navigation', () => {
  it('lists every episode/condition as a clickable trace under its run', () => {
    const html = renderToStaticMarkup(<TraceSidebar runs={[{
      id: 'run/memory_traces', revision: '1', rootName: 'Pilot 1B', synthetic: false,
      manifest: { ...demoManifest, episodes: {
        'episode-01/correct_full': 'episodes/one.json.gz',
        'episode-01/reads_disabled': 'episodes/two.json.gz',
        'episode-02/swapped': 'episodes/three.json.gz',
      } },
    }]} selectedRun="run/memory_traces" selectedKey="episode-01/reads_disabled" onSelect={() => {}} />);
    expect(html.match(/class="trace-entry[ "]/g)).toHaveLength(3);
    expect(html.match(/aria-current="page"/g)).toHaveLength(1);
    expect(html).toContain('Correct memory · full reads');
    expect(html).toContain('Memory reads off');
    expect(html).toContain('Swapped memory');
    expect(html).not.toContain('<select');
  });

  it('keeps unfamiliar future condition names readable without assuming their meaning', () => {
    expect(conditionLabel('custom_condition')).toBe('custom condition');
    expect(conditionLabel('zeroed')).toBe('Zeroed memory');
  });
});
