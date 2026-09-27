import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { InternalToken } from './InternalToken';
import { demoEpisode } from './demo';

const write = demoEpisode.writes.find(event => event.position === null)!;
function render(mode: 'writes' | 'reads', event = write) {
  return renderToStaticMarkup(<InternalToken index={0} write={event} mode={mode} writeP99={0.24} selected={false} onSelect={() => {}} />);
}

describe('persistent token colors', () => {
  it('uses the same red hue and reference scale as text writes', () => {
    const html = render('writes');
    expect(html).toContain('--hue:0 72% 53%');
    expect(html).toContain('--strength:0.375');
    expect(html).not.toContain(' neutral');
  });
  it('keeps discarded reads neutral rather than presenting a measured zero', () => {
    const html = render('reads');
    expect(html).toContain(' neutral');
    expect(html).toContain('Read output discarded');
  });
  it('distinguishes missing and masked writes', () => {
    const html = renderToStaticMarkup(<InternalToken index={0} mode="writes" writeP99={0.24} selected={false} onSelect={() => {}} />);
    expect(html).toContain(' missing');
    expect(html).toContain('Write not measured');
    expect(render('writes', { ...write, write_enabled: false })).toContain('Write masked');
  });
});
