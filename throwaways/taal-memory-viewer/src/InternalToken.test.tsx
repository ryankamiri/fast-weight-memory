import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { InternalToken } from './InternalToken';
import type { ActivitySummary } from './activity';

const measured: ActivitySummary = { status: 'measured', strength: 0.375, measuredLayers: 2, unusuallyActiveLayers: 0 };
function render(mode: 'writes' | 'reads', activity = measured) {
  return renderToStaticMarkup(<InternalToken index={0} activity={activity} mode={mode} selected={false} onSelect={() => {}} />);
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
    const html = render('writes', { ...measured, status: 'missing' });
    expect(html).toContain(' missing');
    expect(html).toContain('Write not measured');
    expect(render('writes', { ...measured, status: 'disabled' })).toContain('Write masked');
  });
});
