import type { CSSProperties } from 'react';
import type { ActivitySummary, Mode } from './activity';

export function InternalToken({ index, activity, mode, selected, onSelect }: {
  index: number;
  activity: ActivitySummary;
  mode: Mode;
  selected: boolean;
  onSelect: () => void;
}) {
  const status = mode === 'reads' ? ' neutral'
    : activity.status === 'missing' ? ' missing'
    : activity.status === 'disabled' || activity.strength === 0 ? ' neutral' : '';
  const style = { '--strength': mode === 'reads' ? 0 : activity.strength, '--hue': '0 72% 53%' } as CSSProperties;
  return <button className={`token internal-token${status}${selected ? ' selected' : ''}`} style={style}
    title={mode === 'reads' ? 'Read output discarded' : activity.status === 'missing' ? 'Write not measured' : activity.status === 'disabled' ? 'Write masked' : `Writes measured at ${activity.measuredLayers} layers`}
    onClick={onSelect}>P{index + 1}</button>;
}
