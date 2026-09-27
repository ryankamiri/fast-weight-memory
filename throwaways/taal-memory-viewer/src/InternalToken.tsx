import type { CSSProperties } from 'react';
import type { WriteEvent } from './trace';
import { intensity } from './calibration';

export function InternalToken({ index, write, mode, writeP99, selected, onSelect }: {
  index: number;
  write?: WriteEvent;
  mode: 'writes' | 'reads';
  writeP99: number | null;
  selected: boolean;
  onSelect: () => void;
}) {
  let status = '';
  let strength = 0;
  if (mode === 'reads') status = ' neutral';
  else if (!write) status = ' missing';
  else if (!write.write_enabled || write.proposed_write_norm === 0) status = ' neutral';
  else if (writeP99 !== null) strength = intensity(write.proposed_write_norm, writeP99);
  const style = { '--strength': strength, '--hue': '0 72% 53%' } as CSSProperties;
  return <button className={`token internal-token${status}${selected ? ' selected' : ''}`} style={style}
    title={mode === 'reads' ? 'Read output discarded' : !write ? 'Write not measured' : !write.write_enabled ? 'Write masked' : `Write magnitude: ${write.proposed_write_norm}`}
    onClick={onSelect}>P{index + 1}</button>;
}
