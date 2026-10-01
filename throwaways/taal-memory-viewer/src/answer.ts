import type { Episode, TraceToken } from './trace.ts';

function nonemptyText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

export function expectedAnswer(episode: Episode): string | null {
  const explicit = nonemptyText(episode.metadata.expected_answer)
    ?? nonemptyText(episode.outcome?.expected_answer);
  if (explicit) return explicit;

  const factPosition = episode.metadata.fact_position;
  if (typeof factPosition === 'number') {
    const factToken = episode.tokens.find(token => token.position === factPosition);
    if (factToken) return nonemptyText(factToken.text);
  }

  // Older TTCD exports did not include the answer in metadata. Recover it
  // from this example's source fact, never from a prediction or arbitrary filler.
  const factId = /^fact-(\d{4})-/.exec(episode.example_id)?.[1];
  if (!factId) return null;
  const text = episode.tokens.map(token => token.text).join('');
  const fact = new RegExp(`User: Remember that Record R${factId}'s archive label is\\s+([A-Za-z]+)\\.`, 'i');
  return fact.exec(text)?.[1] ?? null;
}

export function exampleName(exampleId: string, labels?: Record<string, string>): string {
  return labels?.[exampleId] ?? exampleId;
}

export function trajectoryTokens(episode: Episode): TraceToken[] {
  if (episode.tokens.some(token => token.phase === 'generated')) return episode.tokens;
  const tokenId = episode.outcome?.vocabulary_top_token_id;
  if (typeof tokenId !== 'number') return episode.tokens;
  const decoded = episode.outcome?.vocabulary_top_token;
  const text = typeof decoded === 'string' ? decoded : `⟨${tokenId}⟩`;
  return [
    ...episode.tokens,
    {
      position: (episode.tokens.at(-1)?.position ?? -1) + 1,
      token_id: tokenId,
      text,
      phase: 'generated',
    },
  ];
}
