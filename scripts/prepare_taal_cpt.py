"""Build pinned, document-isolated pilot corpora and paired held-out probes."""

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from data.prolong import stream_prolong

ROOT = Path('data/taal_cpt')
REVISION = '1447128c751f0fbc7d2082b2a113a938426a9bba'
TOKENIZER = 'Qwen/Qwen3-0.6B-Base'
TOKENIZER_REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
SOURCES = ['book-65536', 'thestackv1_concat_by_repo-65536', 'fineweb-edu',
           'fineweb-2023-50', 'dolmawiki', 'tuluv2', 'stackexchange', 'arxiv', 'openwebmath']
SHORT = ['fineweb-edu'] * 5 + ['fineweb-2023-50'] * 5 + ['dolmawiki', 'tuluv2', 'stackexchange', 'arxiv', 'openwebmath'] * 2
LENGTHS = [256, 512, 256, 1024]


def dump(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def pool(source, decoder, tokenizer):
    path = ROOT / f'pool-{source}.jsonl'
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    rows = []
    seen = set()
    for record_index, record in enumerate(stream_prolong(source, REVISION)):
        for document_index, (start, end) in enumerate(record['indices']):
            text = decoder.decode(record['input_ids'][start:end], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 1100:
                continue
            rows.append({'input_ids': ids[:4096], 'document_sha256': digest,
                         'source': source, 'source_record': record_index,
                         'document_index': document_index, 'original_tokens': len(ids)})
            if len(rows) == 100:
                dump(path, rows)
                print('PREPARED', source, len(rows), flush=True)
                return rows
    raise ValueError(f'Insufficient documents in {source}')


def retrieval(tokenizer, filler, length, index, answer, pair, heldout):
    name = f'P{pair:04d}' if heldout else f'T{pair:04d}'
    template = 'Remember that' if not heldout else 'The stored fact is that'
    head = tokenizer.encode(f'User: {template} Record {name}\'s archive label is {answer}.\nAssistant: Understood.\n', add_special_tokens=False)
    query = tokenizer.encode(f'\nUser: What is Record {name}\'s archive label?\nAssistant: The archive label is', add_special_tokens=False)
    target = tokenizer.encode(' ' + answer, add_special_tokens=False)
    candidates = [tokenizer.encode(' ' + word, add_special_tokens=False)[0] for word in ('amber', 'anchor')]
    if len(target) != 1:
        raise ValueError('Expected one answer token')
    count = length - 1 - len(head) - len(query)
    if count < 64 or len(filler) < count:
        raise ValueError('Retrieval filler must fit with complete KV eviction')
    ids = head + filler[:count] + query
    return {'example_id': f'cpt-{index:04d}', 'fact_id': index, 'record_name': name,
            'answer': answer, 'condition': 'cpt_heldout', 'query_variant': 'exact',
            'input_ids': ids, 'final_query_position': len(ids) - len(query),
            'final_answer_position': len(ids), 'prompt_length': len(ids),
            'target_token_id': target[0], 'candidate_token_ids': candidates}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    decoder = AutoTokenizer.from_pretrained('princeton-nlp/Llama-3-8B-ProLong-64k-Base', revision='ee4b1380059c70aaeab703934ca669f368cc7905')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    pools = {source: pool(source, decoder, tokenizer) for source in SOURCES}
    # All arms share held-out documents; no validation document enters any training arm.
    val = []
    for source in SOURCES:
        row = pools[source][0]
        val.append(row | {'input_ids': row['input_ids'][:256], 'source_record_index': 80 + len(val)})
    short_index = 0
    schedules = {arm: [] for arm in 'ABCD'}
    for index in range(80):
        length = LENGTHS[index // 20]
        slot = index % 10
        if slot < 4:
            common = SHORT[short_index % len(SHORT)]
            short_index += 1
            b_source = c_source = common
        else:
            b_source = SOURCES[0]
            c_source = SOURCES[0] if slot < 7 else SOURCES[1]
        for arm, source in [('A', 'fineweb-edu'), ('B', b_source), ('C', c_source), ('D', c_source)]:
            document = pools[source][index + 10]
            ids = document['input_ids'][:length]
            row = document | {'input_ids': ids, 'labels': ids.copy(), 'source_record_index': index, 'objective': 'causal_lm'}
            if arm == 'D' and index % 4 == 3:
                answer = ('amber', 'anchor')[(index // 4) % 2]
                episode = retrieval(tokenizer, document['input_ids'], length, index, answer, index // 8, False)
                ids = episode['input_ids'] + [episode['target_token_id']]
                row.update(input_ids=ids, labels=[-100] * (len(ids) - 1) + ids[-1:], objective='retrieval')
            schedules[arm].append(row)
    for arm, rows in schedules.items():
        dump(ROOT / f'{arm}.jsonl', rows + val)
    probes = []
    for pair, length in enumerate((256, 512, 1024, 2048)):
        filler = pools[SOURCES[pair % 3]][1]['input_ids']
        for answer in ('amber', 'anchor'):
            probes.append(retrieval(tokenizer, filler, length, len(probes), answer, pair, True))
    dump(ROOT / 'retrieval.jsonl', probes)
    metadata = {'tokenizer': TOKENIZER, 'tokenizer_revision': TOKENIZER_REVISION}
    (ROOT / 'retrieval.metadata.json').write_text(json.dumps(metadata, indent=2))
    manifest = {'source_repo': 'princeton-nlp/prolong-data-64K', 'source_revision': REVISION,
                **metadata, 'length_schedule': LENGTHS, 'updates': 80,
                'filter': 'documents >=1100 Qwen tokens, first 100 unique documents per source; length-conditioned pilot',
                'split': 'heldout source documents 0-9; training documents 10-89',
                'limitation': 'Source document hashes prevent exact duplicates; source book/repository identity is unavailable in packed release. This is not a guaranteed book-disjoint generalization benchmark.',
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.jsonl')}}
    train_hashes = {row['document_sha256'] for rows in schedules.values() for row in rows}
    eval_hashes = {row['document_sha256'] for source in SOURCES for row in pools[source][:10]}
    if train_hashes & eval_hashes:
        raise ValueError('Cross-source train/eval duplicate documents')
    manifest['arms'] = {}
    for arm, rows in schedules.items():
        counts = {}
        for row in rows:
            key = row['source'] + '/' + row['objective']
            counts[key] = counts.get(key, 0) + len(row['input_ids'])
        manifest['arms'][arm] = {'tokens': sum(len(r['input_ids']) for r in rows), 'supervised_tokens': sum(sum(v != -100 for v in r['labels'][1:]) for r in rows), 'source_tokens': counts}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['arms'], indent=2), flush=True)


if __name__ == '__main__':
    main()
