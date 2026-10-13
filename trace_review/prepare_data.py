"""Audit TRACE splits and optionally create separate cleaned copies."""
import argparse
import json
from pathlib import Path
from collections import Counter
from train import read_jsonl, validate_splits


def key(row):
    return ' '.join(row['text'].split()).casefold()


def prepare(source, output, exclude_validation_duplicates=False):
    rows = {n: read_jsonl(Path(source) / f'{n}.jsonl') for n in ('train', 'val', 'test')}
    val = {key(r): r for r in rows['val']}
    overlaps = [{'validation_id': val[key(r)]['id'], 'test_id': r['id'],
                 'same_labels': sorted(val[key(r)]['labels']) == sorted(r['labels'])}
                for r in rows['test'] if key(r) in val]
    report = {'original_split_sizes': {n: len(v) for n, v in rows.items()},
              'training_support': dict(Counter(label for r in rows['train'] for label in r['labels'])),
              'validation_test_normalized_overlap': overlaps}
    excluded = []
    if exclude_validation_duplicates:
        test_keys = {key(r) for r in rows['test']}
        excluded = [r['id'] for r in rows['val'] if key(r) in test_keys]
        rows['val'] = [r for r in rows['val'] if key(r) not in test_keys]
    validate_splits(rows)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, records in rows.items():
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as stream:
            for row in records:
                stream.write(json.dumps({k: v for k, v in row.items() if k != 'date_ord'}, ensure_ascii=False) + '\n')
    report.update(excluded_validation_ids=excluded, final_split_sizes={n: len(v) for n, v in rows.items()})
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclude-validation-duplicates', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.exclude_validation_duplicates), indent=2))
