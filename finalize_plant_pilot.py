import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'data/prepared/plant_care_pilot'


def read_jsonl(name):
    return [json.loads(line) for line in (FOLDER / name).read_text(encoding='utf-8').splitlines()]


def write_jsonl(name, rows):
    (FOLDER / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def main():
    global FOLDER
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use .venv/bin/python.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=FOLDER)
    arguments = parser.parse_args()
    FOLDER = arguments.folder.resolve()
    schema = json.loads((FOLDER / 'schema.json').read_text(encoding='utf-8'))
    target = schema['target_count']
    review_logs = [row for domain in ['indoor', 'garden'] for row in read_jsonl(domain + '_review_log.jsonl')]
    logs_by_id = {}
    for row in review_logs:
        logs_by_id.setdefault(row['case_id'], []).append(row)
    cases = read_jsonl('indoor_reviewed.jsonl') + read_jsonl('garden_reviewed.jsonl')
    if len(cases) != target or len({row['id'] for row in cases}) != target:
        raise ValueError(f'Expected {target} distinct reviewed cases.')
    if dict(Counter(case['domain'] for case in cases)) != schema['domain_targets']:
        raise ValueError('Domain counts differ from schema targets.')
    for case in cases:
        if case['review_status'] not in ['reviewed', 'revised']:
            raise ValueError(f'{case["id"]} has not been reviewed.')
        domain = case['domain']
        logs = logs_by_id.get(case['id'], [])
        if len(logs) != 1:
            raise ValueError(f'{case["id"]}: expected exactly one review record.')
        log = logs[0]
        annotation = case.get('annotation', {})
        if FOLDER.name != 'plant_care_pilot' and not annotation.get('writer', log.get('writer')):
            raise ValueError(f'{case["id"]}: actual writer metadata missing.')
        case['annotation'] = {
            **annotation,
            'method': 'AI translation, editing and annotation with reciprocal AI review',
            'writer': annotation.get('writer', log.get('writer', 'indoor_cases' if domain == 'indoor' else 'garden_cases')),
            'reviewer': log['reviewer'],
            'review_log': domain + '_review_log.jsonl',
        }
    cases.sort(key=lambda row: row['id'])
    write_jsonl('cases.jsonl', cases)
    sources = read_jsonl('source_records.jsonl')
    selected_ids = {case['source_question_id'] for case in cases}
    write_jsonl('selected_sources.jsonl', [source for source in sources if source['question_id'] in selected_ids])
    subprocess.run([sys.executable, str(ROOT / 'validate_plant_pilot.py'), '--folder', str(FOLDER)], cwd=ROOT, check=True)
    print(f"Prepared {len(cases)} cases and {len(selected_ids)} source records.")


if __name__ == '__main__':
    main()
