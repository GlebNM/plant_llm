import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'data/prepared/plant_care_pilot'


def read_jsonl(path):
    records = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f'{path.name}:{number}: {error}') from error
    return records


def normalized(value):
    return re.sub(r'[^\w]+', ' ', value.casefold()).strip()


def main():
    global FOLDER
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use .venv/bin/python.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=FOLDER)
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--inputs', nargs='+')
    arguments = parser.parse_args()
    FOLDER = arguments.folder.resolve()
    schema = json.loads((FOLDER / 'schema.json').read_text(encoding='utf-8'))
    inputs = [FOLDER / name for name in (arguments.inputs or ['cases.jsonl'])]
    cases = [case for path in inputs for case in read_jsonl(path)]
    source_rows = read_jsonl(FOLDER / 'source_records.jsonl')
    sources = {source['question_id']: source for source in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError('Duplicate source question IDs in source_records.jsonl')
    reference_records = [record for name in ['indoor_reference_records.jsonl', 'garden_reference_records.jsonl'] if (FOLDER / name).exists() for record in read_jsonl(FOLDER / name)]
    recorded_reference_urls = {record.get('url') for record in reference_records}
    errors = []
    exclusions_path = FOLDER / 'exclusions.json'
    training = FOLDER.name == 'plant_care_train_1000'
    if training and not exclusions_path.exists():
        raise ValueError('Training collection requires exclusions.json')
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    excluded_questions = {normalized(question) for question in exclusions.get('excluded_questions', [])}
    old_sources = []
    if training:
        baseline = ROOT / 'data/prepared/plant_care_pilot'
        old_sources = read_jsonl(baseline / 'selected_sources.jsonl')
        pilot = read_jsonl(baseline / 'cases.jsonl')
        initial = read_jsonl(ROOT / 'data/evaluation/init_questions/questions.jsonl')
        excluded_questions.update(normalized(row['question']) for row in pilot + initial)
    excluded_originals = {source['question_html'] for source in old_sources}
    review_logs = [record for domain in ['indoor', 'garden'] if (FOLDER / f'{domain}_review_log.jsonl').exists() for record in read_jsonl(FOLDER / f'{domain}_review_log.jsonl')]
    domains = Counter()
    seen = {key: set() for key in ['id', 'source_question_id', 'source_answer_id', 'source_group_id', 'question']}
    for index, case in enumerate(cases, 1):
        identifier = case.get('id', f'row-{index}')
        domains[case.get('domain')] += 1
        def check(condition, message):
            if not condition:
                errors.append(f'{identifier}: {message}')
        for field in schema['required_fields']:
            check(field in case, f'missing {field}')
        for field in ['domain', 'topic', 'information_sufficiency', 'response_mode', 'review_status', 'origin']:
            value = case.get(field)
            permitted = schema['topics'] if field == 'topic' else schema[field]
            if field == 'response_mode':
                permitted = permitted.values()
            check(value in permitted, f'invalid {field}: {value}')
        for key in seen:
            value = normalized(case.get(key, '')) if key == 'question' else case.get(key)
            check(value not in seen[key], f'duplicate {key}: {value}')
            seen[key].add(value)
        for field, exclusion_key in [('source_question_id', 'excluded_source_question_ids'), ('source_answer_id', 'excluded_source_answer_ids'), ('source_group_id', 'excluded_source_group_ids')]:
            check(case.get(field) not in exclusions.get(exclusion_key, []), f'excluded {field}')
        check(normalized(case.get('question', '')) not in excluded_questions, 'question duplicates existing evaluation question')
        source = sources.get(case.get('source_question_id'))
        check(source is not None, 'unknown source question')
        if source:
            check(source['question_html'] not in excluded_originals, 'original source duplicates pilot')
            check(case.get('source_answer_id') == source['answer_id'], 'answer does not match source')
            check(case.get('source_group_id') == f'gardening.stackexchange.com:q:{source["question_id"]}', 'incorrect source group')
        sufficiency = case.get('information_sufficiency')
        missing = case.get('missing_information', [])
        check(isinstance(missing, list), 'missing_information must be a list')
        if isinstance(missing, list):
            for item in missing:
                check(isinstance(item, dict), 'missing_information entries must be objects')
                if isinstance(item, dict):
                    check(item.get('field') in schema['missing_information_fields'], f'unknown missing field {item.get("field")}')
                    check(isinstance(item.get('reason'), str) and len(item['reason']) >= 8, 'missing reason is too short')
            check((not missing) if sufficiency == 'sufficient' else bool(missing), 'missing information is inconsistent with sufficiency')
            check(len({item.get('field') for item in missing if isinstance(item, dict)}) == len(missing), 'duplicate missing field')
        check(case.get('response_mode') == schema['response_mode'].get(sufficiency), 'response mode does not match sufficiency')
        check(isinstance(case.get('sufficiency_reason'), str) and len(case.get('sufficiency_reason', '')) >= 15, 'sufficiency reason is missing or too short')
        check('split' not in case and 'dataset_split' not in case, 'split must not be assigned')
        check(isinstance(case.get('plants'), list), 'plants must be a list')
        tags = case.get('tags', [])
        check(isinstance(tags, list) and bool(tags), 'tags must be a nonempty list')
        if isinstance(tags, list):
            check(len(set(tags)) == len(tags), 'duplicate tags')
        fact_check = case.get('fact_check', {})
        level = fact_check.get('level')
        check(level in ['forum_only', 'reference_supported'], 'invalid factual-check level')
        references = fact_check.get('reference_urls', [])
        check(isinstance(references, list), 'reference_urls must be a list')
        if level == 'reference_supported':
            check(bool(references), 'reference-supported case has no reference')
        for reference in references:
            check(reference in recorded_reference_urls, f'reference evidence not preserved: {reference}')
        check(isinstance(fact_check.get('note'), str) and bool(fact_check.get('note', '').strip()), 'factual-check note is required')
        for field in ['question', 'answer']:
            value = case.get(field, '')
            check(isinstance(value, str) and len(value.strip()) >= 15, f'{field} is empty or too short')
            if not isinstance(value, str):
                continue
            check(bool(re.search('[А-Яа-яЁё]', value)), f'{field} lacks Russian text')
            check(not re.search(r'<\|.*?\|>|<think>|</think>', value), f'{field} has model control tokens')
            check(not re.search(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', value), f'{field} contains email address')
        if not arguments.draft:
            check(case.get('review_status') in ['reviewed', 'revised'], 'case has not completed reciprocal review')
            logs = [log for log in review_logs if log.get('case_id') == identifier]
            check(len(logs) == 1, 'expected exactly one reciprocal review record')
            if len(logs) == 1:
                log = logs[0]
                annotation = case.get('annotation', {})
                check(log.get('reviewer') == annotation.get('reviewer'), 'reviewer metadata mismatch')
                check(annotation.get('writer') and annotation.get('writer') != log.get('reviewer'), 'writer and reviewer must differ')
                check(log.get('verdict') == ('pass' if case.get('review_status') == 'reviewed' else 'revised'), 'review verdict mismatch')
                check(bool(log.get('checks')), 'review checks missing')
                for field in ['source_question_id', 'source_answer_id']:
                    if exclusions or field in log:
                        check(log.get(field) == case.get(field), f'review {field} mismatch')

    if not arguments.draft:
        if len(cases) != schema['target_count']:
            errors.append(f'Expected {schema["target_count"]} cases, found {len(cases)}')
        for domain, target in schema['domain_targets'].items():
            if domains[domain] != target:
                errors.append(f'Expected {target} {domain} cases, found {domains[domain]}')

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Validated {len(cases)} cases.")



if __name__ == '__main__':
    main()
