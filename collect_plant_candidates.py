import argparse
import html
import re
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data/prepared/plant_care_pilot'
QUERIES = [
    ('indoor', 'houseplants', 1, 100),
    ('indoor', 'houseplants', 2, 100),
    ('indoor', 'indoors', 1, 100),
    ('indoor', 'succulents', 1, 75),
    ('indoor', 'orchids', 1, 50),
    ('garden', 'vegetables', 1, 100),
    ('garden', 'seedlings', 1, 75),
    ('garden', 'pruning', 1, 75),
    ('garden', 'fruit-trees', 1, 75),
    ('garden', 'compost', 1, 50),
    ('garden', 'soil', 1, 75),
]


def request(endpoint, parameters):
    url = 'https://api.stackexchange.com/2.3/' + endpoint + '?' + urllib.parse.urlencode(parameters)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PlantCareEducationalDataset/1.0'})
            with urllib.request.urlopen(req, timeout=40) as response:
                result = json.load(response)
            if result.get('error_id'):
                raise RuntimeError(str(result))
            if result.get('backoff'):
                time.sleep(min(result['backoff'], 45))
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def author(record):
    owner = record.get('owner', {})
    return {key: owner[key] for key in ('display_name', 'link') if key in owner}


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use .venv/bin/python.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    question_cache = OUTPUT / 'candidate_questions.json'
    if args.training:
        exclusions = set(json.loads((OUTPUT / 'exclusions.json').read_text())['excluded_source_question_ids'])
        questions_by_id = {}
        for cache in (ROOT / 'data/prepared/plant_care_pilot/candidate_questions.json', question_cache):
            if cache.exists():
                for question in json.loads(cache.read_text()):
                    if question['question_id'] not in exclusions:
                        questions_by_id.setdefault(question['question_id'], question)
        progress_path = OUTPUT / 'api_progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed': [], 'quota_remaining': None}
        training_queries = [('indoor', tag, page, 100) for tag in ('houseplants', 'indoors', 'succulents', 'orchids', 'cacti', 'containers') for page in range(1, 9)]
        training_queries += [('garden', tag, page, 100) for tag in ('vegetables', 'seedlings', 'pruning', 'fruit-trees', 'soil', 'watering', 'lawn', 'tomatoes') for page in range(1, 5)]
        exhausted = set()
        for domain, tag, page, page_size in training_queries:
            key = f'{domain}/{tag}/{page}'
            if key in progress['completed'] or tag in exhausted:
                continue
            result = request('search/advanced', {
                'site': 'gardening', 'tagged': tag, 'accepted': 'true',
                'sort': 'votes', 'order': 'desc', 'min': 1,
                'page': page, 'pagesize': page_size, 'filter': 'withbody',
            })
            for question in result['items']:
                if question.get('closed_date') or not question.get('accepted_answer_id') or question['question_id'] in exclusions:
                    continue
                question['candidate_domain'] = domain
                questions_by_id.setdefault(question['question_id'], question)
            question_cache.write_text(json.dumps(list(questions_by_id.values()), ensure_ascii=False), encoding='utf-8')
            progress['completed'].append(key)
            progress['quota_remaining'] = result.get('quota_remaining')
            progress_path.write_text(json.dumps(progress), encoding='utf-8')
            print(f'{key}: {len(result["items"])} results; {len(questions_by_id)} unique; quota {result.get("quota_remaining")}', flush=True)
            if not result.get('has_more'):
                exhausted.add(tag)
            time.sleep(0.4)
        questions = list(questions_by_id.values())
        pilot_sources = ROOT / 'data/prepared/plant_care_pilot/source_records.jsonl'
        source_path = OUTPUT / 'source_records.jsonl'
        if not source_path.exists():
            with source_path.open('w', encoding='utf-8') as destination:
                for line in pilot_sources.read_text().splitlines():
                    row = json.loads(line)
                    if row['question_id'] in questions_by_id:
                        destination.write(line + '\n')
    elif question_cache.exists():
        questions = json.loads(question_cache.read_text(encoding='utf-8'))
        print(f'Reusing {len(questions)} cached questions.', flush=True)
    else:
        questions_by_id = {}
        for domain, tag, page, page_size in QUERIES:
            result = request('search/advanced', {
                'site': 'gardening', 'tagged': tag, 'accepted': 'true',
                'sort': 'votes', 'order': 'desc', 'min': 1,
                'page': page, 'pagesize': page_size, 'filter': 'withbody',
            })
            for question in result['items']:
                if question.get('closed_date') or not question.get('accepted_answer_id'):
                    continue
                question.setdefault('candidate_domain', domain)
                questions_by_id.setdefault(question['question_id'], question)
            print(f'{domain}/{tag}/{page}: {len(result["items"])} results; {len(questions_by_id)} unique; quota {result.get("quota_remaining")}', flush=True)
            time.sleep(0.4)
        questions = list(questions_by_id.values())
        question_cache.write_text(json.dumps(questions, ensure_ascii=False), encoding='utf-8')

    source_path = OUTPUT / 'source_records.jsonl'
    existing = {}
    if source_path.exists():
        existing = {row['question_id']: row for line in source_path.read_text(encoding='utf-8').splitlines() if (row := json.loads(line))}
    pending = [q for q in questions if q['question_id'] not in existing]
    with source_path.open('a', encoding='utf-8') as destination:
        for offset in range(0, len(pending), 100):
            batch = pending[offset:offset + 100]
            answer_ids = ';'.join(str(q['accepted_answer_id']) for q in batch)
            result = request('answers/' + answer_ids, {'site': 'gardening', 'pagesize': 100, 'filter': 'withbody'})
            if args.training:
                progress['quota_remaining'] = result.get('quota_remaining')
                progress_path.write_text(json.dumps(progress), encoding='utf-8')
            answers = {a['answer_id']: a for a in result['items']}
            for q in batch:
                a = answers.get(q['accepted_answer_id'])
                if not a or a.get('score', 0) < 1:
                    continue
                q_body, a_body = q['body'], a['body']
                row = {
                    'question_id': q['question_id'], 'answer_id': a['answer_id'],
                    'candidate_domain': q['candidate_domain'], 'title': q['title'],
                    'question_html': q_body, 'answer_html': a_body,
                    'question_url': q['link'], 'answer_url': 'https://gardening.stackexchange.com/a/' + str(a['answer_id']),
                    'tags': q['tags'], 'question_score': q['score'], 'answer_score': a['score'],
                    'question_author': author(q), 'answer_author': author(a),
                    'question_license': q.get('content_license'), 'answer_license': a.get('content_license'),
                    'question_creation_date': q['creation_date'], 'answer_creation_date': a['creation_date'],
                    'retrieved_at': datetime.now(timezone.utc).isoformat(),
                }
                destination.write(json.dumps(row, ensure_ascii=False) + '\n')
                existing[q['question_id']] = row
            destination.flush()
            print(f'Preserved {len(existing)} source pairs.', flush=True)
            time.sleep(0.4)
    if args.training:
        def question_text(row):
            return ' '.join(html.unescape(re.sub('<[^>]+>', ' ', row['question_html'])).split())
        seen_questions = {question_text(row) for line in pilot_sources.read_text().splitlines() if (row := json.loads(line))['question_id'] in exclusions}
        unique = {}
        for question_id, row in existing.items():
            text = question_text(row)
            if question_id in exclusions or text in seen_questions or not row.get('question_license') or not row.get('answer_license'):
                continue
            seen_questions.add(text)
            unique[question_id] = row
        existing = unique
        source_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in existing.values()), encoding='utf-8')
    for domain in ('indoor', 'garden'):
        rows = [row for row in existing.values() if row['candidate_domain'] == domain]
        (OUTPUT / (domain + '_candidates.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        print(f'{domain}: {len(rows)} candidates', flush=True)


if __name__ == '__main__':
    main()
