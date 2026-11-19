from collections import Counter
import json
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/prepared/plant_care_train_1000"
SEED = 3407
SYSTEM_PROMPT = (
    "Ты помощник по уходу за комнатными растениями, садом и огородом. "
    "Отвечай новичку на русском языке кратко и понятно, объясняй причины советов. "
    "Если совет зависит от недостающих условий, задавай уточняющие вопросы. "
    "Обозначай неопределённость и не выдумывай сведения."
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalized(text):
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use .venv/bin/python.")
    source = DATA / "cases.jsonl"
    cases = read_jsonl(source)
    if len(cases) != 1000 or len({row["id"] for row in cases}) != 1000:
        raise ValueError("Expected 1000 unique examples.")
    if len({row["source_group_id"] for row in cases}) != 1000:
        raise ValueError("Expected one example per source group.")
    if Counter(row["domain"] for row in cases) != {"indoor": 400, "garden": 600}:
        raise ValueError("Expected 400 indoor and 600 garden examples.")
    pilot = read_jsonl(ROOT / "data/prepared/plant_care_pilot/cases.jsonl")
    initial = read_jsonl(ROOT / "data/evaluation/init_questions/questions.jsonl")
    excluded_questions = {normalized(row["question"]) for row in pilot + initial}
    excluded_groups = {row["source_group_id"] for row in pilot}
    for row in cases:
        if normalized(row["question"]) in excluded_questions or row["source_group_id"] in excluded_groups:
            raise ValueError(f"Overlap with evaluation data: {row['id']}")
        for field in ("question", "answer"):
            if not row[field].strip() or re.search(r"<\|.*?\|>|</?think>", row[field]):
                raise ValueError(f"Invalid field {field}: {row['id']}")
    rng = random.Random(SEED)
    splits = {"train": [], "validation": []}
    for domain, validation_count in (("indoor", 80), ("garden", 120)):
        rows = sorted((row for row in cases if row["domain"] == domain), key=lambda row: row["source_group_id"])
        rng.shuffle(rows)
        splits["validation"].extend(rows[:validation_count])
        splits["train"].extend(rows[validation_count:])
    for name, rows in splits.items():
        rng.shuffle(rows)
        records = [
            {
                "id": row["id"],
                "source_group_id": row["source_group_id"],
                "domain": row["domain"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
            }
            for row in rows
        ]
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
        (DATA / f"{name}.jsonl").write_bytes(content)
        print(f"{name}: {len(records)} examples")


if __name__ == "__main__":
    main()
