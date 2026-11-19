import asyncio
import json
from pathlib import Path

from .graded import Card, Grade, Grader, grade_payload, score, summary, validate_grade, validate_refs
from .runtime import Journal, atomic_json, transient_failure
from .storage import digest


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


class EvaluationRun:
    def __init__(self, directory, answers, benchmark, model="gpt-6.1-sol", effort="high", backend=None):
        self.directory = Path(directory)
        self.benchmark = Path(benchmark)
        if self.directory.resolve() == self.benchmark.resolve():
            raise ValueError("Use a separate output directory")
        self.cards = read_jsonl(self.benchmark / "cards.jsonl")
        answer_rows = read_jsonl(answers)
        self.answers = {row["id"]: row for row in answer_rows}
        ids = [card["id"] for card in self.cards]
        if not ids or len(set(ids)) != len(ids) or len(self.answers) != len(answer_rows) or set(ids) != self.answers.keys():
            raise ValueError("Answers must cover the frozen questions exactly once")
        for bundle in self.cards:
            if digest({key: value for key, value in bundle.items() if key != "version"}) != bundle["version"]:
                raise ValueError(f"Frozen criteria or sources changed: {bundle['id']}")
            if not bundle["approved"] or bundle["question"] != self.answers[bundle["id"]]["question"]:
                raise ValueError(f"Question or criteria mismatch: {bundle['id']}")
            card = Card.model_validate(bundle["prepared"]["card"])
            passages = bundle["passages"]
            if not card.ready or len({passage["id"] for passage in passages}) != len(passages):
                raise ValueError(f"Invalid criteria or passages: {bundle['id']}")
            known = {passage["id"] for passage in passages}
            for requirement in card.requirements:
                validate_refs(requirement.evidence_ids, known, "evidence IDs")
        self.results = {}
        self.errors = []
        for path in [self.benchmark / "results.jsonl", self.directory / "results.jsonl"]:
            if not path.exists():
                continue
            rows = read_jsonl(path)
            if len({row["id"] for row in rows}) != len(rows):
                raise ValueError(f"Duplicate results: {path}")
            for row in rows:
                if row["id"] not in self.answers:
                    raise ValueError(f"Unknown result ID: {row['id']}")
                self.results[row["id"]] = row
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.directory / "state.sqlite")
        try:
            inputs = digest({"cards": self.cards, "answers": answer_rows, "model": model, "effort": effort})
            previous = self.journal.db.execute("SELECT value FROM metadata WHERE key='inputs'").fetchone()
            if previous and previous[0] != inputs:
                raise ValueError("Inputs changed; use a new output directory")
            with self.journal.db:
                self.journal.db.execute("INSERT OR IGNORE INTO metadata VALUES ('inputs', ?)", (inputs,))
            self.journal.set_budget("evaluation", 360, None)
            self.grader = Grader(self.journal, model, effort, backend=backend)
            cache = self.benchmark / "state.sqlite"
            if cache.exists():
                self.journal.import_results(cache, kind="judge")
        except BaseException:
            self.journal.close()
            raise

    async def close(self):
        await self.grader.close()
        self.journal.close()

    def status(self, stage):
        atomic_json(self.directory / "status.json", {"stage": stage, "completed": len(self.completed),
                    "expected": len(self.cards), "errors": self.errors,
                    "usage": self.journal.usage("evaluation")})

    async def parallel(self, ids, function, stage):
        semaphore = asyncio.Semaphore(3)

        async def one(case_id):
            async with semaphore:
                while True:
                    try:
                        return await function(case_id)
                    except Exception as error:
                        key = f"task-recovery:{stage}:{case_id}"
                        row = self.journal.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                        attempts = json.loads(row[0])["attempts"] if row else 0
                        if transient_failure(error) and attempts < 2:
                            with self.journal.db:
                                self.journal.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                                                        (key, json.dumps({"attempts": attempts + 1})))
                            await asyncio.sleep(getattr(self, "task_retry_delay", 30) * 2**attempts)
                            continue
                        self.errors.append({"id": case_id, "error": f"{type(error).__name__}: {error}"})
                        return None
                    finally:
                        self.status(stage)

        return await asyncio.gather(*(one(case_id) for case_id in ids))

    async def evaluate(self, case_id):
        bundle = self.by_id[case_id]
        answer = self.answers[case_id]["answer"]
        payload = grade_payload(bundle["question"], answer, bundle["prepared"]["card"], bundle["passages"])
        saved = self.results.get(case_id)
        reuse = saved and (saved["question"], saved["answer"], saved["card_version"], saved["judge_signature"]) == (
            bundle["question"], answer, bundle["version"], self.grader.signature)
        if reuse:
            grade = Grade.model_validate(saved["grade"])
            validate_grade(grade, payload)
            if score(grade) != saved["score"]:
                raise ValueError(f"Saved score does not match its grade: {case_id}")
            cached = True
        else:
            grade, cached = await self.grader.call("evaluation", "judge", payload)
        result = {**(saved if reuse else {}), "id": case_id, "variant": "model", "question": bundle["question"], "answer": answer,
                  "card_version": bundle["version"], "judge_signature": self.grader.signature,
                  "topic": bundle["topic"], "source_group_id": bundle["source_group_id"],
                  "information_sufficiency": bundle["information_sufficiency"],
                  "score": score(grade), "grade": grade.model_dump(), "cache_hit": cached}
        self.completed[case_id] = result
        self.save()
        print(f"{len(self.completed)}/{len(self.cards)}", flush=True)
        return result

    def save(self):
        rows = [self.completed[bundle["id"]] for bundle in self.cards if bundle["id"] in self.completed]
        write_jsonl(self.directory / "results.jsonl", rows)
        report = summary(rows, len(self.cards))
        report["usage"] = self.journal.usage("evaluation")
        atomic_json(self.directory / "summary.json", report)

    async def full(self):
        self.by_id = {bundle["id"]: bundle for bundle in self.cards}
        self.completed = {}
        self.status("started")
        await self.parallel(list(self.by_id), self.evaluate, "evaluation")
        self.save()
        complete = len(self.completed) == len(self.cards) and not self.errors
        self.status("complete" if complete else "incomplete")
        return complete
