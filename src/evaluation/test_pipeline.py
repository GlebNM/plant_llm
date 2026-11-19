import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from .graded import Grader, summary
from .pipeline import EvaluationRun, read_jsonl, write_jsonl
from .runtime import Journal
from .storage import digest
from .test_resilience import good_grade, payload

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "data/runs/plant_care_pilot/20261102_evaluation_v11"
CANDIDATE = ROOT / "data/runs/plant_care_pilot/20261104_163730_lora_4b_evaluation_v11"


def backend():
    def respond(messages, info):
        raise AssertionError("A saved answer must not call the model")
    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


@pytest.mark.skipif(not (BASELINE / "results.jsonl").exists() or not (CANDIDATE / "results.jsonl").exists(),
                    reason="Saved evaluation files are required")
def test_saved_300_scores_and_schema_remain_compatible(tmp_path):
    async def scenario():
        cards = read_jsonl(BASELINE / "cards.jsonl")
        for source in (BASELINE, CANDIDATE):
            rows = read_jsonl(source / "results.jsonl")
            benchmark = tmp_path / source.name
            benchmark.mkdir()
            write_jsonl(benchmark / "cards.jsonl", cards)
            write_jsonl(benchmark / "results.jsonl", rows)
            answers = benchmark / "answers.jsonl"
            write_jsonl(answers, [{key: row[key] for key in ("id", "question", "answer")} for row in rows])
            run = EvaluationRun(tmp_path / (source.name + "_output"), answers, benchmark, backend=backend())
            try:
                assert run.grader.signature == rows[0]["judge_signature"]
                assert await run.full()
                actual = read_jsonl(run.directory / "results.jsonl")
                assert [{key: row[key] for key in row if key != "cache_hit"} for row in actual] == [
                    {key: row[key] for key in row if key != "cache_hit"} for row in rows]
                assert run.journal.usage("evaluation")["requests"] == 0
                assert json.loads((run.directory / "summary.json").read_text())["score_bounds_100"] == summary(rows, 300)["score_bounds_100"]
            finally:
                await run.close()
            resumed = EvaluationRun(run.directory, answers, benchmark, backend=backend())
            try:
                assert await resumed.full()
                assert resumed.journal.usage("evaluation")["requests"] == 0
            finally:
                await resumed.close()
    asyncio.run(scenario())


def test_frozen_question_mismatch_stops_before_model_creation(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    bundle = {"id": "a", "question": "original", "approved": True}
    bundle["version"] = digest(bundle)
    write_jsonl(benchmark / "cards.jsonl", [bundle])
    answers = tmp_path / "answers.jsonl"
    write_jsonl(answers, [{"id": "a", "question": "changed", "answer": "text"}])
    with pytest.raises(ValueError, match="mismatch"):
        EvaluationRun(tmp_path / "output", answers, benchmark, backend=backend())


def test_only_changed_answer_calls_model_and_resume_uses_cache(tmp_path):
    async def scenario():
        calls = []
        def respond(messages, info):
            calls.append(messages)
            return ModelResponse(parts=[TextPart(good_grade().model_dump_json())])
        model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))
        benchmark = tmp_path / "benchmark"
        benchmark.mkdir()
        data = payload()
        bundle = {"id": "a", "question": data["question"], "prepared": {"card": data["card"]},
                  "passages": data["passages"], "approved": True,
                  "topic": "watering", "source_group_id": "source", "information_sufficiency": "sufficient"}
        bundle["version"] = digest(bundle)
        write_jsonl(benchmark / "cards.jsonl", [bundle])
        probe = Grader(Journal(tmp_path / "probe.sqlite"), effort="high", backend=model)
        signature = probe.signature
        await probe.close()
        probe.journal.close()
        saved = {key: bundle[key] for key in ("id", "question", "topic", "source_group_id", "information_sufficiency")}
        saved.update(variant="model", answer="old", card_version=bundle["version"], judge_signature=signature,
                     score=2, grade=good_grade().model_dump(), cache_hit=False)
        write_jsonl(benchmark / "results.jsonl", [saved])
        answers = tmp_path / "answers.jsonl"
        write_jsonl(answers, [{"id": "a", "question": data["question"], "answer": "new answer"}])
        for _ in range(2):
            run = EvaluationRun(tmp_path / "output", answers, benchmark, backend=model)
            try:
                assert await run.full()
                assert len(calls) == 1
            finally:
                await run.close()
        write_jsonl(answers, [{"id": "a", "question": data["question"], "answer": "another answer"}])
        with pytest.raises(ValueError, match="Inputs changed"):
            EvaluationRun(tmp_path / "output", answers, benchmark, backend=model)
    asyncio.run(scenario())
