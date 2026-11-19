import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.agents import DEFAULT_MODEL
from evaluation.metrics import compare_scores
from evaluation.pipeline import EvaluationRun, read_jsonl
from evaluation.runtime import atomic_json, status_snapshot


async def evaluate(args):
    run = EvaluationRun(args.run_dir, args.answers, args.benchmark, args.model, args.effort)
    try:
        return 0 if await run.full() else 1
    finally:
        await run.close()


def main():
    parser = argparse.ArgumentParser(description="Grade saved answers using frozen criteria and sources")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--model", choices=[DEFAULT_MODEL], default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status_snapshot(args.run_dir), ensure_ascii=False, indent=2))
        return 0
    if args.compare:
        result = compare_scores(*(read_jsonl(path) for path in args.compare))
        args.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.run_dir / "comparison.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not args.answers or not args.benchmark:
        parser.error("--answers and --benchmark are required for grading")
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
