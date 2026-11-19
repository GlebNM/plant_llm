# Plant Care LoRA

## Data

Obtain the input files separately and upload them to Colab.

| Files | Colab location |
|---|---|
| `questions.jsonl` with 15 initial questions | `/content/data/evaluation/init_questions/questions.jsonl` |
| `questions.jsonl` with 300 evaluation questions | `/content/data/prepared/plant_care_pilot/questions.jsonl` |
| `train.jsonl` and `validation.jsonl` | `/content/data/prepared/plant_care_train_1000/` |

## Run

Use Google Colab with an NVIDIA T4. Upload the inputs to the paths above and run each notebook from top to bottom. Use a fresh runtime when switching between training and generation.

| Order | Notebook | Task |
|---|---|---|
| 1 | [01 — Baseline](notebooks/01_qwen35_baseline.ipynb) | Generate answers to the 15 initial questions. |
| 2 | [02 — Evaluation questions](notebooks/02_qwen35_plant_care_pilot.ipynb) | Generate baseline answers to the 300 evaluation questions. |
| 3 | [04 — Training](notebooks/04_qwen35_4b_plant_lora.ipynb) | Train LoRA on 800 examples, with 200 validation examples. |
| 4 | [05 — LoRA answers](notebooks/05_qwen35_4b_lora_pilot.ipynb) | Answer the same 300 questions with the adapter. |

Before running 05, copy `answers.jsonl` and `config.json` from 02 to `/content/data/runs/plant_care_pilot/qwen35_4b_nonthinking/`. Copy the `adapter/` folder from 04 to `/content/data/runs/plant_care_train_1000/qwen35_4b_lora/adapter/`, or use a separately supplied adapter at that path.

## LLM evaluation

Download `answers.jsonl` from 05. Run locally with the separately supplied `cards.jsonl` in the benchmark folder. Codex authentication is required.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-evaluation.txt
.venv/bin/python scripts/evaluate.py \
  --run-dir data/runs/plant_care_pilot/lora_evaluation \
  --benchmark path/to/benchmark \
  --answers path/to/answers.jsonl \
  --model gpt-6.1-sol --effort high
```
