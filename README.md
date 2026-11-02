# Self-Healing Quantum Circuits with GPT-OSS 20B and PennyLane

**Begin with [START_HERE.md](START_HERE.md).** Research design: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

A **research MVP (v0.1)** for a Windows PC with 32 GB RAM and NVIDIA GTX 1080 8 GB. All AI inference is local through Ollama. The model weights are NOT fine-tuned. **No API key, subscription, or paid API is required.**

## What is implemented

- `benchmarks/pilot_cases.json`: 12 manually defined faulty quantum operations (6 single fault, 6 compound fault), each with a trusted reference.
- `selfheal/schema.py`: gate allowlist and strict parameter validation; *model-generated Python is never executed*.
- `selfheal/quantum.py`: **PennyLane `default.qubit`** simulation (default) and an independent NumPy oracle for cross-checks. Supports 2–4 qubits.
- `selfheal/verifier.py`: state fidelity on |0...0> and a stricter whole-operation fidelity test (relative to the reference operation).
- `selfheal/agent.py`: GPT-OSS via the **local** Ollama `/api/generate` JSON interface; no token or cloud access.
- `selfheal/runner.py`: iterative repair, structured feedback, and per-attempt metrics.
- `selfheal/cli.py`: doctor / verify / repair / benchmark commands.
- `draw_circuit.py`: circuit text rendering and optional PNG screenshot.
- `app.py`: optional Streamlit interface.
- `tests`: unit checks, a stubbed repair loop, and PennyLane integration tests where installed.

The **independent reference circuit is never given to GPT-OSS**. Only the natural language specification, faulty JSON, and permitted feedback are visible to the model.

## 1. VS Code / Windows installation

1. Install **Python 3.12 x64** and [Ollama](https://ollama.com/download).
2. Extract this ZIP. In VS Code use **File → Open Folder** to open `GPT_OSS_PennyLane_SelfHealing`.
3. Open **Terminal → New Terminal**, using **PowerShell**.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, use the virtual environment's Python directly, e.g. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, or choose the interpreter with VS Code **Python: Select Interpreter**.

## 2. Ollama setup

```powershell
ollama list
ollama pull gpt-oss:20b   # only if not already downloaded
```

Check `ollama run gpt-oss:20b` first, then type `/bye`. The Ollama Windows app normally starts its local service automatically. You can check `ollama ps` while a model is loaded. GPT-OSS 20B may require CPU/GPU offloading with an 8 GB GTX 1080, making responses slow. To test a smaller model, use `ollama pull qwen3:4b` and pass `--model qwen3:4b` to repair/benchmark commands. GPU driver compatibility should be checked with `nvidia-smi`.

## 3. Verify the benchmark WITHOUT calling the LLM

```powershell
python -m selfheal.cli doctor
python -m unittest discover -s tests -v
python -m selfheal.cli verify --case bell_reverse_cnot --which faulty
python -m selfheal.cli verify --case bell_reverse_cnot --which reference
python draw_circuit.py --case bell_reverse_cnot --which faulty --png
```

The faulty circuit should report `passed: false`, and the trusted reference `passed: true`. `draw_circuit` saves `results/bell_reverse_cnot_faulty.png` (a real PennyLane circuit image).

**If PennyLane is missing:** the pure NumPy unit tests run but an integration test is skipped. On your PC with PennyLane installed, the test should run and check the agreement with the reference NumPy implementation.

## 4. Run your FIRST GPT-OSS self-healing experiment

Start with just one, easy two-qubit case:

```powershell
python -m selfheal.cli repair --case bell_reverse_cnot --model gpt-oss:20b --strategy full --attempts 3
```

What happens:
1. GPT-OSS sees the goal and faulty circuit in JSON, **not** the reference circuit.
2. It proposes corrected **JSON gate operations**, not executable Python.
3. PennyLane simulates the candidate and compares against the independent NumPy reference.
4. Failed attempts produce feedback, up to three total attempts.
5. Successful repairs pass both target-state and whole-operation fidelity thresholds.
6. JSON results are saved under `results/` for your research report.

If you get an Ollama timeout, try a smaller local model or increase `--timeout 600`. Model speed depends on how much is offloaded to system RAM.

## 5. Run a small benchmark, then compare strategies

```powershell
python -m selfheal.cli benchmark --limit 3 --model gpt-oss:20b --strategy plain --attempts 3
python -m selfheal.cli benchmark --limit 3 --model gpt-oss:20b --strategy full --attempts 3
```

When these are working, set `--limit 12` to cover all pilot cases. Other strategies: `structured`, `runtime`, `semantic`. **Each strategy is always graded with the same full verifier**, even if its feedback is restricted. The three candidate attempts for `plain` are independent no-feedback attempts, so report both first-attempt success and success within the attempt budget.

**Research caution:** This set has only six reference operations with 12 injected faults. It is for prototype validation and is NOT a publishable benchmark on its own. Add unseen, independently checked tasks, random seeds, and controls before presenting general reliability results. Do not claim the method is better than alternatives without running comparative experiments.

## 6. Optional browser app

```powershell
python -m streamlit run app.py
```

Use one case at a time. Each action can take minutes on offloaded GPT-OSS. The application does **not** accept or execute arbitrary Python from GPT-OSS.

## 7. Save to GitHub

Create an empty GitHub repository at <https://github.com/new>, then in project folder:

```powershell
git init
git status
git add .gitignore .env.example README.md START_HERE.md EXPERIMENT_PROTOCOL.md requirements.txt app.py draw_circuit.py selfheal benchmarks tests results/.gitkeep
git diff --cached --stat
git commit -m "Initial self-healing PennyLane MVP with GPT-OSS repair and verification"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Never commit `.venv`, crawled text, cached embeddings, model weights, private documents, credentials, or unreviewed experiment outputs.

## Architecture and study scope

`task spec + faulty circuit → GPT-OSS JSON proposal → strict gate validation → PennyLane simulation → fidelity and operation checks → feedback/retry → JSON results`

The 12 pilot specifications explicitly demand the **whole reference operation**, not only the output from input |00...0>. The extra operation-fidelity check ensures behavior is preserved across all basis inputs and relative phases. These tests use ideal noiseless simulation and do not establish physical-device correctness.

`nomic-embed-text` and Common Crawl are **not required for the first circuit-repair MVP**. Web text by itself is not a reliable quantum debugging benchmark. If you later add documentation RAG, embed licensed, curated PennyLane documentation with Nomic and retrieve it for GPT-OSS without revealing trusted test answers.

## Baselines / research metrics

- Primary: verified repair success (`successful`), first-attempt repair rate, repair after up to 3 attempts.
- Safety: regression rejections, invalid JSON and gate-operation errors.
- Physics: target state fidelity, whole-operation fidelity (global-phase insensitive).
- Resources: wall-clock seconds, number of attempts, Ollama prompt and generated tokens.
- Recommended study: fixed model versions, comparable attempt and token budgets, untouched reference tests, different random seeds, holdout tasks, confidence intervals and error-category analysis.

**Limitations:** No physical quantum device or noise models; minimal gate set and 2–4 qubits; independent NumPy reference has not been externally audited; model may hallucinate or fail; no GPT-OSS fine-tuning; structured gate plans only; no untrusted Python execution. Prototype code should be reviewed before expanding to complex quantum algorithms.

## Papers motivating the specific experiments

- *QBugLM* (2026): further investigation of multi-fault debugging, regression behavior, and framework generalization.
- *QuanBench* (2025): semantic correctness and algorithmic bugs in LLM-generated quantum programs.
- *QASM-Eval* (2026): limits of synthetic tasks and need for broader reasoning/repository-level evaluation.
- *QUASAR* (2025): simulator verification + learning already explored; this MVP does not claim to invent that concept.
- *A structured LLM prompt engineering framework for generating quality, performant simulation code* (2026): reproducible behavioral tests and structured requirements.
