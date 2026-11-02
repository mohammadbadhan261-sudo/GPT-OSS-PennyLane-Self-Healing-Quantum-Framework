# Experimental protocol: planned (not yet executed)

## Key research hypothesis

H1: For comparable local inference budgets and identical independent reference
checks, physics-based simulator feedback increases the proportion of quantum
circuits repaired correctly compared with no-feedback and syntax-only baselines.

H2: Adding full-operation equivalence to target-state tests identifies false
repairs that succeed on |0...0> but alter behavior on other inputs.

H3: Compound mutations show lower verified repair success than single faults.

## Current benchmark and its scope

- Six hand-constructed ideal unitary tasks, each with single- and double-fault mutants.
- Twelve total faulty examples. All independently verified wrong on the reference
  initial state by a local NumPy simulator.
- All tasks are small (2-4 qubits) and may be trivial if the natural-language
  requirement explicitly describes the reference gates. This is a FIRST
  implementation test only, not a generalization evaluation.
- Whole-operation equivalence is checked because each task specification asks
  the candidate to reproduce a specific full quantum operation.

## Baseline strategies

- `plain`: basic prompt, no feedback across allowed attempts.
- `structured`: more explicit gate and correctness requirements, no feedback.
- `runtime`: only invalid-schema/runtime feedback.
- `semantic`: state fidelity feedback for initial zero-state input.
- `full`: state feedback plus complete unitary overlap (global-phase insensitive).

All are scored independently with the same full correctness test. One attempt
corresponds to one proposed candidate. Use three attempts for each strategy,
and report first-attempt success separately. Record the model tag, Ollama
version, GPU offload fraction, Python/PennyLane versions, prompt template,
seed, elapsed time, token counts, and any timeouts.

## Quantitative metrics

- Verified repair success: cases passing state and operation fidelity.
- Pass@1: cases passing the first attempt / cases tested.
- Success within 3 proposals (empirical pass-within-budget, not the unbiased
  Pass@k estimator used with independent random draws).
- Regression rejection: target state correct but whole-operation wrong.
- Invalid-output rate: malformed schema, out-of-range gates/parameters, etc.
- Mean/median end-to-end wall-clock time with variability across runs.
- Target and whole-operation fidelities, gate counts, two-qubit gate counts.

## Requirements for a defensible study

1. Create an additional independently designed HOLDOUT task set, without direct
   gate-by-gate answers in prompts. Include task-level specifications and
   property-based tests. Do not construct it by copying the demo goals.
2. Add several independent mutations per reference and randomize fault
   location and type, but inspect reference truths and verify mutants are faulty.
3. Run at least three different fixed random seeds and repeat experiments;
   calculate paired bootstrap confidence intervals or McNemar's test for
   matched binary outcomes.
4. Budget equal numbers of model calls, output token caps and context length.
5. Record one GPU/RAM status per experiment and note thermal offloading.
6. Avoid using a single reference state when tasks require full-unitary
   equivalence or have relative phase requirements.
7. Do not interpret ideal simulator results as hardware reliability or quantum
   advantage. Add noise/shots only as a separate labeled experimental extension.
8. Protect model and trusted evaluations: do not include the reference circuit,
   hidden tests, or reference operator in any prompt or retrieved materials.

## Future extensions

- Documentation retrieval with Nomic embeddings and curated, suitably licensed
  PennyLane documentation; NOT arbitrary Common Crawl web text.
- Unseen PennyLane API tasks, compound faults, multiple quantum backends,
  noise models, optional pyproject CI on GitHub Actions.
- Independent judge of correctness is simulator + trusted reference tests,
  never an LLM grading its own output.
