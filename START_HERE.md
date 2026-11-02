# Start here: First successful repair on Windows

## What you need

- Windows 10/11, VS Code, Python 3.12, Ollama.
- 32 GB system RAM and GTX 1080 FE 8 GB: GPT-OSS 20B can be slow due to offloading.
- `ollama list` should include `gpt-oss:20b` (or `qwen3:4b` as a faster backup).

## Do this in VS Code PowerShell, from the extracted project folder

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m selfheal.cli doctor
python -m unittest discover -s tests -v
python -m selfheal.cli verify --case bell_reverse_cnot --which faulty
python -m selfheal.cli verify --case bell_reverse_cnot --which reference
```

Expected: faulty `passed: false`, reference `passed: true`.

**Your first AI experiment:**

```powershell
python -m selfheal.cli repair --case bell_reverse_cnot --model gpt-oss:20b --strategy full --attempts 3
```

Results go under `results/` (file name includes date/time).

Optional circuit picture:

```powershell
python draw_circuit.py --case bell_reverse_cnot --which faulty --png
```

Optional browser:

```powershell
python -m streamlit run app.py
```

If 20B is too slow, use the smaller local model:

```powershell
ollama pull qwen3:4b
python -m selfheal.cli repair --case bell_reverse_cnot --model qwen3:4b --strategy full --attempts 3
```

## What to send back for troubleshooting

Paste the output of `python -m selfheal.cli doctor` and the terminal output of the `repair` command. If the model succeeds, copy the saved JSON's `successful`, `attempts_used`, and `attempts` evaluation values. No passwords or other personal files are needed.
