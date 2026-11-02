"""CLI: python -m selfheal.cli doctor|verify|repair|benchmark"""
import argparse
import datetime
import json
from pathlib import Path
import sys
from .benchmark import get_case, load_cases
from .verifier import assess
from .agent import OllamaAgent
from .runner import STRATEGIES, repair_case, summarize

ROOT = Path(__file__).resolve().parent.parent


def pretty(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description='Local GPT-OSS + PennyLane self-healing quantum circuits')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor', help='Check installation and local Ollama model')
    v = sub.add_parser('verify', help='Check the faulty circuit or the reference without Ollama')
    v.add_argument('--case', default='bell_reverse_cnot')
    v.add_argument('--which', choices=['faulty', 'reference'], default='faulty')
    v.add_argument('--backend', choices=['pennylane', 'numpy'], default='pennylane')
    r = sub.add_parser('repair', help='Repair one case using the local LLM')
    b = sub.add_parser('benchmark', help='Run more than one case with a local LLM')
    for q in [r, b]:
        q.add_argument('--model', default='gpt-oss:20b')
        q.add_argument('--strategy', choices=STRATEGIES, default='full')
        q.add_argument('--attempts', type=int, default=3)
        q.add_argument('--backend', choices=['pennylane', 'numpy'], default='pennylane')
        q.add_argument('--timeout', type=int, default=420)
        q.add_argument('--num-ctx', type=int, default=2048)
        q.add_argument('--save', type=Path)
    r.add_argument('--case', default='bell_reverse_cnot')
    b.add_argument('--limit', type=int, default=3)
    args = p.parse_args(argv)
    if args.command == 'doctor':
        info = {'python': sys.version.split()[0], 'project': str(ROOT),
                'benchmark_cases': len(load_cases())}
        try:
            import pennylane as qml
            info['pennylane'] = qml.__version__
        except ImportError:
            info['pennylane'] = 'NOT INSTALLED (pip install -r requirements.txt)'
        try:
            info['ollama_models'] = OllamaAgent().models()
        except Exception as exc:
            info['ollama_models'] = f'Unable to contact local Ollama: {exc}'
        pretty(info)
        return 0
    if args.command == 'verify':
        case = get_case(args.case)
        info = assess(case, case[args.which], backend=args.backend, require_operation=True)
        pretty({'case_id': case['id'], 'which': args.which, 'backend': args.backend, **info})
        return 0 if (info['passed'] if args.which == 'reference' else not info['passed']) else 1
    if args.timeout < 15 or args.num_ctx < 512:
        p.error('--timeout must be >=15 and --num-ctx >=512')
    agent = OllamaAgent(model=args.model, timeout=args.timeout, num_ctx=args.num_ctx)
    if args.command == 'repair':
        cases = [get_case(args.case)]
    else:
        if not 1 <= args.limit <= 100:
            p.error('--limit must be from 1 to 100')
        cases = load_cases()[:args.limit]
    saved = []
    out = args.save or ROOT / 'results' / f'{args.command}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        print(f'Running {case["id"]} with {args.model} / {args.strategy}...', flush=True)
        record = repair_case(case, agent, strategy=args.strategy, attempts=args.attempts, backend=args.backend)
        saved.append(record)
        print(f'  verified_success={record["successful"]} attempts={record["attempts_used"]} '
              f'elapsed={record["elapsed_seconds"]}s', flush=True)
        # Save incrementally; a long local inference can be interrupted.
        out.write_text(json.dumps({'settings': {'model': args.model, 'strategy': args.strategy,
                                                 'max_attempts': args.attempts, 'backend': args.backend},
                                   'summary': summarize(saved), 'cases': saved}, indent=2), encoding='utf-8')
    print('Results:', out)
    pretty(summarize(saved))
    return 0


if __name__ == '__main__':
    sys.exit(main())
