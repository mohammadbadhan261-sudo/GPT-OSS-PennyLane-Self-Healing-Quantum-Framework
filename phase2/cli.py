"""Execute: python -m phase2.cli validate | run | compare"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
from .core import evaluate
from .study import run_case, summarize
from .local_agent import LocalAgent
from selfheal.benchmark import load_cases

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / 'benchmarks' / 'phase2_behavior_cases.json'


def check_suite(path, backend):
    suite = load_cases(path)
    for c in suite:
        if c['task_type'] not in ('state', 'operation'):
            raise ValueError('task_type must be state or operation')
        if len(c['goal']) > 700:
            raise ValueError('Goal exceeds prompt budget')
        correct = evaluate(c, c['reference'], backend=backend)
        mutant = evaluate(c, c['faulty'], backend=backend)
        if not correct['passed'] or mutant['passed']:
            raise ValueError(f'Broken case {c["id"]}: reference or mutant validation failed')
    return suite


def save_report(path, args, suite_path, records):
    # Prevent accidentally overwriting the user's previous phase-1 pilot output.
    data = {'study': 'phase2_behavior_spec_v1',
            'settings':{'model':args.model,'backend':args.backend,
                        'strategies': ['plain', 'full'] if args.action=='compare' else [args.strategy],
                        'max_proposals':args.attempts,
                        'max_transport_retries_per_proposal':args.transport_retries,
                        'base_seed':args.seed,'repeats':args.repeats,
                        'suite_file': str(suite_path),
                        'suite_sha256': hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                        'python_version':platform.python_version()},
            'summary': {s:summarize([x for x in records if x['strategy']==s]) for s in sorted({r['strategy'] for r in records})},
            'cases': records}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Behavior-spec quantum circuit repair study')
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('validate','run','compare'):
        p = sub.add_parser(name)
        p.add_argument('--suite',type=Path,default=SUITE)
        p.add_argument('--backend',choices=['numpy','pennylane'],default='pennylane')
        if name != 'validate':
            p.add_argument('--model',default='gpt-oss:20b')
            p.add_argument('--strategy',choices=['plain','runtime','semantic','full'],default='full')
            p.add_argument('--attempts',type=int,default=3)
            p.add_argument('--transport-retries',type=int,default=1)
            p.add_argument('--timeout',type=int,default=420)
            p.add_argument('--num-ctx',type=int,default=2048)
            p.add_argument('--seed',type=int,default=42)
            p.add_argument('--limit',type=int,default=4)
            p.add_argument('--repeats',type=int,default=1)
            p.add_argument('--save',type=Path)
    args=parser.parse_args()
    suite=check_suite(args.suite,args.backend)
    if args.action == 'validate':
        print(f'Validated {len(suite)} cases with backend={args.backend}; all references pass and all mutants fail.')
        print('State cases:',sum(x['task_type']=='state' for x in suite),
              '| Operation cases:',sum(x['task_type']=='operation' for x in suite))
        return
    if not 1 <= args.limit <= len(suite):
        parser.error(f'--limit must be from 1 to {len(suite)}')
    if not 1 <= args.repeats <= 20 or not 1 <= args.attempts <= 5 or not 0 <= args.transport_retries <= 3:
        parser.error('Invalid --repeats/--attempts/--transport-retries')
    suite=suite[:args.limit]
    out=args.save or (ROOT/'results'/f'phase2_{args.action}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json')
    if out.exists():
        parser.error('Output file exists. Use a new --save path to preserve all results.')
    records=[]
    for trial in range(args.repeats):
        # A paired trial uses the same seed for both strategies and alternates order.
        # This does not eliminate latency confounds but reduces systematic ordering.
        agent=LocalAgent(model=args.model,seed=args.seed+1000*trial,
                         num_ctx=args.num_ctx,timeout=args.timeout)
        order=list(range(len(suite)))
        random.Random(args.seed+trial).shuffle(order)
        for j in order:
            case=suite[j]
            if args.action=='compare':
                strategies=['plain','full'] if ((j+trial)%2==0) else ['full','plain']
            else:
                strategies=[args.strategy]
            for strategy in strategies:
                print(f'Trial {trial+1}/{args.repeats}: {case["id"]} / {strategy}',flush=True)
                rec=run_case(case,agent,strategy=strategy,attempts=args.attempts,
                             transport_retries=args.transport_retries,backend=args.backend)
                rec['trial']=trial+1
                records.append(rec)
                print(f'  repaired={rec["successful"]}, proposals={rec["proposals"]}, '
                      f'server_failures={rec["server_failures"]}, elapsed={rec["elapsed_seconds"]}s',flush=True)
                save_report(out,args,args.suite,records)
    print('Results:',out)
    print(json.dumps({k:summarize([x for x in records if x['strategy']==k])
                      for k in sorted({r['strategy'] for r in records})},indent=2))

if __name__=='__main__':
    main()
