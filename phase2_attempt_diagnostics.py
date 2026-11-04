"""Inspect GPT-OSS + PennyLane Phase 2 repairs without changing research data.

Usage on Windows:
  python phase2_attempt_diagnostics.py results/phase2_compare_14.json

Creates an attempt-level CSV alongside the input JSON.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def kind_of(attempt):
    kind = attempt.get('kind')
    if kind:
        return kind
    ev = attempt.get('evaluation') or {}
    if ev.get('passed'):
        return 'verified_success'
    if not ev:
        return 'transport_exhausted'
    return 'incorrect_proposal' if ev.get('valid') else 'invalid_proposal'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--csv', type=Path)
    args = p.parse_args()
    data = json.loads(args.report.read_text(encoding='utf-8'))
    if data.get('study') != 'phase2_behavior_spec_v1':
        raise SystemExit('Input must be a Phase 2 JSON report from phase2.cli')

    pairs = defaultdict(dict)
    rows = []
    recovery = Counter()
    for case in data.get('cases', []):
        trial = case.get('trial', 1)
        case_id = case.get('case_id', '?')
        strategy = case.get('strategy', '?')
        pairs[(trial, case_id)][strategy] = case
        attempts = case.get('attempts', [])
        previous = []
        for a in attempts:
            k = kind_of(a)
            ev = a.get('evaluation') or {}
            if k == 'verified_success' and a.get('feedback_applied'):
                if 'incorrect_proposal' in previous:
                    recovery[(strategy, 'semantic_error_seen_before_success')] += 1
                elif 'invalid_proposal' in previous:
                    recovery[(strategy, 'structural_error_seen_before_success')] += 1
                else:
                    recovery[(strategy, 'other_feedback_before_success')] += 1
            rows.append({
                'trial': trial,
                'case_id': case_id,
                'strategy': strategy,
                'proposal_index': a.get('proposal_index'),
                'kind': k,
                'feedback_applied': bool(a.get('feedback_applied')),
                'valid': ev.get('valid'),
                'passed': ev.get('passed'),
                'target_fidelity': ev.get('target_fidelity'),
                'operation_fidelity': ev.get('operation_fidelity'),
                'error': ev.get('error', ''),
                'seconds': a.get('seconds'),
                'generated_tokens': (a.get('usage') or {}).get('generated_tokens'),
            })
            previous.append(k)

    print(f"Report: {args.report}")
    print(f"Benchmark: {data.get('study')}")
    print(f"Records: {len(data.get('cases', []))} | Distinct trial/case pairs: {len(pairs)}")
    print('\nPAIRED CASE OUTCOMES')
    counts = Counter()
    for (trial, case_id), by_strategy in sorted(pairs.items()):
        if 'plain' not in by_strategy or 'full' not in by_strategy:
            print(f'Trial {trial} {case_id}: incomplete pair')
            continue
        plain, full = by_strategy['plain'], by_strategy['full']
        a, b = bool(plain.get('successful')), bool(full.get('successful'))
        outcome = ('both_pass' if a and b else 'full_only' if b else
                   'plain_only' if a else 'both_fail')
        counts[outcome] += 1
        print(f'{case_id:39s}  plain={str(a):5s}  full={str(b):5s}  {outcome}')
    print('Paired totals:', dict(counts))

    print('\nPROPOSAL OUTCOMES')
    by_strategy = defaultdict(Counter)
    for row in rows:
        by_strategy[row['strategy']][row['kind']] += 1
    for strategy in sorted(by_strategy):
        print(strategy, dict(by_strategy[strategy]))
    print('\nSUCCESS AFTER FEEDBACK, CLASSIFIED BY PREVIOUS FAILURE')
    print(dict(recovery) or 'None observed')

    print('\nFAILED OR MULTI-PROPOSAL CASE DETAILS')
    for case in data.get('cases', []):
        attempts = case.get('attempts', [])
        if case.get('successful') and len(attempts) == 1:
            continue
        print(f"  {case.get('case_id')} [{case.get('strategy')}] "
              f"success={case.get('successful')} attempts={len(attempts)}")
        for a in attempts:
            ev = a.get('evaluation') or {}
            err = (ev.get('error') or '').replace('\n', ' ')
            print(f"    #{a.get('proposal_index')}: {kind_of(a)}; "
                  f"feedback={bool(a.get('feedback_applied'))}; "
                  f"fidelity={ev.get('target_fidelity')}; "
                  f"error={err[:180]}")

    dest = args.csv or args.report.with_suffix('.attempts.csv')
    if rows:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved attempt-level CSV: {dest}')
    print('\nNOTE: The Phase 2 field feedback_needed_cases counts cases with valid-but-incorrect '
          'proposals. successful_after_feedback also includes recoveries after invalid proposals. '
          'Neither proves improved semantic repair by itself.')


if __name__ == '__main__':
    main()
