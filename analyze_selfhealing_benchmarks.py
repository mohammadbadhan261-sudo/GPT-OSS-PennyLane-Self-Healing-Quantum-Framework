"""Compare two Self-Healing Quantum Circuits benchmark JSON files.

Run from the project root:
 python analyze_selfhealing_benchmarks.py --full results/pilot_full.json --plain results/pilot_plain.json

Uses standard-library Python only; never modifies original input JSON.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def load(path):
    with path.open(encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get('cases'), list):
        raise ValueError(f'{path}: expected a JSON report with a cases list')
    rows = data['cases']
    ids = [row['case_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate case IDs')
    return data


def count_metrics(report):
    cases = report['cases']
    attempts = [a for c in cases for a in c.get('attempts', [])]
    errors = [str(a.get('evaluation', {}).get('error') or '') for a in attempts]
    transport = sum('Ollama connection/model request failed' in err for err in errors)
    generated_proposals = [a for a in attempts if a.get('candidate') is not None]
    first_generated = []
    for c in cases:
        generated = next((a for a in c.get('attempts', []) if a.get('candidate') is not None), None)
        if generated is not None:
            first_generated.append(bool(generated.get('evaluation', {}).get('passed')))
    prompt = sum((a.get('usage') or {}).get('prompt_tokens') or 0 for a in attempts)
    generated = sum((a.get('usage') or {}).get('generated_tokens') or 0 for a in attempts)
    time_values = [float(c.get('elapsed_seconds') or 0) for c in cases]
    return {
        'cases': len(cases),
        'successes': sum(bool(c.get('successful')) for c in cases),
        'first_request_successes': sum(bool(c.get('attempts')) and bool(c['attempts'][0].get('evaluation', {}).get('passed')) for c in cases),
        'attempts': len(attempts),
        'generated_proposals': len(generated_proposals),
        'first_generated_proposal_successes': sum(first_generated),
        'first_generated_proposal_denominator': len(first_generated),
        'connection_failures': transport,
        'invalid_or_failed_proposals': sum(not a.get('evaluation', {}).get('passed') for a in generated_proposals),
        'regression_rejections': sum(int(c.get('regression_rejections') or 0) for c in cases),
        'total_seconds': sum(time_values),
        'mean_seconds': statistics.mean(time_values) if time_values else None,
        'median_seconds': statistics.median(time_values) if time_values else None,
        'prompt_tokens': prompt,
        'generated_tokens': generated,
    }


def fmt(s):
    return f'{s:.3f}' if isinstance(s, float) else str(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--plain', type=Path, required=True)
    parser.add_argument('--csv', type=Path, default=Path('results/pilot_comparison.csv'))
    args = parser.parse_args()
    reports = {'full': load(args.full), 'plain': load(args.plain)}
    for name, report in reports.items():
        m = count_metrics(report)
        print(f'\n{name.upper()}')
        for key, value in m.items():
            print(f'  {key}: {fmt(value)}')
    a, b = [{c['case_id']: c for c in reports[k]['cases']} for k in ('full','plain')]
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError('Reports have no matching case IDs')
    print(f'\nMatching case IDs: {len(common)}')
    print(f'Only in full: {sorted(set(a)-set(b))}')
    print(f'Only in plain: {sorted(set(b)-set(a))}')
    print('Comparison: positive time delta means full took longer.')
    table = []
    for key in common:
        full, plain = a[key], b[key]
        secs_full, secs_plain = float(full['elapsed_seconds']), float(plain['elapsed_seconds'])
        table.append({'case_id': key, 'category': full.get('category',''),
                      'fault_count': full.get('fault_count',''),
                      'full_success': bool(full['successful']),
                      'plain_success': bool(plain['successful']),
                      'full_attempts': len(full.get('attempts', [])),
                      'plain_attempts': len(plain.get('attempts', [])),
                      'full_seconds': round(secs_full, 3),
                      'plain_seconds': round(secs_plain, 3),
                      'full_minus_plain_seconds': round(secs_full - secs_plain, 3),
                      'full_server_errors': sum('Ollama connection/model request failed' in str(t.get('evaluation', {}).get('error') or '') for t in full.get('attempts', [])),
                      'plain_server_errors': sum('Ollama connection/model request failed' in str(t.get('evaluation', {}).get('error') or '') for t in plain.get('attempts', []))})
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    print(f'\nSaved {args.csv}')
    print('Caution: one trial per method is descriptive only, not a statistical demonstration.')
    print('Caution: if goals specify the exact gate sequence, results are not a blind repair benchmark.')


if __name__ == '__main__':
    main()
