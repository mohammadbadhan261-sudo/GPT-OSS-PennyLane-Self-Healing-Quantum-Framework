"""Summarize paired Phase 2 results from the same compare JSON. No pandas required."""
import argparse, json, csv
from pathlib import Path
from collections import defaultdict
from .study import summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('report',type=Path)
    p.add_argument('--csv',type=Path)
    args=p.parse_args()
    report=json.loads(args.report.read_text(encoding='utf-8'))
    if report.get('study')!='phase2_behavior_spec_v1':
        raise ValueError('Expected Phase 2 research report')
    cases=report['cases']
    for strategy in sorted({c['strategy'] for c in cases}):
        print('\n',strategy.upper(),json.dumps(summarize([x for x in cases if x['strategy']==strategy]),indent=2))
    paired=defaultdict(dict)
    for case in cases:
        k=(case['trial'],case['case_id'])
        paired[k][case['strategy']]=case
    rows=[]
    for (trial,case_id),v in sorted(paired.items()):
        if not ('plain' in v and 'full' in v):continue
        a,b=v['plain'],v['full']
        rows.append({'trial':trial,'case_id':case_id,'task_type':a['task_type'],
                     'plain_success':int(a['successful']), 'full_success':int(b['successful']),
                     'plain_proposals':a['proposals'], 'full_proposals':b['proposals'],
                     'plain_request_errors':a['server_failures'], 'full_request_errors':b['server_failures'],
                     'plain_seconds':a['elapsed_seconds'], 'full_seconds':b['elapsed_seconds'],
                     'delta_full_minus_plain_seconds':round(b['elapsed_seconds']-a['elapsed_seconds'],3),
                     'semantic_feedback_used_full':int(b['semantic_feedback_used'])})
    print('\nPaired comparisons:',len(rows),'of',len(paired),'trial/case combinations')
    if rows:
        better=sum(x['full_success']>x['plain_success'] for x in rows)
        worse=sum(x['full_success']<x['plain_success'] for x in rows)
        ties=len(rows)-better-worse
        print('Full-only success:',better,'Plain-only success:',worse,'Tied:',ties)
        print('Full strategy used semantic feedback:',sum(x['semantic_feedback_used_full'] for x in rows),'pairs')
        dest=args.csv or args.report.with_suffix('.comparison.csv')
        with dest.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        print('Saved:',dest)
    print('\nCautions: paired exploratory study, not an independent holdout; '
          'two faults sharing a task specification are correlated. '
          'Exclude transport-exhausted trials from model-skill rate estimates while reporting deployment failures separately.')

if __name__=='__main__':main()
