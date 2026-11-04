"""Evaluate Phase 2 under strict fixed candidate budgets.

Transport failures retry without consuming an AI proposal slot, but count toward
wall time/deployment reliability. Invalid model output consumes a proposal slot.
Every candidate is independently graded for state or full operation according to
its stated task; semantic feedback goes only to the full-feedback condition.
"""
import time
from .core import evaluate, model_feedback
from .local_agent import OllamaHTTPFailure

STRATEGIES = ('plain', 'runtime', 'semantic', 'full')


def run_case(case, agent, *, strategy='full', attempts=3,
             transport_retries=1, backend='pennylane'):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown strategy')
    if not 1 <= attempts <= 5 or not 0 <= transport_retries <= 3:
        raise ValueError('Invalid attempt budgets')
    initial = evaluate(case, case['faulty'], backend=backend)
    if initial['passed']:
        raise ValueError('Invalid benchmark: faulty circuit already satisfies goal')
    started = time.perf_counter()
    log=[]
    feedback=''
    requests=0
    server_failures=0
    valid_proposals=0
    for number in range(1, attempts+1):
        candidate=None
        usage={}
        request_errors=[]
        proposal_error=None
        attempted_since = time.perf_counter()
        incoming_feedback = feedback
        feedback_applied = bool(incoming_feedback) and strategy in ('semantic', 'full')
        for _ in range(transport_retries+1):
            requests+=1
            try:
                candidate, usage = agent.propose(case, feedback=incoming_feedback, attempt=number)
                break
            except OllamaHTTPFailure as exc:
                server_failures+=1
                request_errors.append(str(exc))
            except (ValueError, TypeError) as exc:
                # A response arrived, but the LLM did not return valid structured JSON.
                # This uses one proposal slot, unlike a transport failure.
                proposal_error=f'{type(exc).__name__}: {str(exc)[:200]}'
                break
        else:
            log.append({'proposal_index': number, 'candidate': None,
                        'request_errors': request_errors,
                        'feedback_applied': feedback_applied,
                        'feedback_sent': incoming_feedback if feedback_applied else '',
                        'evaluation': None, 'kind': 'transport_exhausted',
                        'usage': {}, 'seconds': round(time.perf_counter()-attempted_since,3)})
            # No response: cannot evaluate LLM skill, stop this case.
            break
        if proposal_error:
            observation={'passed':False,'valid':False,'task_type':case['task_type'],
                         'target_fidelity':None,'operation_fidelity':None,
                         'target_pass':False,'regression_pass':False if case['task_type']=='operation' else None,
                         'gate_count':None,'two_qubit_gates':None,
                         'error':'ModelOutputError: '+proposal_error}
        else:
            observation=evaluate(case, candidate, backend=backend)
        if observation['valid']:
            valid_proposals+=1
        feedback=model_feedback(observation, strategy)
        log.append({'proposal_index': number,'candidate': candidate,
                    'request_errors': request_errors, 'evaluation': observation,
                    'feedback_applied': feedback_applied, 'feedback_sent': incoming_feedback if feedback_applied else '',
                    'kind': 'verified_success' if observation['passed'] else ('invalid_proposal' if not observation['valid'] else 'incorrect_proposal'),
                    'usage': usage, 'seconds': round(time.perf_counter()-attempted_since,3)})
        if observation['passed']:
            break
    success=any(x['kind']=='verified_success' for x in log)
    return {'case_id':case['id'], 'task_id':case['task_id'], 'category':case['category'],
            'fault_count':case['fault_count'], 'task_type':case['task_type'],
            'strategy':strategy, 'model':agent.model, 'seed':agent.seed,
            'initial_target_fidelity':initial['target_fidelity'],
            'initial_operation_fidelity':initial['operation_fidelity'],
            'successful':success, 'requests':requests,
            'server_failures':server_failures, 'proposals':sum(x['evaluation'] is not None for x in log),
            'valid_proposals':valid_proposals,
            'first_proposal_success':bool(log and log[0]['kind']=='verified_success'),
            'semantic_feedback_used':any(x.get('feedback_applied') and x['evaluation'] is not None for x in log),
            'elapsed_seconds':round(time.perf_counter()-started,3),'attempts':log}


def summarize(records):
    n=len(records)
    return {'cases':n, 'repaired':sum(r['successful'] for r in records),
            'verified_repair_rate':round(sum(r['successful'] for r in records)/n,4) if n else None,
            'first_generated_proposal_successes':sum(r['first_proposal_success'] for r in records),
            'total_generated_proposals':sum(r['proposals'] for r in records),
            'transport_failures':sum(r['server_failures'] for r in records),
            'incorrect_valid_proposals':sum(sum(a['kind']=='incorrect_proposal' for a in r['attempts']) for r in records),
            'invalid_proposals':sum(sum(a['kind']=='invalid_proposal' for a in r['attempts']) for r in records),
            'operation_regression_rejections':sum(sum((a['evaluation'] is not None and a['evaluation'].get('target_pass') and a['evaluation'].get('regression_pass') is False) for a in r['attempts']) for r in records),
            'total_prompt_tokens':sum(sum(a.get('usage',{}).get('prompt_tokens') or 0 for a in r['attempts']) for r in records),
            'total_generated_tokens':sum(sum(a.get('usage',{}).get('generated_tokens') or 0 for a in r['attempts']) for r in records),
            'feedback_needed_cases':sum(sum(a['kind']=='incorrect_proposal' for a in r['attempts'])>0 for r in records),
            'successful_after_feedback':sum(r['successful'] and r['semantic_feedback_used'] for r in records),
            'elapsed_seconds':round(sum(r['elapsed_seconds'] for r in records),3)}
