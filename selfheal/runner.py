"""Repair attempts, fair evaluation, and append-friendly result records."""
import time
from .verifier import assess, feedback_for

STRATEGIES = ('plain', 'structured', 'runtime', 'semantic', 'full')


def repair_case(case, agent, *, strategy='full', attempts=3, backend='pennylane'):
    if strategy not in STRATEGIES:
        raise ValueError('Invalid strategy')
    if type(attempts) is not int or not 1 <= attempts <= 5:
        raise ValueError('attempts must be between 1 and 5')
    original = assess(case, case['faulty'], backend=backend, require_operation=True)
    if original['passed']:
        raise ValueError('Invalid benchmark: faulty circuit already passes all tests')
    results = []
    feedback = ''
    start = time.perf_counter()
    for attempt in range(1, attempts + 1):
        start_attempt = time.perf_counter()
        metadata = {}
        candidate = None
        try:
            candidate, metadata = agent.propose(case, strategy, feedback=feedback, attempt=attempt)
            diagnostic = assess(case, candidate, backend=backend, require_operation=(strategy == 'full'))
            # Always grade against exactly the same held-out, full verification,
            # regardless of what the LLM was shown.
            outcome = (diagnostic if strategy == 'full' else
                       assess(case, candidate, backend=backend, require_operation=True))
            feedback = feedback_for(diagnostic, strategy)
        except Exception as exc:
            outcome = {'passed': False, 'valid': False, 'target_fidelity': None,
                       'operation_fidelity': None, 'target_pass': False,
                       'regression_pass': False, 'error': f'{type(exc).__name__}: {str(exc)[:300]}',
                       'gate_count': None, 'two_qubit_gates': None}
            feedback = feedback_for(outcome, strategy)
        results.append({'attempt': attempt, 'candidate': candidate,
                        'evaluation': outcome, 'seconds': round(time.perf_counter() - start_attempt, 3),
                        'usage': metadata})
        if outcome['passed']:
            break
    last = results[-1]['evaluation']
    return {
        'case_id': case['id'], 'category': case['category'], 'fault_count': case['fault_count'],
        'strategy': strategy, 'model': agent.model, 'backend': backend,
        'initial_target_fidelity': original['target_fidelity'],
        'initial_operation_fidelity': original['operation_fidelity'],
        'successful': bool(last['passed']), 'attempts_used': len(results),
        'elapsed_seconds': round(time.perf_counter() - start, 3),
        'regression_rejections': sum(bool(a['evaluation']['target_pass']) and not bool(a['evaluation']['regression_pass']) for a in results),
        'attempts': results,
    }


def summarize(records):
    n = len(records)
    repaired = sum(x['successful'] for x in records)
    first = sum(x['successful'] and x['attempts_used'] == 1 for x in records)
    return {'cases': n, 'repaired': repaired,
            'verified_repair_rate': round(repaired / n, 4) if n else None,
            'first_attempt_rate': round(first / n, 4) if n else None,
            'total_seconds': round(sum(x['elapsed_seconds'] for x in records), 3),
            'regression_rejections': sum(x['regression_rejections'] for x in records)}
