"""Reference-based grading for two distinct task specifications.

STATE: only the output from |00...0> is specified by the prompt, so
full-unitary equivalence MUST NOT be required for passing.
OPERATION: all inputs are specified; require full-unitary equivalence.

Reference circuit stays in the trusted benchmark and never enters the LLM request.
"""
from selfheal.schema import validate, CircuitError, two_qubit_count
from selfheal.quantum import simulate_state, operation_fidelity, fidelity
import numpy as np

TOL = 1e-6


def evaluate(case, candidate, backend='pennylane'):
    task_type = case['task_type']
    if task_type not in ('state', 'operation'):
        raise ValueError('Unknown task type')
    try:
        spec = validate(candidate, expected_qubits=case['reference']['n_qubits'])
        truth = simulate_state(case['reference'], 'numpy')
        got = simulate_state(spec, backend)
        state_f = fidelity(got, truth)
        operation_f = (operation_fidelity(spec, case['reference'], backend)
                       if task_type == 'operation' else None)
        target_pass = state_f >= (1.0 - TOL)
        operation_pass = (operation_f >= 1.0 - TOL) if task_type == 'operation' else None
        passed = target_pass and (operation_pass if task_type == 'operation' else True)
        return {'passed': bool(passed), 'valid': True, 'task_type': task_type,
                'target_fidelity': round(state_f, 8),
                'operation_fidelity': round(operation_f, 8) if operation_f is not None else None,
                'target_pass': bool(target_pass),
                'regression_pass': bool(operation_pass) if operation_pass is not None else None,
                'probabilities': (np.abs(got)**2).round(6).tolist(),
                'expected_probabilities': (np.abs(truth)**2).round(6).tolist(),
                'gate_count': len(spec['gates']), 'two_qubit_gates': two_qubit_count(spec), 'error': ''}
    except (ValueError, CircuitError, RuntimeError, TypeError, KeyError, IndexError) as exc:
        return {'passed': False, 'valid': False, 'task_type': task_type,
                'target_fidelity': None, 'operation_fidelity': None,
                'target_pass': False, 'regression_pass': False if task_type == 'operation' else None,
                'gate_count': None, 'two_qubit_gates': None,
                'error': f'{type(exc).__name__}: {str(exc)[:300]}'}


def model_feedback(evaluation, strategy):
    """Only candidate observations are returned. No reference gate sequence."""
    if strategy == 'plain':
        return ''
    if not evaluation['valid']:
        return 'Candidate rejected due to structural error: ' + evaluation['error']
    if strategy == 'runtime':
        return 'The circuit executed without a structural or runtime exception.'
    parts = [f"Target-state fidelity: {evaluation['target_fidelity']}. Required: >= {1-TOL}.",
             f"Reference output probabilities from |0...0>: {evaluation['expected_probabilities']}.",
             f"Candidate output probabilities: {evaluation['probabilities']}."]
    # Full-op results only when needed by the actual OPERATION specification.
    if strategy == 'full' and evaluation['task_type'] == 'operation':
        parts.append(f"Full-operation fidelity: {evaluation['operation_fidelity']}. Required: >= {1-TOL}.")
        parts.append('Check relative phases and effects on nonzero input basis states.')
    if evaluation['task_type'] == 'state':
        parts.append('Only the specified initial-state output is required; alternative correct preparations are acceptable.')
    return ' '.join(parts)
