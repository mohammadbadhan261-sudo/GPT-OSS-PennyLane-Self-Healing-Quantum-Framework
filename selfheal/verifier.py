"""Untrusted model output is accepted only after independent checks."""
import numpy as np
from .schema import CircuitError, validate, two_qubit_count
from .quantum import simulate_state, fidelity, operation_fidelity

TARGET_TOLERANCE = 1e-6
OPERATION_TOLERANCE = 1e-6


def assess(case, candidate, *, backend='pennylane', require_operation=True):
    """Return structured evidence. Benchmark truth stays outside GPT-OSS prompt.

    Primary target: output on |00...0> matches trusted reference.
    Extra regression: full unitary equivalence, as all pilot task goals specify
    implementing an operation, not just one particular output state.
    """
    try:
        spec = validate(candidate, expected_qubits=case['reference']['n_qubits'])
        expected = simulate_state(case['reference'], 'numpy')
        actual = simulate_state(spec, backend)
        target_fid = fidelity(actual, expected)
        probs = (np.abs(actual)**2).round(6).tolist()
        ideal_probs = (np.abs(expected)**2).round(6).tolist()
        op_fid = operation_fidelity(spec, case['reference'], backend) if require_operation else None
        target_pass = target_fid >= 1 - TARGET_TOLERANCE
        op_pass = (op_fid is not None and op_fid >= 1 - OPERATION_TOLERANCE)
        passed = target_pass and (op_pass if require_operation else True)
        return {'passed': bool(passed), 'valid': True, 'target_fidelity': round(target_fid, 8),
                'operation_fidelity': round(op_fid, 8) if op_fid is not None else None,
                'target_pass': bool(target_pass), 'regression_pass': bool(op_pass) if require_operation else None,
                'probabilities': probs, 'expected_probabilities': ideal_probs,
                'gate_count': len(spec['gates']), 'two_qubit_gates': two_qubit_count(spec),
                'error': ''}
    except (ValueError, CircuitError, RuntimeError, TypeError, KeyError, IndexError) as exc:
        return {'passed': False, 'valid': False, 'target_fidelity': None,
                'operation_fidelity': None, 'target_pass': False, 'regression_pass': False,
                'gate_count': None, 'two_qubit_gates': None,
                'error': f'{type(exc).__name__}: {str(exc)[:300]}'}


def feedback_for(result, strategy):
    if strategy in ('plain', 'structured'):
        return ''
    if not result['valid']:
        return 'Circuit schema or runtime error: ' + result['error']
    if strategy == 'runtime':
        return 'Circuit executed without an exception; no semantic feedback available.'
    parts = [f"Target-state fidelity: {result['target_fidelity']}; must equal 1."]
    parts.append(f"Expected output probabilities: {result['expected_probabilities']}.")
    parts.append(f"Candidate output probabilities: {result['probabilities']}.")
    parts.append('Check wire directions, phase, angles, and gate ordering.')
    if strategy == 'full':
        parts.append(f"Whole-operation fidelity: {result['operation_fidelity']}; must equal 1 to preserve behavior across input states.")
        parts.append('Correct target behavior AND whole-operation equivalence; avoid regressions.')
    return ' '.join(parts)
