"""Strict allowlisted circuit representation. No generated Python is ever executed."""
import math

ONE = {'H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ'}
TWO = {'CNOT', 'CZ', 'SWAP'}
PARAMETRIC = {'RX', 'RY', 'RZ'}
MAX_QUBITS = 4
MAX_GATES = 24


class CircuitError(ValueError):
    pass


def validate(spec, *, expected_qubits=None):
    if not isinstance(spec, dict) or set(spec) != {'n_qubits', 'gates'}:
        raise CircuitError('Circuit must be JSON object with exactly n_qubits and gates')
    n = spec['n_qubits']
    if type(n) is not int or not 2 <= n <= MAX_QUBITS:
        raise CircuitError('n_qubits must be an integer from 2 to 4')
    if expected_qubits is not None and n != expected_qubits:
        raise CircuitError('n_qubits differs from the benchmark requirement')
    gates = spec['gates']
    if not isinstance(gates, list) or not 0 <= len(gates) <= MAX_GATES:
        raise CircuitError(f'gates must be a list with at most {MAX_GATES} items')
    clean = []
    for i, g in enumerate(gates):
        if not isinstance(g, dict):
            raise CircuitError(f'gate {i}: expected JSON object')
        name = g.get('name')
        if type(name) is not str or name not in ONE | TWO:
            raise CircuitError(f'gate {i}: unsupported gate name {name!r}')
        allowed_keys = {'name', 'wires', 'theta'} if name in PARAMETRIC else {'name', 'wires'}
        if set(g) != allowed_keys:
            raise CircuitError(f'gate {i}: expected keys {sorted(allowed_keys)}')
        wires = g['wires']
        expected_count = 2 if name in TWO else 1
        if not isinstance(wires, list) or len(wires) != expected_count or any(type(w) is not int or w < 0 or w >= n for w in wires) or len(set(wires)) != len(wires):
            raise CircuitError(f'gate {i}: invalid qubit indices')
        entry = {'name': name, 'wires': list(wires)}
        if name in PARAMETRIC:
            theta = g['theta']
            if type(theta) not in (int, float) or not math.isfinite(theta) or abs(theta) > 16 * math.pi:
                raise CircuitError(f'gate {i}: theta must be a finite number in [-16*pi,16*pi]')
            entry['theta'] = float(theta)
        clean.append(entry)
    return {'n_qubits': n, 'gates': clean}


def circuit_gate_count(spec):
    return len(spec['gates'])


def two_qubit_count(spec):
    return sum(g['name'] in TWO for g in spec['gates'])
