"""Independent NumPy oracle plus PennyLane statevector verification backend.

The numpy backend is for local unit tests and cross-checks. Actual user research
should use --backend pennylane and record that backend in every result.
"""
import cmath
import math
import numpy as np
from .schema import validate

SINGLE = {
    'H': np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.diag([1, -1]).astype(complex),
    'S': np.diag([1, 1j]).astype(complex),
    'T': np.diag([1, cmath.exp(1j * math.pi / 4)]).astype(complex),
}


def _matrix(g):
    name = g['name']
    if name in SINGLE:
        return SINGLE[name]
    t = g['theta']
    c, s = math.cos(t / 2), math.sin(t / 2)
    if name == 'RX':
        return np.array([[c, -1j * s], [-1j * s, c]])
    if name == 'RY':
        return np.array([[c, -s], [s, c]])
    if name == 'RZ':
        return np.diag([cmath.exp(-1j * t / 2), cmath.exp(1j * t / 2)])
    raise ValueError('Unsupported matrix')


def _state_numpy(spec, basis_index=0):
    spec = validate(spec)
    n = spec['n_qubits']
    size = 2 ** n
    state = np.zeros(size, dtype=complex)
    state[basis_index] = 1
    for g in spec['gates']:
        name, wires = g['name'], g['wires']
        if name in SINGLE or name in ('RX', 'RY', 'RZ'):
            w = wires[0]
            bit = 1 << (n - 1 - w)
            matrix = _matrix(g)
            for i in range(size):
                if not (i & bit):
                    j = i | bit
                    a, b = state[i], state[j]
                    state[i] = matrix[0, 0] * a + matrix[0, 1] * b
                    state[j] = matrix[1, 0] * a + matrix[1, 1] * b
        else:
            w0, w1 = (1 << (n - 1 - x) for x in wires)
            output = np.zeros_like(state)
            for i in range(size):
                j = i
                if name == 'CNOT' and (i & w0):
                    j ^= w1
                elif name == 'CZ' and (i & w0) and (i & w1):
                    output[i] -= state[i]
                    continue
                elif name == 'SWAP' and bool(i & w0) != bool(i & w1):
                    j ^= w0 | w1
                output[j] += state[i]
            state = output
    return state


def _state_pennylane(spec, basis_index=0):
    try:
        import pennylane as qml
    except ImportError as exc:
        raise RuntimeError('PennyLane is missing: python -m pip install -r requirements.txt') from exc
    n = spec['n_qubits']
    dev = qml.device('default.qubit', wires=n, shots=None)

    @qml.qnode(dev)
    def qnode():
        if basis_index:
            bits = [(basis_index >> (n - 1 - i)) & 1 for i in range(n)]
            qml.BasisState(np.array(bits), wires=list(range(n)))
        for g in spec['gates']:
            name, wires = g['name'], g['wires']
            if name == 'CNOT':
                qml.CNOT(wires=wires)
            elif name == 'CZ':
                qml.CZ(wires=wires)
            elif name == 'SWAP':
                qml.SWAP(wires=wires)
            elif name == 'H':
                qml.Hadamard(wires=wires[0])
            elif name == 'X':
                qml.PauliX(wires=wires[0])
            elif name == 'Y':
                qml.PauliY(wires=wires[0])
            elif name == 'Z':
                qml.PauliZ(wires=wires[0])
            elif name == 'S':
                qml.S(wires=wires[0])
            elif name == 'T':
                qml.T(wires=wires[0])
            elif name == 'RX':
                qml.RX(g['theta'], wires=wires[0])
            elif name == 'RY':
                qml.RY(g['theta'], wires=wires[0])
            elif name == 'RZ':
                qml.RZ(g['theta'], wires=wires[0])
        return qml.state()
    return np.asarray(qnode(), dtype=complex)


def simulate_state(spec, backend='pennylane', basis_index=0):
    spec = validate(spec)
    if type(basis_index) is not int or not 0 <= basis_index < 2 ** spec['n_qubits']:
        raise ValueError('Invalid basis_index')
    if backend == 'pennylane':
        return _state_pennylane(spec, basis_index)
    if backend == 'numpy':
        return _state_numpy(spec, basis_index)
    raise ValueError('backend must be pennylane or numpy')


def fidelity(a, b):
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    return float(min(1.0, abs(np.vdot(a, b)) ** 2 / (np.vdot(a,a).real * np.vdot(b,b).real)))


def unitary_of(spec, backend='pennylane'):
    n = spec['n_qubits']
    return np.column_stack([simulate_state(spec, backend, basis_index=i) for i in range(2 ** n)])


def operation_fidelity(candidate, reference, backend='pennylane'):
    u = unitary_of(candidate, backend)
    v = unitary_of(reference, 'numpy')
    d = u.shape[0]
    # Trace overlap is insensitive to one global phase of the whole operation.
    return float(min(1.0, abs(np.trace(v.conj().T @ u)) ** 2 / d ** 2))
