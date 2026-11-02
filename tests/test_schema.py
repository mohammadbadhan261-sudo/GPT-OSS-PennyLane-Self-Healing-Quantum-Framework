import unittest
from selfheal.schema import validate, CircuitError


class TestCircuitSchema(unittest.TestCase):
    def test_valid(self):
        s = {'n_qubits': 2, 'gates': [{'name': 'H', 'wires': [0]}, {'name': 'CNOT', 'wires': [0,1]}]}
        self.assertEqual(validate(s), s)

    def test_reject_executable_and_unknown_fields(self):
        bads = [
            {'n_qubits': 2, 'gates': [], 'python': 'print(1)'},
            {'n_qubits': 2, 'gates': [{'name': 'EVAL', 'wires': [0]}]},
            {'n_qubits': 2, 'gates': [{'name': 'CNOT', 'wires': [0,0]}]},
            {'n_qubits': 2, 'gates': [{'name': 'H', 'wires': [9]}]},
            {'n_qubits': True, 'gates': []},
            {'n_qubits': 2, 'gates': [{'name': 'RY', 'wires': [0], 'theta': float('nan')}]},
            {'n_qubits': 2, 'gates': [{'name': 'RY', 'wires': [0], 'theta': '1.0'}]},
            {'n_qubits': 2, 'gates': [{'name': 'H', 'wires': [0]}] * 25},
        ]
        for b in bads:
            with self.subTest(b=b), self.assertRaises(CircuitError):
                validate(b)
