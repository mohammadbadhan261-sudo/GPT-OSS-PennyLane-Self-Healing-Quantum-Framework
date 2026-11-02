import unittest
import math
import numpy as np
from selfheal.quantum import simulate_state, fidelity, unitary_of, operation_fidelity
from selfheal.benchmark import load_cases
from selfheal.verifier import assess


class TestQuantumOracle(unittest.TestCase):
    def test_bell_state(self):
        c = {'n_qubits': 2, 'gates': [{'name':'H','wires':[0]}, {'name':'CNOT','wires':[0,1]}]}
        state = simulate_state(c,'numpy')
        np.testing.assert_allclose(np.abs(state)**2, [0.5,0,0,0.5],atol=1e-12)
        self.assertAlmostEqual(fidelity(state,state), 1)
        np.testing.assert_allclose(unitary_of(c,'numpy').conj().T @ unitary_of(c,'numpy'), np.eye(4), atol=1e-12)

    def test_all_12_mutants_actually_fail(self):
        cases = load_cases()
        self.assertEqual(len(cases),12)
        for case in cases:
            with self.subTest(case=case['id']):
                ref = assess(case,case['reference'],backend='numpy')
                faulty = assess(case,case['faulty'],backend='numpy')
                self.assertTrue(ref['passed'])
                self.assertFalse(faulty['passed'])
                self.assertLess(faulty['target_fidelity'],0.99)

    def test_global_phase_invariant(self):
        c = {'n_qubits': 2, 'gates': [{'name':'X','wires':[0]}]}
        # Z followed by X then Z differs by a global -1 from X.
        d = {'n_qubits': 2, 'gates': [{'name':'Z','wires':[0]}, {'name':'X','wires':[0]}, {'name':'Z','wires':[0]}]}
        self.assertAlmostEqual(operation_fidelity(c,d,'numpy'),1.0)


class TestPennyLaneIntegration(unittest.TestCase):
    @unittest.skipUnless(__import__('importlib').util.find_spec('pennylane'), 'PennyLane not installed in test container')
    def test_pennylane_matches_independent_oracle(self):
        for c in load_cases()[:4]:
            with self.subTest(case=c['id']):
                for basis in range(2**c['reference']['n_qubits']):
                    np.testing.assert_allclose(
                        simulate_state(c['reference'],'pennylane',basis),
                        simulate_state(c['reference'],'numpy',basis),atol=1e-7)
