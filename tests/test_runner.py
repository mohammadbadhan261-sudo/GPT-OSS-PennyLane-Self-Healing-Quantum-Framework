import unittest
from selfheal.benchmark import get_case
from selfheal.runner import repair_case, summarize


class FakeAgent:
    model = 'mock-agent-for-test'
    def propose(self, case, strategy, feedback='', attempt=1):
        if attempt == 1:
            return case['faulty'], {}
        return case['reference'], {'generated_tokens': 123}


class TestRunner(unittest.TestCase):
    def test_feedback_repair(self):
        r=repair_case(get_case('bell_reverse_cnot'), FakeAgent(),strategy='full',attempts=2,backend='numpy')
        self.assertTrue(r['successful'])
        self.assertEqual(r['attempts_used'],2)
        self.assertEqual(summarize([r])['verified_repair_rate'],1.0)

    def test_unrepaired_without_retry(self):
        r=repair_case(get_case('bell_reverse_cnot'), FakeAgent(),strategy='plain',attempts=1,backend='numpy')
        self.assertFalse(r['successful'])

    def test_compound(self):
        r=repair_case(get_case('rotation_compound'), FakeAgent(),strategy='full',attempts=2,backend='numpy')
        self.assertTrue(r['successful'])
