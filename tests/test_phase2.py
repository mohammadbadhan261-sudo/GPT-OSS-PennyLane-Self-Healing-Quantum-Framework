import json
from pathlib import Path
import unittest
from unittest.mock import patch
from phase2.cli import check_suite
from phase2.core import evaluate
from phase2.study import run_case
from phase2.local_agent import LocalAgent, OllamaHTTPFailure

SUITE = Path(__file__).resolve().parent.parent/'benchmarks'/'phase2_behavior_cases.json'

class ScriptedAgent:
    model='test-agent'
    seed=42
    def __init__(self, fail_server=False):
        self.seen_feedback=[]
        self.count=0
        self.fail_server=fail_server
    def propose(self,case,*,feedback='',attempt=1):
        self.count+=1
        self.seen_feedback.append(feedback)
        if self.fail_server and self.count==1:
            raise OllamaHTTPFailure('Simulated HTTP 500')
        if feedback:
            return case['reference'],{'generated_tokens':3}
        return case['faulty'],{'generated_tokens':1}

class TestPhase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=check_suite(SUITE,'numpy')
    def test_all_14_reference_and_faults(self):
        self.assertEqual(len(self.cases),14)
        for c in self.cases:
            self.assertTrue(evaluate(c,c['reference'],backend='numpy')['passed'],c['id'])
            self.assertFalse(evaluate(c,c['faulty'],backend='numpy')['passed'],c['id'])
    def test_state_spec_allows_different_correct_unitary(self):
        c=next(x for x in self.cases if x['task_id']=='bell_phi')
        alternative={'n_qubits':2,'gates':[{'name':'H','wires':[1]},
                                             {'name':'CNOT','wires':[1,0]}]}
        e=evaluate(c,alternative,backend='numpy')
        self.assertTrue(e['passed']);self.assertIsNone(e['operation_fidelity'])
    def test_operation_rejects_same_zero_state(self):
        c=next(x for x in self.cases if x['task_id']=='phase_condition')
        self.assertEqual(evaluate(c,c['faulty'],backend='numpy')['target_fidelity'],1.0)
        self.assertFalse(evaluate(c,c['faulty'],backend='numpy')['passed'])
    def test_same_first_prompt_feedback_changes_only_after_failure(self):
        c=self.cases[0]
        a=ScriptedAgent()
        full=run_case(c,a,strategy='full',attempts=2,backend='numpy')
        self.assertTrue(full['successful'])
        self.assertTrue(full['semantic_feedback_used'])
        self.assertEqual(a.seen_feedback[0],'')
        self.assertTrue(a.seen_feedback[1])
        b=ScriptedAgent()
        plain=run_case(c,b,strategy='plain',attempts=2,backend='numpy')
        self.assertFalse(plain['successful'])
        self.assertEqual(b.seen_feedback,['',''])
    def test_transport_retry_not_counted_as_proposal(self):
        c=self.cases[0]
        a=ScriptedAgent(fail_server=True)
        result=run_case(c,a,strategy='full',attempts=2,transport_retries=1,backend='numpy')
        self.assertTrue(result['successful'])
        self.assertEqual(result['requests'],3)
        self.assertEqual(result['server_failures'],1)
        self.assertEqual(result['proposals'],2)
        self.assertTrue(result['first_proposal_success'] is False)
    def test_malformed_model_response_consumes_proposal_slot(self):
        c=self.cases[0]
        class BadJSONAgent:
            model='mock';seed=1
            def propose(self,case,*,feedback='',attempt=1):
                if attempt==1: raise ValueError('invalid JSON')
                self.last_feedback=feedback
                return case['reference'],{}
        a=BadJSONAgent()
        r=run_case(c,a,attempts=2,strategy='full',backend='numpy')
        self.assertTrue(r['successful'])
        self.assertEqual(r['attempts'][0]['kind'],'invalid_proposal')
        self.assertEqual(r['attempts'][1]['kind'],'verified_success')
        self.assertTrue(r['attempts'][1]['feedback_applied'])
        self.assertIn('structural error',r['attempts'][1]['feedback_sent'])
        self.assertEqual(a.last_feedback,r['attempts'][1]['feedback_sent'])

    def test_transport_exhausted_does_not_fake_model_output(self):
        class AlwaysError:
            model='test-agent';seed=5
            def propose(self,*args,**kwargs):raise OllamaHTTPFailure('server')
        r=run_case(self.cases[0],AlwaysError(),attempts=3,transport_retries=1,backend='numpy')
        self.assertFalse(r['successful'])
        self.assertEqual(r['proposals'],0)
        self.assertEqual(r['server_failures'],2)
        self.assertEqual(r['attempts'][0]['kind'],'transport_exhausted')
    def test_no_reference_gate_list_in_prompt(self):
        c=self.cases[0]
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):return False
        with patch('phase2.local_agent.urllib.request.urlopen') as u:
            u.return_value.__enter__.return_value = {'response':'{"n_qubits":2,"gates":[]}'}
            # json.load expects file-like, so instead capture request and raise cleanly.
            def intercept(request, timeout):
                payload=json.loads(request.data.decode())
                p=payload['prompt']
                self.assertIn(c['goal'],p)
                self.assertIn('Faulty circuit',p)
                self.assertNotIn('Reference circuit:',p)
                self.assertNotIn('H on wire 0, then CNOT',p)
                raise OllamaHTTPFailure('stop')
            u.side_effect=intercept
            with self.assertRaises(OllamaHTTPFailure):
                LocalAgent().propose(c)

if __name__=='__main__':unittest.main()
