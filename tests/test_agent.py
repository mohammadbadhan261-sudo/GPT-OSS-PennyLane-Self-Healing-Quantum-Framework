import io
import json
import unittest
from unittest.mock import patch
from selfheal.agent import OllamaAgent
from selfheal.benchmark import get_case


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class TestOllamaInterface(unittest.TestCase):
    def test_sends_json_receives_structured_circuit(self):
        expected = get_case('bell_reverse_cnot')['reference']
        envelope = {'response': json.dumps(expected), 'eval_count': 35,
                    'prompt_eval_count': 210, 'total_duration': 100000}
        def fake_urlopen(req, timeout=None):
            payload = json.loads(req.data.decode('utf-8'))
            self.assertEqual(payload['model'],'gpt-oss:20b')
            self.assertFalse(payload['stream'])
            self.assertEqual(payload['format'],'json')
            self.assertNotIn('"reference"',payload['prompt'])
            return FakeResponse(json.dumps(envelope).encode('utf-8'))
        with patch('urllib.request.urlopen',side_effect=fake_urlopen):
            got, meta = OllamaAgent().propose(get_case('bell_reverse_cnot'),'full')
        self.assertEqual(got,expected)
        self.assertEqual(meta['generated_tokens'],35)
