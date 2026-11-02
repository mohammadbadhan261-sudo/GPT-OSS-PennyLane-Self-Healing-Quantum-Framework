"""Local Ollama inference. Returns DATA ONLY, never executable Python."""
import json
import urllib.error
import urllib.request


class OllamaAgent:
    def __init__(self, model='gpt-oss:20b', url='http://127.0.0.1:11434', timeout=420,
                 num_ctx=2048, num_predict=512, seed=42):
        self.model = model
        self.base_url = url.rstrip('/')
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed

    def models(self):
        with urllib.request.urlopen(self.base_url + '/api/tags', timeout=8) as r:
            return [x['name'] for x in json.load(r).get('models', [])]

    def propose(self, case, strategy, feedback='', attempt=1):
        from .schema import ONE, TWO
        faulty_json = json.dumps(case['faulty'], sort_keys=True)
        goal = case['goal'][:1500]
        prompt = (
            'You are repairing a SMALL, quantum gate circuit represented as JSON. '
            'Return ONE JSON object only, no explanation, no markdown, no code blocks. '
            'The response object must have exactly n_qubits (integer) and gates (array). '
            'Each gate is {"name":"H","wires":[0]} or for a rotation '
            '{"name":"RY","wires":[0],"theta":1.5708}. '
            f'Valid gate names: {", ".join(sorted(ONE | TWO))}. '
            f'Use EXACTLY {case["reference"]["n_qubits"]} qubits and no more than 24 gates. '
            'CNOT wires are [control,target]. All indices start at 0. '
            'You MUST implement the entire described operation, not merely make Python execute.\n'
            f'Task specification: {goal}\n'
            f'Faulty circuit JSON: {faulty_json}\n'
        )
        if strategy != 'plain':
            prompt += ('Verify the gate order and control/target directions mentally. '
                       'Prefer a small correction preserving the intended operation.\n')
        if feedback:
            prompt += 'Independent verifier feedback from previous attempt: ' + feedback[:1200] + '\n'
        prompt += 'Corrected circuit JSON:'
        payload = {
            'model': self.model, 'prompt': prompt, 'stream': False, 'format': 'json',
            'options': {'temperature': 0.2, 'seed': self.seed + attempt - 1,
                        'num_ctx': self.num_ctx, 'num_predict': self.num_predict},
            'keep_alive': '2m'
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.base_url + '/api/generate', data=data,
                                     headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                answer = json.load(resp)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f'Ollama connection/model request failed: {exc}') from exc
        content = answer.get('response', '')
        try:
            candidate = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f'Ollama did not return valid JSON: {str(content)[:160]}') from exc
        if isinstance(candidate, dict) and set(candidate) == {'circuit'}:
            candidate = candidate['circuit']
        return candidate, {'prompt_tokens': answer.get('prompt_eval_count'),
                           'generated_tokens': answer.get('eval_count'),
                           'ollama_duration_ns': answer.get('total_duration')}
