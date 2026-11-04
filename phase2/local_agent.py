"""Direct, local Ollama API calls. Generated programs are data-only JSON.

Every strategy gets exactly the SAME initial prompt and seed. Strategies
change only what validation feedback is visible on subsequent attempts.
"""
import json
import urllib.error
import urllib.request
from selfheal.schema import ONE, TWO


class OllamaHTTPFailure(RuntimeError):
    """No candidate was produced because Ollama failed to respond."""


class LocalAgent:
    def __init__(self, model='gpt-oss:20b', url='http://127.0.0.1:11434',
                 timeout=420, num_ctx=2048, num_predict=512, seed=42):
        self.model = model
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed

    def propose(self, case, *, feedback='', attempt=1):
        n = case['faulty']['n_qubits']
        goal = case['goal']
        task_type = case['task_type']
        semantic = ('Only the quantum state produced from initial |0...0> is specified. '
                    'A different circuit with the same output state is acceptable.'
                    if task_type == 'state' else
                    'The requested transformation must be correct for ALL inputs, up to ONE global phase.')
        prompt = (
            'Repair a quantum gate circuit defined as DATA. Do not output Python, analysis, markdown, or an explanation. '
            'Output a JSON object with EXACTLY two keys: n_qubits and gates. '
            'Each gate uses {"name":"H","wires":[0]} for non-rotation gates; '
            'RX/RY/RZ gates additionally require numeric theta in radians. '
            f'Allowed gate names: {", ".join(sorted(ONE|TWO))}. '
            f'Use {n} qubits (wire indices 0 to {n-1}); maximum 24 gates. '
            'CNOT wire order is [control, target]. '
            f'{semantic}\n'
            f'Task: {goal}\n'
            f'Faulty circuit: {json.dumps(case["faulty"], separators=(",", ":"))}\n'
        )
        if feedback:
            prompt += 'Verifier observations for the previous candidate: ' + feedback[:1500] + '\n'
        prompt += 'Corrected circuit JSON:'
        payload = {'model': self.model, 'prompt': prompt, 'stream': False,
                   'format': 'json', 'keep_alive': '2m',
                   'options': {'temperature': 0.2, 'seed': self.seed + attempt - 1,
                               'num_ctx': self.num_ctx, 'num_predict': self.num_predict}}
        req = urllib.request.Request(self.url+'/api/generate', data=json.dumps(payload).encode(),
                                     headers={'Content-Type':'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                answer = json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise OllamaHTTPFailure(f'{type(exc).__name__}: {str(exc)[:200]}') from exc
        raw = answer.get('response', '')
        try:
            candidate = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid JSON from LLM: {str(raw)[:150]}') from exc
        if isinstance(candidate, dict) and set(candidate)=={'circuit'}:
            candidate = candidate['circuit']
        usage = {'prompt_tokens': answer.get('prompt_eval_count'),
                 'generated_tokens': answer.get('eval_count'),
                 'ollama_duration_ns': answer.get('total_duration')}
        return candidate, usage
