"""Load trusted, committed benchmark. Reference never enters the LLM prompt."""
import json
from pathlib import Path
from .schema import validate

DEFAULT_CASES = Path(__file__).resolve().parent.parent / 'benchmarks' / 'pilot_cases.json'


def load_cases(path=DEFAULT_CASES):
    with Path(path).open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Benchmark must be a list')
    ids = set()
    for item in data:
        if item['id'] in ids:
            raise ValueError('Duplicate case id')
        ids.add(item['id'])
        if type(item['goal']) is not str or not item['goal'].strip():
            raise ValueError('Missing task description')
        ref = validate(item['reference'])
        bad = validate(item['faulty'], expected_qubits=ref['n_qubits'])
        if bad == ref:
            raise ValueError('Faulty circuit identical to reference')
    return data


def get_case(case_id, path=DEFAULT_CASES):
    for c in load_cases(path):
        if c['id'] == case_id:
            return c
    raise KeyError(f'Unknown case: {case_id}')
