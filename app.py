"""Optional Streamlit UI. Start: python -m streamlit run app.py"""
import streamlit as st
from selfheal.benchmark import load_cases
from selfheal.agent import OllamaAgent
from selfheal.runner import repair_case
from selfheal.verifier import assess

st.set_page_config(page_title='Self-Healing Quantum Circuits', page_icon='⚛️')
st.title('Self-Healing Quantum Circuits')
st.caption('GPT-OSS 20B via local Ollama + PennyLane default.qubit verification')
cases = load_cases()
selected = st.selectbox('Select a deliberately faulty circuit', [c['id'] for c in cases])
case = next(c for c in cases if c['id'] == selected)
st.write(case['goal'])
st.json(case['faulty'])
model = st.text_input('Local Ollama model', 'gpt-oss:20b')
strategy = st.selectbox('Feedback strategy', ['full', 'semantic', 'structured', 'runtime', 'plain'])
attempts = st.number_input('Maximum attempts', min_value=1, max_value=5, value=3)
with st.expander('Check the initial fault using PennyLane'):
    if st.button('Verify initial circuit'):
        st.json(assess(case, case['faulty'], backend='pennylane', require_operation=True))
if st.button('Run repair', type='primary'):
    with st.spinner('Running local model and verifying quantum states...'):
        record = repair_case(case, OllamaAgent(model=model), strategy=strategy,
                             attempts=int(attempts), backend='pennylane')
    st.success('Verified repair found') if record['successful'] else st.warning('No verified repair within the attempt budget')
    st.metric('Attempts used', record['attempts_used'])
    st.metric('Elapsed seconds', record['elapsed_seconds'])
    for attempt in record['attempts']:
        with st.expander(f'Attempt {attempt["attempt"]}: {"PASS" if attempt["evaluation"]["passed"] else "FAIL"}'):
            st.json(attempt)
st.info('Research prototype. Only allowlisted JSON gates are executed; no generated Python is run.')
