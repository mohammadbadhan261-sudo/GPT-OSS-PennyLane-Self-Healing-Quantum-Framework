"""Print the circuit and optionally save a PNG using PennyLane.
Usage: python draw_circuit.py --case bell_reverse_cnot --which faulty
"""
import argparse
from pathlib import Path
from selfheal.benchmark import get_case
from selfheal.schema import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='bell_reverse_cnot')
    parser.add_argument('--which', choices=['reference','faulty'],default='faulty')
    parser.add_argument('--png',action='store_true',help='Save a labeled PNG (matplotlib required)')
    a=parser.parse_args()
    import pennylane as qml
    case = get_case(a.case)
    spec = validate(case[a.which])
    dev = qml.device('default.qubit',wires=spec['n_qubits'])
    @qml.qnode(dev)
    def circuit():
        for g in spec['gates']:
            n,w = g['name'],g['wires']
            name_map={'H':'Hadamard','X':'PauliX','Y':'PauliY','Z':'PauliZ'}
            operation = getattr(qml,name_map.get(n,n))
            if 'theta' in g:
                operation(g['theta'], wires=w[0])
            else:
                operation(wires=w if len(w)==2 else w[0])
        return qml.probs(wires=range(spec['n_qubits']))
    print(qml.draw(circuit)())
    if a.png:
        import matplotlib.pyplot as plt
        fig, ax = qml.draw_mpl(circuit)()
        path = Path('results') / f'{a.case}_{a.which}.png'
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path,dpi=180,bbox_inches='tight')
        print('Saved:',path)
        plt.close(fig)


if __name__ == '__main__': main()
