import pennylane as qml
from pennylane import numpy as np


def load_h2_problem():
    dataset = qml.data.load("qchem", molname="H2")[0]

    hamiltonian = dataset.hamiltonian
    exact_energy = float(dataset.fci_energy)

    n_qubits = len(hamiltonian.wires)
    hf_state = qml.qchem.hf_state(2, n_qubits)

    return hamiltonian, exact_energy, n_qubits, hf_state


def build_vqe_cost_function(hamiltonian, n_qubits, hf_state):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost_fn(theta):
        qml.BasisState(hf_state, wires=range(n_qubits))
        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return qml.expval(hamiltonian)

    return cost_fn


def run_vqe_usecase(steps=25, learning_rate=0.2, start_theta=0.0):
    hamiltonian, exact_energy, n_qubits, hf_state = load_h2_problem()
    cost_fn = build_vqe_cost_function(hamiltonian, n_qubits, hf_state)

    opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
    theta = np.array(start_theta, requires_grad=True)

    energy_history = []

    for _ in range(steps):
        theta = opt.step(cost_fn, theta)
        energy = float(cost_fn(theta))
        energy_history.append(energy)

    quantum_energy = energy_history[-1]
    error = abs(quantum_energy - exact_energy)

    return {
        "quantum_energy": quantum_energy,
        "exact_energy": exact_energy,
        "error": error,
        "energy_history": energy_history,
    }


def explain_results_vqe(error):
    if error < 0.001:
        return "Excellent: the quantum result is very close to the classical solution."
    elif error < 0.01:
        return "Good: the quantum result is close to the classical solution."
    else:
        return "The quantum result is still far from the classical solution."