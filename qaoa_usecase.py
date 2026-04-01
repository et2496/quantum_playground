import time
from itertools import product

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def get_demo_graph():
    """
    - nodes = stops
    - edges = traffic
    """
    n_wires = 4
    edges = [(0, 1), (0, 3), (1, 2), (2, 3)]

    return n_wires, edges

def plot_init(n_wires, edges, bitstring=None, title="Graph"):

    graph = nx.Graph()
    graph.add_nodes_from(range(n_wires))
    graph.add_edges_from(edges)

    pos = nx.spring_layout(graph, seed=7)

    if bitstring is None:
        node_colors = ["lightgray"] * n_wires
    else:
        bits = [int(b) for b in bitstring]
        node_colors = ["orange" if b == 1 else "lightblue" for b in bits]

    fig, ax = plt.subplots(figsize=(4, 4))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=900,
        font_size=12,
        edgecolors="black",
        ax=ax,
    )
    ax.set_title(title)
    return fig

def plot_graph_solution(n_wires, edges, bitstring=None, title="Graph"):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_wires))
    graph.add_edges_from(edges)

    pos = nx.spring_layout(graph, seed=7)

    if bitstring is None:
        node_colors = ["lightgray"] * n_wires
    else:
        bits = [int(b) for b in bitstring]
        node_colors = ["orange" if b == 1 else "lightblue" for b in bits]

    fig, ax = plt.subplots(figsize=(4, 4))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=900,
        font_size=12,
        edgecolors="black",
        ax=ax,
    )
    ax.set_title(title)

    return fig 

def cut_value(bitstring, edges):
    """Return the cut value for one bitstring."""
    value = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            value += 1
    return value


def brute_force_maxcut(n_wires, edges):
    best_value = -1
    best_bitstring = None

    for bits in product([0, 1], repeat=n_wires):
        value = cut_value(bits, edges)
        if value > best_value:
            best_value = value
            best_bitstring = "".join(map(str, bits))

    return best_value, best_bitstring


def measure_bruteforce_times():
    import time

    node_sizes_measured = [4, 8, 10, 12]
    node_sizes_extended = [14, 16, 64]

    rows = []

    # --- measure real runtimes ---
    for n in node_sizes_measured:
        edges = [(i, (i + 1) % n) for i in range(n)]

        start = time.perf_counter()
        brute_force_maxcut(n, edges)
        elapsed = time.perf_counter() - start

        rows.append({
            "nodes": n,
            "search_space": 2 ** n,
            "time_seconds": elapsed,
            "type": "measured"
        })

    # --- estimate larger sizes ---
    # use last measured time as baseline
    last = rows[-1]
    base_n = last["nodes"]
    base_time = last["time_seconds"]

    for n in node_sizes_extended:
        # exponential scaling: time doubles for each added node
        estimated_time = base_time * (2 ** (n - base_n))

        rows.append({
            "nodes": n,
            "search_space": 2 ** n,
            "time_seconds": estimated_time,
            "type": "estimated"
        })

    return rows


def build_qaoa_objective(n_wires, edges, p=1):

    dev = qml.device("default.qubit", wires=n_wires)

    def cost_layer(gamma):
        for i, j in edges:
            qml.CNOT(wires=[i, j])
            qml.RZ(gamma, wires=j)
            qml.CNOT(wires=[i, j])

    def mixer_layer(beta):
        for wire in range(n_wires):
            qml.RX(2 * beta, wires=wire)

    @qml.qnode(dev)
    def zz_expval_circuit(gammas, betas):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        for layer in range(p):
            cost_layer(gammas[layer])
            mixer_layer(betas[layer])

        zz_sum = qml.sum(*(qml.Z(i) @ qml.Z(j) for i, j in edges))
        return qml.expval(zz_sum)

    def expected_cut(params):
        gammas = params[0]
        betas = params[1]
        zz_expectation = zz_expval_circuit(gammas, betas)
        return 0.5 * (len(edges) - zz_expectation)

    def objective(params):
        # optimizer minimizes, so negate expected cut
        return -expected_cut(params)

    return objective, expected_cut


def build_sampling_circuit(n_wires, edges, p=1, shots=100):
    dev = qml.device("default.qubit", wires=n_wires, shots=shots)

    def cost_layer(gamma):
        for i, j in edges:
            qml.CNOT(wires=[i, j])
            qml.RZ(gamma, wires=j)
            qml.CNOT(wires=[i, j])

    def mixer_layer(beta):
        for wire in range(n_wires):
            qml.RX(2 * beta, wires=wire)

    @qml.qnode(dev)
    def sample_circuit(gammas, betas):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        for layer in range(p):
            cost_layer(gammas[layer])
            mixer_layer(betas[layer])

        return qml.sample()

    return sample_circuit


def run_qaoa_usecase(p=1, steps=20, learning_rate=0.5, shots=100):
    n_wires, edges = get_demo_graph()
    initial_fig = plot_graph_solution(
        n_wires,
        edges,
        title="Initial Traffic Graph"
    )

    classical_best_cut, classical_best_bitstring = brute_force_maxcut(n_wires, edges)
    classical_fig = plot_graph_solution(
        n_wires,
        edges,
        classical_best_bitstring,
        title=f"Classical Best: {classical_best_bitstring}"
    )
    search_space = 2 ** n_wires

    objective, expected_cut = build_qaoa_objective(
        n_wires=n_wires,
        edges=edges,
        p=p,
    )

    params = 0.01 * np.random.rand(2, p, requires_grad=True)
    opt = qml.AdagradOptimizer(stepsize=learning_rate)

    expected_cut_history = []

    for _ in range(steps):
        params = opt.step(objective, params)
        current_expected_cut = float(expected_cut(params))
        expected_cut_history.append(current_expected_cut)

    final_expected_cut = expected_cut_history[-1]
    approximation_ratio = final_expected_cut / classical_best_cut

    # optional sampled candidate for display
    sample_circuit = build_sampling_circuit(n_wires, edges, p=p, shots=shots)
    gammas = params[0]
    betas = params[1]
    samples = sample_circuit(gammas, betas)

    bitstrings = ["".join(str(int(x)) for x in sample) for sample in samples]
    counts = {}
    for b in bitstrings:
        counts[b] = counts.get(b, 0) + 1
    most_likely_bitstring = max(counts, key=counts.get)
    most_likely_cut = cut_value(tuple(int(b) for b in most_likely_bitstring), edges)

    qaoa_fig = plot_graph_solution(
        n_wires,
        edges,
        most_likely_bitstring,
        title=f"QAOA Result: {most_likely_bitstring}"
    )

    return {
        "n_wires": n_wires,
        "edges": edges,
        "classical_best_cut": classical_best_cut,
        "classical_best_bitstring": classical_best_bitstring,
        "classical_search_space": search_space,
        "qaoa_expected_cut": final_expected_cut,
        "approximation_ratio": approximation_ratio,
        "expected_cut_history": expected_cut_history,
        "most_likely_bitstring": most_likely_bitstring,
        "most_likely_cut": most_likely_cut,
        "initial_fig": initial_fig,
        "classical_fig": classical_fig,
        "qaoa_fig": qaoa_fig,
    }


def explain_qaoa_results(results):
    ratio = results["approximation_ratio"]

    if ratio >= 0.99:
        return "Perfect."
    if ratio >= 0.6:
        return "Good."
    return "QAOA far from the classical optimum. Try more steps or more layers."