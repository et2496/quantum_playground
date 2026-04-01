import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml


def sample_real_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.35, 0.08, int(0.7 * n))
    b = rng.normal(0.65, 0.06, n - len(a))
    x = np.concatenate([a, b])
    rng.shuffle(x)
    return np.clip(x, 0, 1)


def distance(real, fake):
    return abs(np.mean(real) - np.mean(fake)) + abs(np.std(real) - np.std(fake))


def hist_loss(real, fake, bins=12):
    real_hist, _ = np.histogram(real, bins=bins, range=(0, 1), density=True)
    fake_hist, _ = np.histogram(fake, bins=bins, range=(0, 1), density=True)
    return np.mean((real_hist - fake_hist) ** 2)


def plot_hist(real, fake, title):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.hist(real, bins=20, density=True, alpha=0.6, label="Real")
    ax.hist(fake, bins=20, density=True, alpha=0.6, label="Generated")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_before(results):
    return plot_hist(results["real"], results["q_before"], "Before training")


def plot_after_q(results):
    return plot_hist(results["real"], results["q_after"], "Quantum after training")


def plot_after_c(results):
    return plot_hist(results["real"], results["c_after"], "Classical after training")


def plot_distances(results):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    labels = ["Q before", "Q after", "C after"]
    vals = [results["q_before_dist"], results["q_after_dist"], results["c_after_dist"]]
    ax.bar(labels, vals)
    ax.set_title("Distance to real data")
    fig.tight_layout()
    return fig


dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def q_circuit(z, w):
    qml.RY(z[0], wires=0)
    qml.RY(z[1], wires=1)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=1)
    return qml.expval(qml.PauliZ(0))


def q_model(zs, w):
    out = [(q_circuit(z, w) + 1) / 2 for z in zs]
    return np.clip(np.array(out), 0, 1)


def c_model(zs, w):
    h1 = np.tanh(zs @ w[0:2] + w[2])
    h2 = np.tanh(zs @ w[3:5] + w[5])
    h3 = np.tanh(zs @ w[6:8] + w[8])
    h4 = np.tanh(zs @ w[9:11] + w[11])

    y = 1 / (1 + np.exp(-(h1 * w[12] + h2 * w[13] + h3 * w[14] + h4 * w[15] + w[16])))
    return np.clip(y, 0, 1)


def step_update(params, loss_fn, lr, eps=1e-2):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        p1 = params.copy()
        p2 = params.copy()
        p1[i] += eps
        p2[i] -= eps
        grads[i] = (loss_fn(p1) - loss_fn(p2)) / (2 * eps)
    return params - lr * grads


def run_usecase(steps=10, lr=0.15, seed=0):
    rng = np.random.default_rng(seed)

    # smaller = faster
    real = sample_real_data(120, seed)
    z = rng.normal(size=(80, 2))

    q_params = rng.normal(scale=0.2, size=4)
    c_params = rng.normal(scale=0.2, size=17)

    q_before = q_model(z, q_params)
    q_before_dist = distance(real, q_before)

    def q_loss(p):
        return hist_loss(real, q_model(z, p))

    def c_loss(p):
        return hist_loss(real, c_model(z, p))

    for _ in range(steps):
        q_params = step_update(q_params, q_loss, lr)
        c_params = step_update(c_params, c_loss, lr)

    q_after = q_model(z, q_params)
    c_after = c_model(z, c_params)

    return {
        "real": real,
        "q_before": q_before,
        "q_after": q_after,
        "c_after": c_after,
        "q_before_dist": distance(real, q_before),
        "q_after_dist": distance(real, q_after),
        "c_after_dist": distance(real, c_after),
    }


def explain_results(results):
    if results["q_after_dist"] < results["c_after_dist"]:
        return "Quantum matched the real demand better in this run."
    return "Classical matched the real demand better in this run."