import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
import scipy

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, grad, random, vmap
from jax.example_libraries import optimizers
from jax.example_libraries import stax


def deserialize_array(d):
    shape, serialized = d.split('!')
    shape = shape[1:-1].split(',')
    shape = tuple(int(x) for x in shape if x != '')
    return np.fromstring(serialized[1:-1], sep=',').reshape(shape)


def phi_inv(l0, eps, n, step_size):
    # numerically more stable version of phi
    phi = lambda s: s + np.log(1 - np.exp(-s)) - np.exp(-s) / (1 - np.exp(-s))

    ## we need to return (t, b) such that b = phi_inv(phi(l0) - (eps/n) ** 2 * t)
    ## but phi_inv would be numerically unstable, so we leverage [ u = phi(b) ] instead
    ## and [ u = c0 - (eps / n) ** 2 * t ]

    thres = l0 * 0.99
    l_data = np.hstack([
        np.logspace(-6, np.log10(thres), 10000),
        np.linspace(thres, l0 * 1.001, 5000)
        ])

    u, c0 = phi(l_data), phi(l0)
    t = (n / eps) ** 2 * (c0 - u)

    return t / step_size, l_data


for filename in sys.argv[1:]:
    fp = open(filename, "r")
    data = yaml.safe_load(fp)
    fp.close()

    eid = data["experiment_id"]
    name = f"measurements ({eid})"

    iterations = np.array([ e["iteration"] for e in data["training_data"] ])
    train_loss = np.array([ e["train_loss"] for e in data["training_data"] ])
    d, k = int(data["d"]), int(data["k"])
    step_size = float(data["step_size"])


    n, eps = int(data["n_train"]), float(data["margin"])
    del data

    l0 = train_loss[0]

    plt.figure(figsize=(6.8,3))

    plt.subplot(1,2,1)
    plt.xlabel("Iteration count")
    plt.ylabel("Loss value")

    plt.plot(iterations, train_loss, label=name, linewidth=1, zorder=3)

    bound_x, bound_y = phi_inv(l0, eps, n, step_size)
    plt.plot(bound_x, bound_y, label="predicted loss upper bound", linewidth=1, zorder=1)

    max_x = 2 * np.max(iterations)
    x_low = bound_x[np.sum(bound_y < 1e-5)]
    max_x = max(max_x, x_low)
    del bound_x, bound_y

    plt.xlim(1, max_x)
    plt.grid(alpha=.25)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize='small', loc='lower left')


    delta_t = (iterations[1:] - iterations[:-1]) * step_size
    assert np.all(delta_t > 0)
    variation = (train_loss[:-1] - train_loss[1:]) / delta_t

    plt.subplot(1,2,2)
    plt.xlabel("Loss value")
    plt.ylabel("Local loss decrease")
    plt.plot(train_loss[:-1], variation, label=name, linewidth=1, zorder=3)
    del variation


    min_l, max_l = np.min(train_loss), np.max(train_loss)
    all_loss = np.logspace(np.log10(0.2 * min_l), np.log10(2 * max_l), 1000)
    plt.plot(all_loss, (eps / n) ** 2 * (np.exp(-all_loss) - 1) ** 2, label="predicted decrease lower bound", linewidth=1, zorder=1)

    plt.xlim(0.5 * np.min(train_loss), 2 * np.max(train_loss))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(alpha=.25)
    plt.legend(fontsize='small', loc="lower right")
    plt.tight_layout()

    plt.savefig(f"logistic-{eid}.png", dpi=450)
    plt.close()
