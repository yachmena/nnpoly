import numpy as np
from numpy.polynomial.hermite import hermval, hermder, hermgauss
from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import List, Callable
from genpoly import fejer_quadrature, lanczos, batch_polval, polder, modified_chebyshev, legendre_monic
from functools import partial
import optax
import sys
from h2s_tyuterev import poten as poten_h2s
import jaxopt


class Dense(nn.Module):
    sizes: List[int]
    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.sizes):
            kernel = self.param(f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size))
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = jax.nn.gelu(jnp.dot(x, kernel) + bias) #+ x
            # x = jax.nn.sigmoid(jnp.dot(x, kernel) + bias) #+ x
            # x = jax.nn.tanh(jnp.dot(x, kernel) + bias) #+ x
            size_ = size
        return jnp.exp(-x)
        # return jnp.abs(x) * jnp.exp(-jnp.abs(x))
        # return jnp.abs(x)


def potential_2_4(x, k2: float = 0.5, k4: float = 0.0):
    return k2 * x**2 + k4 * x**4

def potential_morse(x, de=30000, am=1, x0=1):
    return de * (1 - jnp.exp(-am * (x-x0)))**2

def potential_double_well(x, k0=-132.7074997, k2=7, k3=0.5, k4=1):
    return k0 - k2*x**2 + k3*x**3 + k4*x**4

def jac_model(model, params, x_batch):
    def jac(params):
        def jac(x):
            return jax.jacrev(model.apply, 1)(params, x)
        return jax.vmap(jac, in_axes=0)(x_batch)
    return jax.jit(jac)(params)


def solve_hermite(potential: Callable, nbas: int, nquad: int = 100):
    x, w = hermgauss(nquad)
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag([1.0 / np.sqrt(2.0**n * np.math.factorial(n)) / sqsqpi for n in range(nbas+1)])
    h = hermval(x, c)
    dh = hermval(x, hermder(c, m=1))
    dh = (dh - x * h)
    psi = h * np.sqrt(w)
    psi = psi.T
    dpsi = dh * np.sqrt(w)
    dpsi = dpsi.T
    keo = 0.5 * jnp.einsum('gi,gj->ij', dpsi, dpsi)
    pot = jnp.einsum('gi,gj,g->ij', psi, psi, potential(x))
    ham = keo + pot
    e, _ = jnp.linalg.eigh(ham)
    return e


def solve_chebyshev(poten: Callable, nbas: int, left: float, right: float,
                    nleg: int = 100, sizes: List = [128, 128, 128, 128],
                    learning_rate: float = 0.001, nepochs: int = 1000):

    global model, params
    @jax.jit
    def hamiltonian(params):
        weight_func = model.apply(params, leg_x)[:, 0]
        points, weights, alpha, beta = modified_chebyshev(
            nbas, leg_w, leg_p, leg_a, leg_b, weight_func
        )

        points_arr = jnp.array([points]).T
        wf = model.apply(params, points_arr)[:, 0]
        dwf = jac_model(model, params, points_arr)[:, 0, 0]
        sqw = jnp.sqrt(weights)

        pol = batch_polval(points, alpha, beta)
        dpol = polder(points, alpha, beta)

        psi = pol * sqw[:, None]
        dpsi = (dpol + 0.5 * pol / wf[:, None] * dwf[:, None]) * sqw[:, None] * 2.0 / (right - left)

        keo = 0.5 * jnp.einsum('gi,gj->ij', dpsi, dpsi, optimize='optimal')

        r = 0.5 * (right - left) * points + 0.5 * (right + left)
        pot = jnp.einsum('gi,gj,g->ij', psi, psi, poten(r), optimize='optimal')
        return keo + pot

    @jax.jit
    def overlap(params):
        weight_func = model.apply(params, leg_x)[:, 0]
        points, weights, alpha, beta = modified_chebyshev(
            nbas, leg_w, leg_p, leg_a, leg_b, weight_func
        )
        sqw = jnp.sqrt(weights)
        pol = batch_polval(points, alpha, beta)
        psi = pol * sqw[:, None]
        ovlp = jnp.einsum('gi,gj->ij', psi, psi, optimize='optimal')
        return ovlp

    @jax.jit
    def loss_fn(params):
        def trace(params):
            h = hamiltonian(params)
            return jnp.diag(h)
        return jnp.sum(trace(params)) 

    leg_x, leg_w, leg_p, leg_a, leg_b = legendre_monic(nbas, deg=nleg)
    leg_x = np.array([leg_x]).T

    if sizes[-1] != 1:
        sizes_ = sizes + [1]
    else:
        sizes_ = sizes
    model = Dense(sizes=sizes_)
    params = model.init(jax.random.PRNGKey(0), leg_x)

    optx = optax.adam(learning_rate=learning_rate)
    opt_state = optx.init(params)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    for i in range(nepochs):
        loss_val, grad = loss_grad_fn(params)
        updates, opt_state = optx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        h = hamiltonian(jax.lax.stop_gradient(params))
        e, _ = np.linalg.eigh(h)
        print(i, loss_val, e)


def solve_lanczos(poten: Callable, nbas: int, left: float, right: float,
                  nquad: int = 100, sizes: List = [128, 128, 128, 128],
                  learning_rate: float = 0.001, nepochs: int = 1000,
                  nstates: int = 1):

    global model, params
    @jax.jit
    def hamiltonian(params):
        weights = model.apply(params, points) * w[:, None]
        alpha, beta = lanczos(nbas, points[:, 0], weights[:, 0])

        pol = batch_polval(points[:, 0], alpha, beta)
        dpol = polder(points[:, 0], alpha, beta)

        sqw = jnp.sqrt(weights[:, 0])
        deriv_weights = jac_model(model, params, points) * w[:, None, None]
        dw = deriv_weights[:, 0, 0]

        psi = pol * sqw[:, None]
        dpsi = dpol * sqw[:, None] + 0.5 * pol / sqw[:, None] * dw[:, None]

        keo = 0.5 * jnp.einsum('gi,gj->ij', dpsi, dpsi, optimize='optimal')
        pot = jnp.einsum('gi,gj,g->ij', psi, psi, poten(points[:,0]), optimize='optimal')
        return keo + pot

    @jax.jit
    def loss_trace(params):
        def trace(params):
            h = hamiltonian(params)
            return jnp.diag(h)
        return jnp.sum(trace(params)) 

    @jax.jit
    def loss_enr(params):
        def enr(params):
            h = hamiltonian(params)
            e, _ = jnp.linalg.eigh(h)
            return e[:nstates]
        return jnp.sum(enr(params)) 

    points, w = fejer_quadrature(nquad, left, right)
    points = np.array([points]).T

    if sizes[-1] != 1:
        sizes_ = sizes + [1]
    else:
        sizes_ = sizes
    model = Dense(sizes=sizes_)
    params = model.init(jax.random.PRNGKey(0), points)

    optx = optax.adam(learning_rate=learning_rate)
    opt_state = optx.init(params)
    loss_grad_fn = jax.value_and_grad(loss_enr)

    # jaxopt_solver = jaxopt.LBFGS(fun=loss_grad_fn, value_and_grad=True, maxiter=4, verbose=True)

    enr = []
    loss = []
    for i in range(nepochs):
        loss_val, grad = loss_grad_fn(params)
        updates, opt_state = optx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # jaxopt_state = jaxopt_solver.run(params)
        # params, state = jaxopt_state
        # loss_val = loss_enr(params)

        h = hamiltonian(jax.lax.stop_gradient(params))
        e, _ = np.linalg.eigh(h)
        print(i, loss_val, e[:nstates])
        enr.append(e)
        loss.append(loss_val)

    weights = model.apply(params, points) * w[:, None]
    return points, weights, enr, loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # poten = jax.vmap(lambda r: poten_h2s(jnp.array([r, r, 1.53116836])), in_axes=(0,))
    # x = np.linspace(0.5, 2.5, 100)
    # print(poten(x).shape)
    # exit()
    # plt.plot(x, poten(x))
    # plt.ylim([0, 50000])
    # plt.show()
    # exit()
    # poten = partial(potential_2_4, k2=0.5, k4=0.0)
    poten = partial(potential_double_well, k0=0)
    x = np.linspace(-5, 5, 100)
    plt.plot(x, poten(x), 'k-')
    plt.ylim([-20, 150])
    plt.show()
    exit()
    # poten = potential_morse
    # enr = []
    # nmax = (30, 40, 50, 60, 70, 80, 90, 100)
    # for n in nmax:
    #     e0 = solve_hermite(poten, n)
    #     enr.append(e0[:30])
    #     # print(e0[:30])
    # np.savez("double_well_hermite.npz", nmax, enr)
    # exit()

    filename = "double_well.npz"
    nbas = 30
    left = -5
    right = 5
    # left = 0
    # right = 5

    # x = np.linspace(left, right, 100)
    # plt.plot(x, poten(x))
    # plt.show()

    x, w, enr, loss = solve_lanczos(
        poten, nbas, left, right, nquad=100, sizes=[512, 512, 512, 512, 512, 512],
        nstates=30, nepochs=1000, learning_rate=1e-3
    )
    np.savez(filename, x, w, enr, loss)

    # print(e0[:11])

    plt.plot(x, w)
    plt.show()
    # exit()

    # solve_chebyshev(poten, nbas, left, right, nleg=200, sizes=[512, 512, 512, 512])

    # e0 = solve_hermite(poten, 80)
    # x = np.linspace(left, right, 100)
    # plt.plot(x, poten(x))
    # for e in e0[:10]:
    #     plt.plot(x, [e]*len(x))
    #
    # print(e0[:11])
    # for nbas in [10, 20, 30, 40, 50]:
    #     e = solve_hermite(poten, nbas)
    #     print(nbas, e[:10] - e0[:10])
    #
    # plt.show()
