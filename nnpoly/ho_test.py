import numpy as np
from numpy.polynomial.hermite import hermval, hermder, hermgauss
from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import List, Callable
from genpoly import fejer_quadrature, lanczos, batch_polynom, dpolynom, modified_chebyshev, legendre_monic
from functools import partial
import optax
import sys


class Dense(nn.Module):
    sizes: List[int]
    @nn.compact
    def __call__(self, x):
        # w0 = jnp.exp(-x**2)
        size_ = x.shape[-1]
        for i, size in enumerate(self.sizes):
            kernel = self.param(f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size))
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = jax.nn.relu(jnp.dot(x, kernel) + bias)
            # if i == len(self.sizes) - 1:
            #     x = (1 + x) * w0
            size_ = size
        return jnp.exp(-x)
        # return x


def potential(x, k2: float = 0.5, k4: float = 0.0):
    return k2 * x**2 + k4 * x**4


def jac_model(model, params, x_batch):
    def jac(params):
        def jac(x):
            return jax.jacrev(model.apply, 1)(params, x)
        return jax.vmap(jac, in_axes=0)(x_batch)
    return jax.jit(jac)(params)


def solve_hermite(potential: Callable, nbas: int, nquad: int = 80):
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


def solve_chebyshev(poten: Callable, nbas: int, left: float, right: float, nleg=100):

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

        pol = batch_polynom(points, alpha, beta)
        dpol = dpolynom(points, alpha, beta)

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
        pol = batch_polynom(points, alpha, beta)
        psi = pol * sqw[:, None]
        ovlp = jnp.einsum('gi,gj->ij', psi, psi, optimize='optimal')
        return ovlp


    @jax.jit
    def loss_trace(params):
        def trace(params):
            h = hamiltonian(params)
            return jnp.diag(h)
        return jnp.sum(trace(params)) 

    @jax.jit
    def loss_eigen(params):
        def enr(params):
            h = hamiltonian(params)
            e, _ = jnp.linalg.eigh(h)
            return e[:5]
        return jnp.sum(enr(params)) 


    leg_x, leg_w, leg_p, leg_a, leg_b = legendre_monic(nbas, deg=nleg)
    leg_x = np.array([leg_x]).T

    model = Dense(sizes=[64, 64, 64, 1])
    params = model.init(jax.random.PRNGKey(0), leg_x)

    h = hamiltonian(params)
    e, _ = jnp.linalg.eigh(h)
    print(e)
    # print(overlap(params))
    # sys.exit()

    optx = optax.adam(learning_rate=0.001)
    opt_state = optx.init(params)
    loss_grad_fn = jax.value_and_grad(loss_trace)

    for i in range(10000):
        loss_val, grad = loss_grad_fn(params)
        updates, opt_state = optx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        h = hamiltonian(jax.lax.stop_gradient(params))
        e, _ = np.linalg.eigh(h)
        print(i, loss_val, e[:5])


def solve_custom_poly(poten: Callable, nbas: int, left: float, right: float, nquad=100):

    @jax.jit
    def hamiltonian(params):
        weights = model.apply(params, points) * w[:, None]
        alpha, beta = lanczos(nbas, points[:, 0], weights[:, 0])

        pol = batch_polynom(points[:, 0], alpha, beta)
        dpol = dpolynom(points[:, 0], alpha, beta)

        sqw = jnp.sqrt(weights[:, 0])
        deriv_weights = jac_model(model, params, points) * w[:, None, None]
        dw = deriv_weights[:, 0, 0]

        psi = pol * sqw[:, None]
        dpsi = dpol * sqw[:, None] + 0.5 * pol / sqw[:, None] * dw[:, None]

        keo = 0.5 * jnp.einsum('gi,gj->ij', dpsi, dpsi, optimize='optimal')
        pot = jnp.einsum('gi,gj,g->ij', psi, psi, poten(points[:,0]), optimize='optimal')
        return keo + pot

    @jax.jit
    def loss_fn(params):
        def trace(params):
            h = hamiltonian(params)
            return jnp.diag(h)
        return jnp.sum(trace(params)) 


    points, w = fejer_quadrature(nquad, left, right)
    points = np.array([points]).T

    model = Dense(sizes=[64, 64, 64, 1])
    params = model.init(jax.random.PRNGKey(0), points)

    h = hamiltonian(params)
    e, _ = jnp.linalg.eigh(h)
    print(e)
    # sys.exit()

    optx = optax.adam(learning_rate=0.001)
    opt_state = optx.init(params)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    for i in range(10000):
        loss_val, grad = loss_grad_fn(params)
        updates, opt_state = optx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        h = hamiltonian(jax.lax.stop_gradient(params))
        e, _ = np.linalg.eigh(h)
        print(i, loss_val, e)

    # weights = model.apply(params, points) * w[:, None]
    # alpha, beta = lanczos(nbas, points[:, 0], weights[:, 0])
    # deriv_weights = jac_model(model, params, points) * w[:, None, None]
    # pol = batch_polynom(points[:, 0], alpha, beta)
    # dpol = dpolynom(points[:, 0], alpha, beta)
    # sqw = jnp.sqrt(weights[:, 0])
    # dw = deriv_weights[:, 0, 0]
    # psi = pol * sqw[:, None]
    # dpsi = dpol * sqw[:, None] + 0.5 * pol / sqw[:, None] * dw[:, None]
    # ovlp = jnp.einsum('gi,gj->ij', psi, psi, optimize='optimal')
    # keo = 0.5 * jnp.einsum('gi,gj->ij', dpsi, dpsi, optimize='optimal')
    # pot = jnp.einsum('gi,gj,g->ij', psi, psi, potential(points[:,0]), optimize='optimal')
    # ham = keo + pot
    # e, _ = jnp.linalg.eigh(ham)
    # print(e)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    poten = partial(potential, k4=10.0)
    nbas = 5
    left = -3
    right = 3
    solve_custom_poly(poten, nbas, left, right, nquad=100)
    # solve_chebyshev(poten, nbas, left, right, nleg=200)

    # e0 = solve_hermite(poten, 80)
    # x = np.linspace(left, right, 100)
    # plt.plot(x, poten(x))
    # for e in e0[:10]:
    #     plt.plot(x, [e]*len(x))
    #
    # print(e0[:10])
    # for nbas in [10, 20, 30, 40, 50]:
    #     e = solve_hermite(poten, nbas)
    #     print(nbas, e[:10] - e0[:10])
    #
    # plt.show()
