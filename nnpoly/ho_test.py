import numpy as np
from numpy.polynomial.hermite import hermval, hermder, hermgauss
from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import List, Callable
from genpoly import fejer_quadrature, lanczos, batch_polynom, dpolynom
from functools import partial
import optax
import sys


class Dense(nn.Module):
    sizes: List[int]
    @nn.compact
    def __call__(self, x):
        w0 = jnp.exp(-x**2)
        size_ = x.shape[-1]
        for i, size in enumerate(self.sizes):
            # kernel = self.param(f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size))
            kernel = self.param(f"w_{i}", jax.nn.initializers.zeros, (size_, size))
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = jax.nn.sigmoid(jnp.dot(x, kernel) + bias)
            size_ = size
        return x * w0


def harmonic_potential(x, k: float = 1.0):
    return 0.5 * k * x**2


def jac_x(model, params, x_batch):
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


def solve_custom_poly(potential: Callable, nbas: int, left: float, right: float, nquad=100):

    points, w = fejer_quadrature(nquad, left, right)
    points = np.array([points]).T

    model = Dense(sizes=[64, 64, 64, 1])
    params = model.init(jax.random.PRNGKey(0), points)
    weights = model.apply(params, points) * w[:, None]
    alpha, beta = lanczos(nbas, points[:, 0], weights[:, 0])
    deriv_weights = jac_x(model, params, points) * w[:, None, None]

    pol = batch_polynom(points[:, 0], alpha, beta)
    dpol = dpolynom(points[:, 0], alpha, beta)

    sqw = jnp.sqrt(weights[:, 0])
    dw = deriv_weights[:, 0, 0]
    psi = pol * sqw[:, None]
    dpsi = dpol * sqw[:, None] + 0.5 * pol / sqw[:, None] * dw[:, None]
    ovlp = jnp.einsum('gi,gj->ij', psi, psi, optimize='optimal')

    keo = 0.5 * jnp.einsum('gi,gj->ij', dpsi, dpsi, optimize='optimal')
    pot = jnp.einsum('gi,gj,g->ij', psi, psi, potential(points[:,0]), optimize='optimal')
    ham = keo + pot
    e, _ = jnp.linalg.eigh(ham)
    print(e)


if __name__ == "__main__":

    potential = partial(harmonic_potential, k=1.0)
    nbas = 10
    left = -10.0
    right = 10.0
    solve_custom_poly(potential, nbas, left, right, nquad=200)
    # solve_hermite(potential, nbas)
