import numpy as np
import orthopy
import quadpy
from typing import Callable
from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp


def legendre_monic(N: int, deg: int = 100):
    scheme = quadpy.c1.gauss_legendre(deg)
    x = scheme.points
    w = scheme.weights
    eval = orthopy.c1.legendre.Eval(x, "monic")
    leg = np.array([next(eval) for _ in range(2 * N)])
    rec_coefs = orthopy.c1.legendre.RecurrenceCoefficients("monic", symbolic=False)
    a = jnp.array([rec_coefs[k][1] for k in range(2 * N)])
    b = jnp.array([rec_coefs[k][2] for k in range(2 * N)])
    return x, w, leg, a, b


@jax.jit
def modified_chebyshev(w, poly, a, b, wf):

    # modified moments nu[k], k = 0..2n-1
    nu = jnp.sum(poly * w * wf, axis=-1)

    # recurrence coefficients alpha[k], beta[k], k = 0..n-1

    k = 0
    alpha = jnp.zeros(N, dtype=np.float64)
    beta = jnp.zeros(N, dtype=np.float64)
    alpha = alpha.at[k].set(a[0] + nu[1] / nu[0])
    beta = beta.at[k].set(nu[0])
    sigma_k_min_1 = nu

    k = 1
    sigma_k = jnp.array([nu[l+1] - (alpha[0] - a[l]) * nu[l] + b[l] * nu[l-1]
                for l in range(k, 2 * N - k)])
    alpha = alpha.at[k].set(
        a[k] + sigma_k[1] / sigma_k[0] - sigma_k_min_1[1] / sigma_k_min_1[0]
    )
    beta = beta.at[k].set(
        sigma_k[0] / sigma_k_min_1[0]
    )

    for k in range(2, N):
        sigma_k_min_2 = sigma_k_min_1
        sigma_k_min_1 = sigma_k
        sigma_k = jnp.array([sigma_k_min_1[il+2] - (alpha[k-1] - a[l]) * sigma_k_min_1[il+1] \
                                - beta[k-1] * sigma_k_min_2[il+2] + b[l] * sigma_k_min_1[il]
                                for il, l in  enumerate(range(k, 2 * N - k))])
        alpha = alpha.at[k].set(
            a[k] + sigma_k[1] / sigma_k[0] - sigma_k_min_1[1] / sigma_k_min_1[0]
        )
        beta = beta.at[k].set(
            sigma_k[0] / sigma_k_min_1[0]
        )

    # quadrature
    sqrt_beta = jnp.sqrt(beta)
    jac = jnp.diag(alpha) + jnp.diag(sqrt_beta[1:], k=1) + jnp.diag(sqrt_beta[1:], k=-1)
    points, v = jnp.linalg.eigh(jac)
    weights = beta[0] * v[0, :]**2

    return points, weights, alpha, beta


@jax.jit
def polynom(x, alpha, beta):
    p = jnp.zeros((len(alpha), len(x)), dtype=np.float64)
    p = p.at[0].set(1)
    p = p.at[1].set(x - alpha[0])
    for k in range(2, len(alpha)):
        p = p.at[k].set((x - alpha[k-1]) * p[k-1] - beta[k-1] * p[k-2])
    norm = jnp.array([1.0 / jnp.sqrt(jnp.prod(beta[:n+1])) for n in range(len(beta))])
    return p * norm[:, None]


def test_modified_chebyshev(weight_func: Callable, N: int):

    scheme = quadpy.c1.gauss_legendre(100)
    x = scheme.points
    eval = orthopy.c1.legendre.Eval(x, "monic")
    wfunc = weight_func(x)
    moments = []
    for i in range(2 * N):
        leg = next(eval)
        val = np.sum(wfunc * leg * scheme.weights)
        moments.append(val)

    rc = orthopy.c1.legendre.RecurrenceCoefficients("monic", symbolic=True)
    rc = [rc[k] for k in range(2 * N)]

    alpha, beta, int_1 = orthopy.tools.chebyshev_modified(moments, rc)
    beta[0] = int_1
    points, weights = quadpy.tools.scheme_from_rc(alpha, beta, int_1, mode="numpy") # or "mpmath", "sympy"

    return points, weights, alpha, beta


def fejer_lanczos(N: int, left: float, right: float, nquad: int, wf: Callable):
    scheme = quadpy.c1.fejer_1(nquad)
    x = scheme.points
    w = scheme.weights
    x = 0.5 * ((right - left) * x + right + left)
    w = w * wf(x) * 0.5 * (right - left)
    alpha, beta = lancz(N+1, x, w)
    return x, w, alpha, beta


def lancz(n, x, w):
    alpha = jnp.array(x)
    beta = jnp.zeros_like(alpha)
    beta = beta.at[0].set(w[0])
    for i in range(len(x)-1):
        dpi = w[i+1]
        dgam = 1.0
        dsig = 0.0
        dt = 0.0
        xlam = x[i+1]
        for k in range(i+2):
            drho = beta[k] + dpi
            dtmp = dgam * drho
            dtsig = dsig
            if drho <= 0.0:
                dgam = 1.0
                dsig = 0.0
            else:
                dgam = beta[k] / drho
                dsig = dpi / drho
            dtk = dsig * (alpha[k] - xlam) - dgam * dt
            alpha = alpha.at[k].set(alpha[k] - (dtk - dt))
            dt = dtk
            if dsig <= 0:
                dpi = dtsig * beta[k]
            else:
                dpi = dt**2 / dsig
            dtsig = dsig
            beta = beta.at[k].set(dtmp)
    return alpha[:n], beta[:n]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    weight_func = lambda x: jnp.exp(-x**2)
    N = 20

    # Method 1, starting from quadrature
    left = 0.001
    right = 10
    nquad = 60
    points, weights, alpha, beta = fejer_lanczos(N, left, right, nquad, weight_func)

    # Method 2, starting from moment integrals
    # x, w, leg, a, b = legendre_monic(N)
    # wf = weight_func(x)
    # points, weights, alpha, beta = modified_chebyshev(w, leg, a, b, wf)

    # points2, weights2, alpha2, beta2 = test_modified_chebyshev(weight_func, N)
    # print(np.array(alpha)-alpha2)
    # print(np.array(beta)-beta2)
    # print(np.array(points)-points2)
    # print(np.array(weights)-weights2)

    # overlap integrals
    p = polynom(points, alpha, beta)
    ovlp = jnp.einsum('ig,jg,g->ij', p, p, weights)
    print("overlap")
    ovlp_off = ovlp - np.diag(np.diag(ovlp))
    for i in range(len(ovlp)):
        print("i = ", i, "max offdiag = ", np.max(np.abs(ovlp_off[i])), "diag = ", ovlp[i, i])

    # plot
    for pp in p[:5]:
        plt.plot(points, pp*np.sqrt(weights))
    plt.show()
