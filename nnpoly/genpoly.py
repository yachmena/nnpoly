import numpy as np
import orthopy
import quadpy
from typing import Callable
from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp
from functools import partial


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


def fejer_quadrature(deg: int, left: float, right: float):
    scheme = quadpy.c1.fejer_1(deg)
    x = scheme.points
    w = scheme.weights
    if np.isinf(left) and np.isinf(right):
        # from [-1, 1] to [-inf, inf]
        x = x / (1.0 - x**2)
        w = w * (x**2 + 1.0) / (1.0 - x**2)**2
    elif np.isinf(right):
        # from [-1, 1] to [0, inf]
        x = left + (1.0 + x) / (1.0 - x)
        w = w * 2.0 / (1.0 - x)**2
    elif np.isinf(left):
        # from [-1, 1] to [-inf, 0]
        x = right - (1.0 - x) / (1.0 + x)
        w = w * 2.0 / (1.0 + x)**2
    else:
        x = 0.5 * ((right - left) * x + right + left)
        w = w * 0.5 * (right - left)
    return x, w


@partial(jax.jit, static_argnums=(0,))
def modified_chebyshev(N, w, poly, a, b, wf):

    def rec_1(carry, l):
        il, sigma = carry
        sigma = sigma.at[il].set(
            nu[l+1] - (alpha[0] - a[l]) * nu[l] + b[l] * nu[l-1]
        )
        return (il+1, sigma), 0

    def rec_k(carry, l):
        il, sigma = carry
        sigma = sigma.at[il].set(
            sigma_k_min_1[il+2] - (alpha[k-1] - a[l]) * sigma_k_min_1[il+1] \
            - beta[k-1] * sigma_k_min_2[il+2] + b[l] * sigma_k_min_1[il]
        )
        return (il+1, sigma), 0

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
    (_, sigma_k), _ = jax.lax.scan(rec_1, (0, jnp.zeros(len(np.arange(k, 2*N-k)))), np.arange(k, 2*N-k))
    # sigma_k = jnp.array([nu[l+1] - (alpha[0] - a[l]) * nu[l] + b[l] * nu[l-1]
    #             for l in range(k, 2 * N - k)])
    alpha = alpha.at[k].set(
        a[k] + sigma_k[1] / sigma_k[0] - sigma_k_min_1[1] / sigma_k_min_1[0]
    )
    beta = beta.at[k].set(
        sigma_k[0] / sigma_k_min_1[0]
    )

    for k in range(2, N):
        sigma_k_min_2 = sigma_k_min_1
        sigma_k_min_1 = sigma_k
        (_, sigma_k), _ = jax.lax.scan(rec_k, (0, jnp.zeros(len(np.arange(k, 2*N-k)))), np.arange(k, 2*N-k))
        # sigma_k = jnp.array([sigma_k_min_1[il+2] - (alpha[k-1] - a[l]) * sigma_k_min_1[il+1] \
        #                         - beta[k-1] * sigma_k_min_2[il+2] + b[l] * sigma_k_min_1[il]
        #                         for il, l in  enumerate(range(k, 2 * N - k))])
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


@partial(jax.jit, static_argnums=(0,))
def lanczos(N: int, x, w):

    assert (N > 0), f"polynomial order N <= 0: {N} <= 0"
    assert (N <= len(x)), f"polynomial order N > quadrature deg: {N} > {len(x)}"

    def inner_loop(carry, k):
        alpha, beta, dpi, dgam, dsig, dt, xlam = carry
        drho = beta[k] + dpi
        dtmp = dgam * drho
        dtsig = dsig
        dgam = jax.lax.cond(drho<=0.0,
                            lambda a,b: 1.0,
                            lambda a,b: a/b,
                            beta[k], drho)
        dsig = jax.lax.cond(drho<=0.0,
                            lambda a,b: 0.0,
                            lambda a,b: a/b,
                            dpi, drho)
        dtk = dsig * (alpha[k] - xlam) - dgam * dt
        alpha = alpha.at[k].set(alpha[k] - (dtk - dt))
        dt = dtk
        dpi = jax.lax.cond(dsig<=0.0,
                           lambda a,b,c,d: a*b,
                           lambda a,b,c,d: c**2/d,
                           dtsig, beta[k], dt, dsig)
        dtsig = dsig
        beta = beta.at[k].set(dtmp)
        carry = (alpha, beta, dpi, dgam, dsig, dt, xlam)
        return carry, 0

    alpha = jnp.array(x)
    beta = jnp.zeros_like(alpha)
    beta = beta.at[0].set(w[0])
    for i in range(len(x)-1):
        dpi = w[i+1]
        dgam = 1.0
        dsig = 0.0
        dt = 0.0
        xlam = x[i+1]
        carry = (alpha, beta, dpi, dgam, dsig, dt, xlam)
        carry, _ = jax.lax.scan(inner_loop, carry, np.arange(i+2))
        alpha, beta, dpi, dgam, dsig, dt, xlam = carry

    return alpha[:N+1], beta[:N+1]


@jax.jit
def polval(x, alpha, beta):

    def f_pol(p, k):
        p = p.at[k].set((x - alpha[k-1]) * p[k-1] - beta[k-1] * p[k-2])
        return p, 0

    def f_norm(norm, k):
        n = norm[k-1] / jnp.sqrt(beta[k])
        norm = norm.at[k].set(n)
        return norm, n

    # p = jnp.zeros((len(alpha), len(x)), dtype=np.float64)
    p = jnp.zeros(len(alpha), dtype=np.float64)
    p = p.at[0].set(1)
    p = p.at[1].set(x - alpha[0])
    p, _ = jax.lax.scan(f_pol, p, np.arange(2, len(alpha)))

    norm = jnp.zeros(len(beta), dtype=np.float64)
    norm = norm.at[0].set(1.0 / jnp.sqrt(beta[0]))
    norm, _ = jax.lax.scan(f_norm, norm, np.arange(1, len(beta)))

    return p * norm


batch_polval = jax.jit(jax.vmap(polval, in_axes=(0, None, None)))


@jax.jit
def polder(x_batch, alpha, beta):
    def deriv(x):
        return jax.jacrev(polval, 0)(x, alpha, beta)
    return jax.vmap(deriv, in_axes=0)(x_batch)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    weight_func = lambda x: x*x #*jnp.exp(-x**2)
    N = 60

    # Method 1, starting from quadrature
    # left = -10
    # right = 10
    # nquad = 100
    # points, w = fejer_quadrature(nquad, left, right)
    # wf = weight_func(points)
    # weights = w * wf
    # alpha, beta = lanczos(N, points, weights)

    # Method 2, starting from moment integrals
    x, w, leg, a, b = legendre_monic(N)
    wf = weight_func(x)
    points, weights, alpha, beta = modified_chebyshev(N, w, leg, a, b, wf)

    points2, weights2, alpha2, beta2 = test_modified_chebyshev(weight_func, N)
    print(np.array(alpha)-alpha2)
    print(np.array(beta)-beta2)
    print(np.array(points)-points2)
    print(np.array(weights)-weights2)

    # overlap integrals
    p = batch_polval(points, alpha, beta)
    ovlp = jnp.einsum('gi,gj,g->ij', p, p, weights)
    print("overlap")
    ovlp_off = ovlp - np.diag(np.diag(ovlp))
    for i in range(len(ovlp)):
        print("i = ", i, "max offdiag = ", np.max(np.abs(ovlp_off[i])), "diag = ", ovlp[i, i])

    dp = polder(points, alpha, beta)
    print(dp.shape)
    # plot
    for pp in p.T[:6]:
        plt.plot(points, pp*np.sqrt(weights))
    for pp in dp.T[:6]:
        plt.plot(points, pp*np.sqrt(weights), '--')
    plt.show()

