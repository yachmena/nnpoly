import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import List, Callable
from genpoly import fejer_quadrature, lanczos, batch_polval, polder
import optax
import itertools


class Dense(nn.Module):

    sizes: List[int]

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.sizes):
            kernel = self.param(f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size))
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = jax.nn.gelu(jnp.dot(x, kernel) + bias)
            size_ = size
        return jnp.exp(-x)


class Dense3D(nn.Module):

    sizes_x: List[int]
    sizes_y: List[int]
    sizes_z: List[int]

    def setup(self):
        self.dense_x = Dense(self.sizes_x)
        self.dense_y = Dense(self.sizes_y)
        self.dense_z = Dense(self.sizes_z)

    @nn.compact
    def __call__(self, xyz):
        x, y, z = xyz.T
        return jnp.array([self.dense_x(jnp.array([x]).T),
                          self.dense_y(jnp.array([y]).T),
                          self.dense_z(jnp.array([z]).T)])


def jac_model(model, params, xyz_batch):
    def jac(params):
        def jac(xyz):
            return jax.jacrev(model.apply, 1)(params, xyz)
        return jax.vmap(jac, in_axes=0)(xyz_batch)
    return jax.jit(jac)(params)


def potential(xyz):
    return -1.0 / jnp.linalg.norm(xyz, axis=-1)


def solve_lanczos(poten: Callable, nbas: int, quanta, left: float, right: float,
                  nquad: int = 100, nstates: int = 1):

    @jax.jit
    def hamiltonian(params):

        w_x, w_y, w_z = model.apply(params, jnp.array([x, y, z]).T)
        w_x *= wx[:, None]
        w_y *= wy[:, None]
        w_z *= wz[:, None]
        alpha_x, beta_x = lanczos(nbas, x, w_x[:, 0])
        alpha_y, beta_y = lanczos(nbas, y, w_y[:, 0])
        alpha_z, beta_z = lanczos(nbas, z, w_z[:, 0])

        pol_x = batch_polval(xyz[:, 0], alpha_x, beta_x)
        pol_y = batch_polval(xyz[:, 1], alpha_y, beta_y)
        pol_z = batch_polval(xyz[:, 2], alpha_z, beta_z)
        dpol_x = polder(xyz[:, 0], alpha_x, beta_x)
        dpol_y = polder(xyz[:, 1], alpha_y, beta_y)
        dpol_z = polder(xyz[:, 2], alpha_z, beta_z)

        w_x, w_y, w_z = model.apply(params, xyz)
        w_x *= wx_[:, None]
        w_y *= wy_[:, None]
        w_z *= wz_[:, None]
        sqw_x = jnp.sqrt(w_x[:, 0])
        sqw_y = jnp.sqrt(w_y[:, 0])
        sqw_z = jnp.sqrt(w_z[:, 0])

        dw = jac_model(model, params, xyz)[:, :, 0, :]
        dw_x = dw[:, 0, 0] * wx_
        dw_y = dw[:, 1, 1] * wy_
        dw_z = dw[:, 2, 2] * wz_

        psi_x = pol_x * sqw_x[:, None]
        psi_y = pol_y * sqw_y[:, None]
        psi_z = pol_z * sqw_z[:, None]

        dpsi_x = dpol_x * sqw_x[:, None] + 0.5 * pol_x / sqw_x[:, None] * dw_x[:, None]
        dpsi_y = dpol_y * sqw_y[:, None] + 0.5 * pol_y / sqw_y[:, None] * dw_y[:, None]
        dpsi_z = dpol_z * sqw_z[:, None] + 0.5 * pol_z / sqw_z[:, None] * dw_z[:, None]

        psi = jnp.array([psi_x[:, i]*psi_y[:, j]*psi_z[:, k] for (i, j, k) in quanta]).T
        dpsi_x = jnp.array([dpsi_x[:, i]*psi_y[:, j]*psi_z[:, k] for (i, j, k) in quanta]).T
        dpsi_y = jnp.array([psi_x[:, i]*dpsi_y[:, j]*psi_z[:, k] for (i, j, k) in quanta]).T
        dpsi_z = jnp.array([psi_x[:, i]*psi_y[:, j]*dpsi_z[:, k] for (i, j, k) in quanta]).T
        # nfunc = psi_x.shape[1] * psi_y.shape[1] * psi_z.shape[1]
        # npts = psi_x.shape[0]
        # psi = jnp.einsum('gi,gj,gk->gijk', psi_x, psi_y, psi_z).reshape(npts, nfunc)
        # dpsi_x = jnp.einsum('gi,gj,gk->gijk', dpsi_x, psi_y, psi_z).reshape(npts, nfunc)
        # dpsi_y = jnp.einsum('gi,gj,gk->gijk', psi_x, dpsi_y, psi_z).reshape(npts, nfunc)
        # dpsi_z = jnp.einsum('gi,gj,gk->gijk', psi_x, psi_y, dpsi_z).reshape(npts, nfunc)

        keo = 0.5 * jnp.einsum('gi,gj->ij', dpsi_x, dpsi_x, optimize='optimal') \
            + 0.5 * jnp.einsum('gi,gj->ij', dpsi_y, dpsi_y, optimize='optimal') \
            + 0.5 * jnp.einsum('gi,gj->ij', dpsi_z, dpsi_z, optimize='optimal')
        pot = jnp.einsum('gi,gj,g->ij', psi, psi, poten(xyz), optimize='optimal')
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

    x, wx = fejer_quadrature(nquad, left, right)
    y, wy = fejer_quadrature(nquad, left, right)
    z, wz = fejer_quadrature(nquad, left, right)
    xyz = np.array([elem for elem in itertools.product(x, y, z)])
    weights = np.array([elem for elem in itertools.product(wx, wy, wz)])
    wx_, wy_, wz_ = weights.T

    sizes = [128, 128, 128, 128, 1]
    model = Dense3D(sizes_x=sizes, sizes_y=sizes, sizes_z=sizes)
    params = model.init(jax.random.PRNGKey(0), xyz)

    h = hamiltonian(params)
    print(h.shape)
    e, _ = jnp.linalg.eigh(h)
    print(e[:nstates])

    optx = optax.adam(learning_rate=0.001)
    opt_state = optx.init(params)
    loss_grad_fn = jax.value_and_grad(loss_trace)
    
    for i in range(1000):
        loss_val, grad = loss_grad_fn(params)
        updates, opt_state = optx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
    
        h = hamiltonian(jax.lax.stop_gradient(params))
        e, _ = np.linalg.eigh(h)
        print(i, loss_val, e[:nstates])


if __name__ == "__main__":

    poten = potential
    nbas = 3
    left = -20.0
    right = 20.0

    quanta = np.array([elem for elem in itertools.product(*[np.arange(nbas)]*3) if sum(elem) <= nbas])
    solve_lanczos(poten, nbas, quanta, left, right, nquad=60, nstates=5)

