import jax
import jax.numpy as np
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree
import numpy as onp
from functools import partial
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
import os
import meshio
import time
from src.utils import read_path, obj_to_vtu, walltime
from src.arguments import args
from src.allen_cahn import polycrystal_fd, phase_field, odeint, rk4, explicit_euler
from src.example import initialization
from src.checkpoint import chunk


def set_params():
    '''
    If a certain parameter is not set, a default value will be used (see src/arguments.py for details).
    '''
    args.case = 'optimize'
    args.num_grains = 10000
    args.domain_length = 0.5
    args.domain_width = 0.2
    args.domain_height = 0.1
    args.r_beam = 0.03
    args.power = 100
    args.write_sol_interval = 1000



def odeint_aug(ys, stepper, f, y0, ts, ode_params):
    state = (y0, ts[0])

    for (i, t_crt) in enumerate(ts[1:]):
        state, y = stepper(state, t_crt, f, ode_params)

        # state[0].at[:args.total_dofs_eta].set(ys[-i - 2].reshape(-1))

        if (i + 1) % 20 == 0:
            print(f"step {i + 1} of {len(ts[1:])}, unix timestamp = {time.time()}")
            if not np.all(np.isfinite(y)):          
                raise ValueError(f"Found np.inf or np.nan in y - stop the program")
    return y


def compute_gradient_fd(aux_args, y0, ts, obj_func, state_rhs_func, ode_params):
    print(f"running finite difference")
    
    h = 1e-3
    ode_params_flat, unravel = ravel_pytree(ode_params)

    grads = []
    for i in range(len(ode_params_flat)):
        ode_params_flat_plus = ode_params_flat.at[i].add(h)
        yf_plus, _ = odeint(*aux_args, explicit_euler, state_rhs_func, y0, ts, unravel(ode_params_flat_plus))
        tau_plus = obj_func(yf_plus)
        ode_params_flat_minus = ode_params_flat.at[i].add(-h)
        yf_minus, _ = odeint(*aux_args, explicit_euler, state_rhs_func, y0, ts, unravel(ode_params_flat_minus))
        tau_minus = obj_func(yf_minus)
        grad = (tau_plus - tau_minus) / (2 * h)
        grads.append(grad)
    grads = unravel(np.array(grads))

    return grads

 

def compute_gradient_ad_late_discretization(yf, ys, ts, obj_func, state_rhs_func, ode_params):
    print(f"running autodiff, late discretization")

    def ravel_f(func, unravel):
        def modified_f(raveled_aug, t, args):
            aug = unravel(raveled_aug)
            return_value = func(aug, t, args)
            rhs, _ = ravel_pytree(return_value)
            return rhs
        return modified_f

    def odeint_ravelled(stepper, f, y0, ts, ode_params):
        y0_flat, unravel = ravel_pytree(y0)
        f_flat = ravel_f(f, unravel)
        out = odeint_aug(ys, stepper, f_flat, y0_flat, ts, ode_params)
        return unravel(out)

    def get_aug_rhs_func(state_rhs_func):
        def aug_rhs_func(aug, neg_t, aug_args):
            y, init_cond_adjoint, ode_params_adjoint = aug
            y_dot, vjpfun = jax.vjp(lambda y, ode_params: state_rhs_func(y, -neg_t, ode_params), y, ode_params)
            init_cond_adjoint_dot, ode_params_adjoint_dot = vjpfun(init_cond_adjoint)
            return (-y_dot, init_cond_adjoint_dot, ode_params_adjoint_dot)

        return aug_rhs_func
    
    aug0 = (yf, jax.grad(obj_func)(yf), jax.tree_util.tree_map(np.zeros_like, ode_params))
    aug_rhs_func = get_aug_rhs_func(state_rhs_func)
    y0_bwd, init_cond_adjoint, ode_params_adjoint = odeint_ravelled(explicit_euler, aug_rhs_func, aug0, -ts[::-1], [])
    grads = ode_params_adjoint
    
    return grads



def compute_gradient_ad_jax(aux_args, y0, ts, obj_func, state_rhs_func, ode_params):
    def func(ode_params):
        yf, _ = odeint(*aux_args, explicit_euler, state_rhs_func, y0, ts, ode_params)
        tau = obj_func(yf)
        return tau

    grad_func = jax.grad(func)
    grads_jax = grad_func(ode_params)
    return grads_jax


def run():
    set_params()
    # TODO: bad symbol ys
    ts, xs, ys, ps = read_path(f'data/txt/fd_example_1.txt')
    dt = ts[1] - ts[0]
    polycrystal, mesh = polycrystal_fd(args.case)
    y0 = initialization(polycrystal)
    state_rhs, get_T = phase_field(polycrystal)

    # Remark: JAX is type sensitive, if you specify [20., 4], it fails.
    ode_params_0 = [24.9, 5.1]


    # For debugging purpose
    # Solving backward in time, explode easily, not accurate
    # args.case = 'fd_example'
    # yf, ys = odeint(polycrystal, mesh, get_T, explicit_euler, state_rhs, y0, ts, ode_params_0)

    # state_rhs_tau = lambda state, tau, ode_params: -state_rhs(state, -tau, ode_params)
    # get_T_tau = lambda tau, ode_params: get_T(-tau, ode_params)

    # args.case = 'optimize'
    # args.write_sol_interval = 10
    # ts, xs, ys, ps = read_path(f'data/txt/fd_example_1.txt')
    # taus = -ts[::-1]
    # odeint(polycrystal, mesh, get_T_tau, explicit_euler, state_rhs_tau, yf, taus, ode_params_0)
    # exit()


    ode_params_gt = [25., 5.2]    
    target_yf, _ = odeint(polycrystal, mesh, get_T, explicit_euler, state_rhs, y0, ts, ode_params_gt)

    def obj_func(yf, target_yf):
        # Some arbitrary objective function
        return np.sum((yf - target_yf)**2)
 
    obj_func_partial = lambda yf: obj_func(yf, target_yf)

    # Early discretization seems to be the best option
    # If we further use checkpoint method, we can compute derivative for a long time chain
    def get_ode_fn():
        @jax.jit
        def ode_fn(y_prev, params_prev):
            ode_params, dt, t_prev = params_prev
            y_crt = y_prev + dt * state_rhs(y_prev, t_prev, ode_params)
            t_crt = t_prev + dt
            params_crt = (ode_params, dt, t_crt)
            return (y_crt, params_crt)
        return ode_fn

    get_ode_fn = get_ode_fn()
    y_combo_ini = (y0, (ode_params_0, dt, 0.))
    print(f"start of checkpoint")
    # chunksize = 5
    chunksize = 50
    num_total_steps = len(ts[1:])
    chunk(get_ode_fn, obj_func_partial, y_combo_ini, chunksize, num_total_steps)

    # Finite difference as ground truth
    grads_fd = compute_gradient_fd([polycrystal, mesh, get_T], y0, ts, obj_func_partial, state_rhs, ode_params_0)
    print(f"grads_fd = {grads_fd}\n")


    # Diff through loops, very slow, memory easily explodes 
    # print(f"start of grads_jax")
    # grads_jax = compute_gradient_ad_jax([polycrystal, mesh, get_T], y0, ts, obj_func_partial, state_rhs, ode_params_0)
    # print(f"grads_jax = {grads_jax}\n")


    # Late discretization, formulation is elegant, but not working probably because inverse solve is not accurate 
    # args.total_dofs_eta = len(yf.reshape(-1))
    # grads_ad = compute_gradient_ad_late_discretization(yf, ys, ts, obj_func_partial, state_rhs, ode_params_0)
    # print(f"grads = {grads_ad}\n")


if __name__ == "__main__":
    # neper_domain()
    # write_vtu_files()
    run()
