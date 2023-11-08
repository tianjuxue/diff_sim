import jax
import jax.numpy as np
import numpy as onp


def chunk(ode_fn, obj_func_partial, y_combo_ini, chunksize):

    def obj_fn(y_combo_ini):
        y_combo_fnl = fwd_pred(y_combo_ini)
        state, params = y_combo_fnl
        return obj_func_partial(state)
 
    @jax.jit
    def single_jvp_fn(y_combo, g): 
        primals, f_vjp = jax.vjp(ode_fn, *y_combo)
        vjp_result = f_vjp(g)
        return vjp_result

    def aux_fn(y_combo_ini):
        y_combos = [y_combo_ini]
        y_combo_crt = y_combo_ini
        for i in range(chunksize):
            print(f"forward crt i = {i}")
            y_combo_crt = ode_fn(*y_combo_crt)
            y_combos.append(y_combo_crt)
        return y_combos


    @jax.custom_vjp
    def fwd_pred(y_combo_ini):
        y_combos = aux_fn(y_combo_ini)
        return y_combos[-1]

    def f_fwd(y_combo_ini):
        y_combos = aux_fn(y_combo_ini)
        return y_combos[-1], y_combos

    def f_bwd(res, g):
        y_combos = res
        crt_jvp = g
        for i in range(chunksize):
            crt_jvp = single_jvp_fn(y_combos[-(i + 2)], crt_jvp)
            print(f"reverse crt i = {i}")
        return crt_jvp,

    fwd_pred.defvjp(f_fwd, f_bwd)


    result = fwd_pred(y_combo_ini)
    print(f"result sum = {np.sum(result[0])}")

    grad_result = jax.grad(obj_fn)(y_combo_ini)
    print(f"grad_result = {grad_result[1]}")


def example():
    def ode_fn(state, params):
        new_sate = state*10 + params
        return new_sate, params

    def obj_func_partial(state):
        return np.sum(state)

    y_combo_ini = (np.array([1., 3.]), 2.)
    chunksize = 3
    chunk(ode_fn, obj_func_partial, y_combo_ini, chunksize)


if __name__ == "__main__":
    example()
