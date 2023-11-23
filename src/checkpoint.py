import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import glob
from src.utils import walltime



crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')
numpy_dir = os.path.join(data_dir, 'numpy')


def chunk(ode_fn, obj_func_partial, y_combo_ini, chunksize, num_total_steps):

    _, unflatten_fn = jax.flatten_util.ravel_pytree(y_combo_ini)

    files = glob.glob(os.path.join(numpy_dir, f'tmp/*'))
    for f in files:
        os.remove(f)

    def obj_fn(y_combo_ini):
        y_combo_fnl = fwd_pred(y_combo_ini)
        state, params = y_combo_fnl
        return obj_func_partial(state)
 
    def obj_fn_chunks(y_combo_ini):
        y_combo_fnl = fwd_pred_chunks(y_combo_ini)
        state, params = y_combo_fnl
        return obj_func_partial(state)

    @jax.jit
    def single_vjp_fn(y_combo, g): 
        primals, f_vjp = jax.vjp(ode_fn, *y_combo)
        vjp_result = f_vjp(g)
        return vjp_result

    def aux_fn(y_combo_ini, local_chunksize):
        y_combos = [y_combo_ini]
        y_combo_crt = y_combo_ini
        for i in range(local_chunksize):
            print(f"forward crt i = {i}")
            y_combo_crt = ode_fn(*y_combo_crt)
            y_combos.append(y_combo_crt)
        return y_combos

    def get_fwd_pred(local_chunksize):
        @jax.custom_vjp
        def fwd_pred(y_combo_ini):
            y_combos = aux_fn(y_combo_ini, local_chunksize)
            return y_combos[-1]

        def f_fwd(y_combo_ini):
            y_combos = aux_fn(y_combo_ini, local_chunksize)
            return y_combos[-1], y_combos

        @walltime
        def f_bwd(res, g):
            y_combos = res
            crt_vjp = g
            local_chunksize = len(y_combos) - 1
            for i in range(local_chunksize):
                crt_vjp = single_vjp_fn(y_combos[-(i + 2)], crt_vjp)
                print(f"reverse crt i = {i}")
            return crt_vjp,

        fwd_pred.defvjp(f_fwd, f_bwd)

        return fwd_pred

    def aux_fn_all_chunks(y_combo_ini):
        y_combo_crt = y_combo_ini
        file_names = []
        local_chunksizes = [chunksize]*(num_total_steps // chunksize) + \
                           [num_total_steps % chunksize] * (1 if num_total_steps % chunksize > 0 else 0)
        print(f"chunks = {local_chunksizes}")
        for i in range(num_total_steps):
            if i % chunksize == 0:
                print(f"save nodal solution {i} to hard drive...")
                file_name = os.path.join(numpy_dir, f'tmp/y_{i:05d}.npy')
                file_names.append(i)
                y_combo_crt_flat, _ = jax.flatten_util.ravel_pytree(y_combo_crt)
                np.save(file_name, y_combo_crt_flat)

            print(f"Running forward crt i = {i}")
            y_combo_crt = ode_fn(*y_combo_crt)

        return y_combo_crt, (file_names, local_chunksizes)

    @jax.custom_vjp
    def fwd_pred_chunks(y_combo_ini):
        y_combo_crt, _ = aux_fn_all_chunks(y_combo_ini)
        return y_combo_crt

    def f_fwd_chunks(y_combo_ini):
        y_combo_crt, file_names = aux_fn_all_chunks(y_combo_ini)
        return y_combo_crt, file_names

    def f_bwd_chunks(res, g):
        file_names, local_chunksizes = res
        crt_vjp = g
        for i in range(len(file_names)):
            local_chunksize = local_chunksizes[-(i + 1)]
            file_name = file_names[-(i + 1)]
            file_name = os.path.join(numpy_dir, f'tmp/y_{file_name:05d}.npy')
            print("load from hard drive solution and do chunk JVP...")
            y_combo_ini_flat = np.load(file_name)
            y_combo_ini = unflatten_fn(y_combo_ini_flat)
            fwd_pred = get_fwd_pred(local_chunksize)
            primals, f_vjp = jax.vjp(fwd_pred, y_combo_ini)
            crt_vjp, = f_vjp(crt_vjp)

        return crt_vjp,

    fwd_pred_chunks.defvjp(f_fwd_chunks, f_bwd_chunks)

    fwd_pred = get_fwd_pred(num_total_steps)

    # result = fwd_pred(y_combo_ini)
    # print(f"result sum = {np.sum(result[0])}")

    # Method 1: The entire trajectory is just one chunk
    grad_result = jax.grad(obj_fn)(y_combo_ini)
    print(f"grad_result = {grad_result[1]}")

    # Method 2: The entire trajectory is divided into chunks
    # grad_result_chunks = jax.grad(obj_fn_chunks)(y_combo_ini)
    # print(f"grad_result_chunk = {grad_result_chunks[1]}")


def example():
    def ode_fn(state, params):
        new_sate = state*params + params
        return new_sate, params

    def obj_func_partial(state):
        return np.sum(state)

    y_combo_ini = (np.array([1., 3.]), 2.)
    chunksize = 3
    num_total_steps = 5
    chunk(ode_fn, obj_func_partial, y_combo_ini, chunksize, num_total_steps)


if __name__ == "__main__":
    example()
