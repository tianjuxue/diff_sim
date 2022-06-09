def compute_gradient_ad(ys, ts, obj_func, state_rhs_func, diff_args):
    rev_ts = ts[::-1]
    rev_ys = ys[::-1]
    y_bar, diff_args_bar = jax.grad(obj_func)(rev_ys[0]), jax.tree_map(np.zeros_like, diff_args)
    y_bars = [y_bar]
    diff_args_bars = [diff_args_bar]

    @jax.jit
    def adjoint_fn(y_prev, t_prev, t_crt, y_bar):
        y_dot, vjpfun = jax.vjp(lambda y_prev, diff_args: rk4((y_prev, t_prev), t_crt, state_rhs_func, diff_args)[1], y_prev, diff_args)
        y_bar, diff_args_bar = vjpfun(y_bar)  
        return y_bar, diff_args_bar   

    for i in range(len(ts) - 1):
        y_prev = rev_ys[i + 1]
        t_prev = rev_ts[i + 1]
        y_crt = rev_ys[i]
        t_crt = rev_ts[i]

        y_bar, diff_args_bar = adjoint_fn(y_prev, t_prev, t_crt, y_bar)

        y_bars.append(y_bar)
        diff_args_bars.append(diff_args_bar)

        if i % 100 == 0:
            print(f"Reverse step {i}") 
            if not np.all(np.isfinite(y_bar)):
                print(f"Found np.inf or np.nan in y - stop the program")             
                exit()

    y_bars = np.stack(y_bars)
    grads = jax.tree_multimap(lambda *xs: np.sum(np.stack(xs), axis=0), *diff_args_bars)