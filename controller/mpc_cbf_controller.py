def solve_mpc(state, ref):
    """
    Solve MPC optimization problem
    (placeholder)
    """
    return 0.2, 0.0


def apply_cbf(u_nominal, state, d_min):
    """
    Enforce safety constraint using CBF
    (placeholder)
    """
    v, omega = u_nominal
    return v, omega


def controller(state, ref, d_min, mode="mpc"):
    u_nominal = solve_mpc(state, ref)

    if mode == "mpc_cbf":
        return apply_cbf(u_nominal, state, d_min)

    return u_nominal
