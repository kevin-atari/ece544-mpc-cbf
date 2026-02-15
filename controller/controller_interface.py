"""
ECE 544 — Controller Interface
Controls Lead: Kevin

Inputs:
  state = (x, y, theta)
  ref   = reference trajectory
  d_min = minimum obstacle distance

Outputs:
  u = (v, omega)
"""

def controller(state, ref, d_min, mode="mpc"):
    """
    mode:
      'mpc'     -> MPC only
      'mpc_cbf' -> MPC + CBF
    """

    # Placeholder control
    v = 0.2
    omega = 0.0

    return v, omega
