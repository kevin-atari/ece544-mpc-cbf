"""
ECE 544 Robot Dynamics Model
============================

Unicycle robot kinematic model (e.g., TurtleBot3).

State: [x, y, theta]
Control Input: [v, omega]

Discrete-time (Euler):
  x[k+1] = x[k] + v[k] * cos(theta[k]) * dt
  y[k+1] = y[k] + v[k] * sin(theta[k]) * dt
  theta[k+1] = theta[k] + omega[k] * dt
"""

import numpy as np
from typing import Tuple, Union

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


class RobotDynamics:
    """Unicycle robot kinematic model with discrete-time integration."""

    def __init__(self, dt: float):
        self.dt = dt
        self.state_dim = 3  # [x, y, theta]
        self.input_dim = 2  # [v, omega]

    def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega

        return np.array([x_dot, y_dot, theta_dot])

    def discrete_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        state_derivative = self.continuous_dynamics(state, control)
        next_state = state + state_derivative * self.dt
        return next_state

    def simulate_trajectory(self, initial_state: np.ndarray,
                           control_sequence: np.ndarray) -> np.ndarray:
        N = control_sequence.shape[0]
        trajectory = np.zeros((N + 1, self.state_dim))
        trajectory[0] = initial_state.copy()

        for k in range(N):
            trajectory[k + 1] = self.discrete_step(trajectory[k], control_sequence[k])

        return trajectory


# ============================================================================
# CasADi VERSIONS (for MPC optimization)
# ============================================================================

def casadi_continuous_dynamics(x_var, u_var):
    if not CASADI_AVAILABLE:
        raise ImportError("CasADi is required for MPC. Install with: pip install casadi")
    theta = x_var[2]
    v = u_var[0]
    omega = u_var[1]

    x_dot = v * ca.cos(theta)
    y_dot = v * ca.sin(theta)
    theta_dot = omega

    return ca.vertcat(x_dot, y_dot, theta_dot)


def casadi_discrete_dynamics(x_var, u_var, dt: float):
    x_dot = casadi_continuous_dynamics(x_var, u_var)
    x_next = x_var + x_dot * dt
    return x_next


def create_discrete_dynamics_function(dt: float):
    x = ca.MX.sym('x', 3)
    u = ca.MX.sym('u', 2)
    x_next = casadi_discrete_dynamics(x, u, dt)
    f = ca.Function('discrete_dynamics', [x, u], [x_next])
    return f


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dynamics():
    dt = 0.1
    np_dynamics = RobotDynamics(dt)

    if not CASADI_AVAILABLE:
        print("[!] CasADi not installed — skipping CasADi validation, testing numpy only")
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([0.5, 0.0])
        next_state = np_dynamics.discrete_step(state, control)
        print(f"[OK] NumPy dynamics works: {state} -> {next_state}")
        return True

    ca_dynamics = create_discrete_dynamics_function(dt)
    state = np.array([0.0, 0.0, 0.0])
    control = np.array([0.5, 0.0])
    next_state_np = np_dynamics.discrete_step(state, control)
    next_state_ca = np.array(ca_dynamics(state, control)).flatten()
    error = np.linalg.norm(next_state_np - next_state_ca)

    if error < 1e-10:
        print(f"[OK] Dynamics validation passed (error: {error:.2e})")
        return True
    else:
        print(f"[FAIL] Dynamics validation FAILED (error: {error:.2e})")
        return False


if __name__ == "__main__":
    validate_dynamics()
    dynamics = RobotDynamics(dt=0.1)
    x0 = np.array([0.0, 0.0, 0.0])
    controls = np.array([
        [0.5, 0.0], [0.5, 0.5], [0.5, 0.0], [0.5, -0.5],
    ])
    traj = dynamics.simulate_trajectory(x0, controls)
    print("\nExample trajectory:")
    print("Step | x (m)  | y (m)  | theta (rad)")
    for k, state in enumerate(traj):
        print(f"{k:4d} | {state[0]:6.3f} | {state[1]:6.3f} | {state[2]:6.3f}")