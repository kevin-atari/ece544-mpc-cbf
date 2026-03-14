"""
ECE 544 Robot Dynamics Model
============================

Unicycle robot kinematic model (e.g., TurtleBot3).

State: [x, y, theta]
Control Input: [v, omega]
"""

import numpy as np
from typing import Tuple, Union

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

from dynamics.differential_drive import step


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
        return step(state, control, self.dt)

    def simulate_trajectory(self, initial_state: np.ndarray,
                           control_sequence: np.ndarray) -> np.ndarray:
        N = control_sequence.shape[0]
        trajectory = np.zeros((N + 1, self.state_dim))
        trajectory[0] = initial_state.copy()
        for k in range(N):
            trajectory[k + 1] = self.discrete_step(trajectory[k], control_sequence[k])
        return trajectory


# CasADi versions for MPC optimization
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
