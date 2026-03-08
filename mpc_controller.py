"""
ECE 544 Model Predictive Control (MPC) Implementation
=====================================================

Nonlinear MPC using CasADi Opti + IPOPT.
Pure tracking controller — no safety constraints.
For collision avoidance, see mpc_cbf_controller.py.
"""

import numpy as np
import casadi as ca
from typing import Tuple, Optional, List
from dynamics import casadi_discrete_dynamics
import config


class MPCController:
    """
    Nonlinear MPC for unicycle robot.

    Solves:
      min_{u} sum_k [ ||x_ref[k] - x[k]||^2_Q + ||u[k]||^2_R ]
      s.t.  x[k+1] = f(x[k], u[k])
            |v[k]| <= v_max, |omega[k]| <= omega_max
            x[0] = current_state
    """

    def __init__(self, horizon: int = 10, dt: float = 0.1):
        self.horizon = horizon
        self.dt = dt
        self.last_solve_time = 0.0
        self.solve_count = 0

        self.solver = ca.Opti()
        self._x_var = []
        self._u_var = []
        self._build_nlp()

    def _build_nlp(self):
        # State variables: x[k] = [x_k, y_k, theta_k]
        for k in range(self.horizon + 1):
            x_k = self.solver.variable(3, 1)
            self._x_var.append(x_k)

        # Control variables: u[k] = [v_k, omega_k]
        for k in range(self.horizon):
            u_k = self.solver.variable(2, 1)
            self._u_var.append(u_k)

        # Cost function
        cost = 0
        for k in range(self.horizon + 1):
            x_k = self._x_var[k]
            pos_error = ca.vertcat(x_k[0] - config.GOAL_POSITION[0],
                                    x_k[1] - config.GOAL_POSITION[1])
            cost += config.W_POSITION * ca.sumsqr(pos_error)
            theta_error = x_k[2] - config.GOAL_POSITION[2]
            cost += config.W_HEADING * theta_error**2
            if k == self.horizon:
                cost *= config.W_TERMINAL

        for k in range(self.horizon):
            u_k = self._u_var[k]
            cost += config.W_CONTROL * ca.sumsqr(u_k)

        self.solver.minimize(cost)

        # Initial state constraint
        self.x_init = self.solver.parameter(3, 1)
        self.solver.subject_to(self._x_var[0] == self.x_init)

        # Dynamics constraints
        for k in range(self.horizon):
            x_k = self._x_var[k]
            u_k = self._u_var[k]
            x_next = casadi_discrete_dynamics(x_k, u_k, self.dt)
            self.solver.subject_to(self._x_var[k + 1] == x_next)

        # Velocity constraints
        for k in range(self.horizon):
            u_k = self._u_var[k]
            self.solver.subject_to(u_k[0] >= -config.MAX_LINEAR_VELOCITY)
            self.solver.subject_to(u_k[0] <= config.MAX_LINEAR_VELOCITY)
            self.solver.subject_to(u_k[1] >= -config.MAX_ANGULAR_VELOCITY)
            self.solver.subject_to(u_k[1] <= config.MAX_ANGULAR_VELOCITY)

        # Solver settings
        opts = {
            "ipopt": {
                "max_iter": config.IPOPT_MAX_ITER,
                "tol": config.IPOPT_TOL,
                "print_level": config.SOLVER_PRINT_LEVEL,
            },
            "print_time": False,
        }
        self.solver.solver("ipopt", opts)

        # Initial guesses
        for k in range(self.horizon + 1):
            self.solver.set_initial(self._x_var[k], [0, 0, 0])
        for k in range(self.horizon):
            self.solver.set_initial(self._u_var[k], [config.REFERENCE_LINEAR_VEL, 0])

    def set_reference_trajectory(self, reference_states: np.ndarray):
        # TODO: Implement trajectory-based cost function
        pass

    def solve(self, current_state: np.ndarray,
              reference_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        import time

        if reference_trajectory is not None:
            self.set_reference_trajectory(reference_trajectory)

        self.solver.set_value(self.x_init, current_state.reshape(3, 1))

        start_time = time.time()
        try:
            sol = self.solver.solve()
            solve_time = time.time() - start_time
            success = True
        except RuntimeError as e:
            solve_time = time.time() - start_time
            success = False
            print(f"[WARNING] MPC solver failed: {e}")
            return np.array([0.0, 0.0]), False

        self.last_solve_time = solve_time
        self.solve_count += 1
        u_opt = np.array(sol.value(self._u_var[0])).flatten()
        return u_opt, success

    def get_predicted_trajectory(self) -> Optional[np.ndarray]:
        try:
            sol = self.solver.debug
            trajectory = []
            for k in range(self.horizon + 1):
                x_k = np.array(sol.value(self._x_var[k])).flatten()
                trajectory.append(x_k)
            return np.array(trajectory)
        except:
            return None

    def get_solver_info(self) -> dict:
        return {
            "solve_time": self.last_solve_time,
            "solve_count": self.solve_count,
            "horizon": self.horizon,
            "dt": self.dt,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("MPC Controller Test")
    print("=" * 60)
    mpc = MPCController(horizon=config.MPC_HORIZON, dt=config.DT)
    current_state = np.array([0.0, 0.0, 0.0])
    print(f"Initial state: {current_state}")
    print(f"Goal: {config.GOAL_POSITION}")
    control, success = mpc.solve(current_state)
    if success:
        print(f"[OK] v={control[0]:.4f}, omega={control[1]:.4f}, time={mpc.last_solve_time*1000:.2f}ms")
    else:
        print("[FAIL] MPC solver failed")