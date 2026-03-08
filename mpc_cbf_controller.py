"""
ECE 544 MPC with Control Barrier Functions (CBF) for Safety
===========================================================

Extends MPC with CBF constraints for collision avoidance.

CBF constraint: h(x[k+1]) >= h(x[k]) - gamma * h(x[k])
Where: h(x) = (x - x_obs)^2 + (y - y_obs)^2 - r_safe^2
"""

import numpy as np
import casadi as ca
from typing import Tuple, Optional, List
from mpc_controller import MPCController
from dynamics import casadi_discrete_dynamics
import config


class MPCCBFController(MPCController):
    """MPC + CBF controller for safe trajectory tracking."""

    def __init__(self, horizon: int = 10, dt: float = 0.1, use_cbf: bool = True):
        super().__init__(horizon=horizon, dt=dt)
        self.use_cbf = use_cbf
        self.obstacles = config.STATIC_OBSTACLES.copy()
        self.cbf_constraints_added = False
        self.last_infeasible = False

        if self.use_cbf:
            print("[MPC+CBF] Controller initialized with CBF enabled")
        else:
            print("[MPC] Controller initialized (CBF disabled)")

    def add_static_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        self.obstacles = obstacles.copy()
        print(f"[MPC+CBF] Updated with {len(obstacles)} static obstacles")

    def add_dynamic_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        self.obstacles = self.obstacles + obstacles

    def _build_cbf_constraints(self):
        if self.cbf_constraints_added:
            pass

        for obs_idx, (x_obs, y_obs, obs_radius) in enumerate(self.obstacles):
            if hasattr(self, '_x_current'):
                dist_to_obs = np.sqrt((self._x_current[0] - x_obs)**2 +
                                     (self._x_current[1] - y_obs)**2)
                if dist_to_obs > config.CBF_ACTIVATION_DISTANCE:
                    continue

            r_safe = obs_radius + config.SAFETY_RADIUS

            for k in range(self.horizon):
                x_k = self._x_var[k]
                x_next = self._x_var[k + 1]

                dx_k = x_k[0] - x_obs
                dy_k = x_k[1] - y_obs
                h_k = dx_k**2 + dy_k**2 - r_safe**2

                dx_next = x_next[0] - x_obs
                dy_next = x_next[1] - y_obs
                h_next = dx_next**2 + dy_next**2 - r_safe**2

                cbf_constraint = h_next + config.CBF_GAMMA * h_k

                try:
                    self.solver.subject_to(cbf_constraint >= 0)
                except:
                    pass

        self.cbf_constraints_added = True

    def solve(self, current_state: np.ndarray,
              obstacles: Optional[List[Tuple[float, float, float]]] = None,
              reference_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        import time

        self._x_current = current_state

        if obstacles is not None:
            self.add_dynamic_obstacles(obstacles)

        if self.use_cbf and len(self.obstacles) > 0:
            self._build_cbf_constraints()

        start_time = time.time()
        try:
            self.solver.set_value(self.x_init, current_state.reshape(3, 1))

            if reference_trajectory is not None:
                self.set_reference_trajectory(reference_trajectory)

            sol = self.solver.solve()
            solve_time = time.time() - start_time
            success = True
            self.last_infeasible = False

        except RuntimeError as e:
            solve_time = time.time() - start_time
            success = False
            self.last_infeasible = True
            print(f"[WARNING] MPC+CBF solver failed: {e}")
            print(f"  [FALLBACK] Returning zero control")
            control = np.array([0.0, 0.0])
            self.last_solve_time = solve_time
            self.solve_count += 1
            return control, False

        u_opt = np.array(sol.value(self._u_var[0])).flatten()
        self.last_solve_time = solve_time
        self.solve_count += 1
        return u_opt, success

    def set_cbf_enabled(self, enabled: bool):
        self.use_cbf = enabled
        mode = "MPC+CBF" if enabled else "MPC-only"
        print(f"[MPC+CBF] Mode switched to: {mode}")

    def get_solver_info(self) -> dict:
        info = super().get_solver_info()
        info.update({
            "cbf_enabled": self.use_cbf,
            "num_obstacles": len(self.obstacles),
            "last_infeasible": self.last_infeasible,
        })
        return info


if __name__ == "__main__":
    print("=" * 60)
    print("MPC+CBF Controller Test")
    print("=" * 60)
    controller = MPCCBFController(horizon=config.MPC_HORIZON, dt=config.DT, use_cbf=True)
    obstacles = [(1.0, 0.5, 0.1), (2.0, 1.5, 0.15)]
    controller.add_static_obstacles(obstacles)
    current_state = np.array([0.0, 0.0, 0.0])
    control, success = controller.solve(current_state)
    if success:
        print(f"[OK] v={control[0]:.4f}, omega={control[1]:.4f}")
    else:
        print("[FAIL] Solver failed")