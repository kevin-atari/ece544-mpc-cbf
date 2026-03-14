"""
ECE 544 — Full MPC / MPC-CBF Simulation
========================================

Requires CasADi: pip install casadi

Usage:
    python run_simulation.py --mode mpc
    python run_simulation.py --mode mpc_cbf
    python run_simulation.py --mode compare
"""

import numpy as np
import argparse
import math
import time
from typing import List, Tuple

from dynamics import RobotDynamics
from waypoint_planner import generate_trajectory, get_reference_window, compute_trajectory_progress
from config import (DT, MPC_HORIZON, START_POSITION, GOAL_POSITION,
                    STATIC_OBSTACLES, MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY,
                    SAFETY_RADIUS)


def run_mpc_simulation(mode="mpc", num_steps=200, obstacles=None, reference=None):
    """
    Run full MPC or MPC-CBF simulation.
    mode: 'mpc' for pure MPC, 'mpc_cbf' for MPC with CBF safety constraints
    """
    try:
        from mpc_controller import MPCController
        from mpc_cbf_controller import MPCCBFController
    except ImportError as e:
        print(f"[ERROR] CasADi required. Install with: pip install casadi")
        print(f"  Details: {e}")
        return None, None, None, None

    dynamics = RobotDynamics(dt=DT)
    state = np.array(START_POSITION, dtype=float)

    if obstacles is None:
        obstacles = STATIC_OBSTACLES.copy()
    if reference is None:
        reference = generate_trajectory('line', num_points=100)

    # Create controller
    if mode == "mpc_cbf":
        controller = MPCCBFController(horizon=MPC_HORIZON, dt=DT, use_cbf=True)
        controller.add_static_obstacles(obstacles)
    else:
        controller = MPCController(horizon=MPC_HORIZON, dt=DT)

    states = [state.copy()]
    controls = []
    min_distances = []
    solve_times = []

    for step_idx in range(num_steps):
        # Find closest waypoint index, then get reference window
        closest_idx, _ = compute_trajectory_progress(state, reference)
        ref_window = get_reference_window(reference, closest_idx, MPC_HORIZON)

        # Solve MPC
        control, success = controller.solve(state, reference_trajectory=ref_window)

        if not success:
            control = np.array([0.0, 0.0])

        # Step dynamics
        state = dynamics.discrete_step(state, control)
        states.append(state.copy())
        controls.append(control.copy())
        solve_times.append(controller.last_solve_time)

        # Compute min distance to obstacles
        min_dist = float('inf')
        for ox, oy, r in obstacles:
            d = math.sqrt((state[0] - ox)**2 + (state[1] - oy)**2) - r
            min_dist = min(min_dist, d)
        min_distances.append(min_dist)

        # Progress update every 50 steps
        if (step_idx + 1) % 50 == 0:
            collisions_so_far = sum(1 for d in min_distances if d < 0)
            avg_solve = np.mean(solve_times) * 1000
            print(f"  Step {step_idx+1}/{num_steps}: collisions={collisions_so_far}, "
                  f"avg_solve={avg_solve:.1f}ms")

    return np.array(states), np.array(controls), np.array(min_distances), np.array(solve_times)


def plot_results(states, controls, min_distances, reference, obstacles,
                 title="MPC Simulation", save_path=None):
    """Plot single simulation results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'ECE 544: {title}', fontsize=14, fontweight='bold')

    # Trajectory
    ax = axes[0]
    ax.plot(reference[:, 0], reference[:, 1], 'g--', alpha=0.4, linewidth=1, label='Reference')
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax.scatter(states[0, 0], states[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(states[-1, 0], states[-1, 1], c='blue', s=100, marker='s', zorder=5, label='End')
    for ox, oy, r in obstacles:
        ax.add_patch(patches.Circle((ox, oy), r, fill=True, facecolor='red', alpha=0.3,
                                     edgecolor='red', linewidth=2))
        ax.add_patch(patches.Circle((ox, oy), r + SAFETY_RADIUS, fill=False,
                                     edgecolor='orange', linestyle='--', linewidth=1))
    collisions = np.sum(np.array(min_distances) < 0)
    ax.set_title(f'Trajectory (Collisions: {collisions})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Min distance
    ax = axes[1]
    t = np.arange(len(min_distances)) * DT
    ax.plot(t, min_distances, 'b-', linewidth=1.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Collision Boundary')
    ax.axhline(y=SAFETY_RADIUS, color='orange', linestyle='--', linewidth=1,
               label=f'Safety Margin ({SAFETY_RADIUS}m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Min Distance (m)')
    ax.set_title('Distance to Nearest Obstacle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Control inputs
    ax = axes[2]
    t_ctrl = np.arange(len(controls)) * DT
    ax.plot(t_ctrl, controls[:, 0], 'b-', label='v (m/s)')
    ax.plot(t_ctrl, controls[:, 1], 'r--', label='omega (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input')
    ax.set_title('Control Inputs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        fname = f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {fname}")


def plot_comparison(states_mpc, states_cbf, controls_mpc, controls_cbf,
                    dists_mpc, dists_cbf, reference, obstacles, save_path=None):
    """Plot MPC vs MPC-CBF comparison."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ECE 544: MPC vs MPC+CBF Comparison', fontsize=14, fontweight='bold')

    for ax_idx, (states, label, min_dists) in enumerate([
        (states_mpc, "MPC Only (No Safety)", dists_mpc),
        (states_cbf, "MPC + CBF (Safety Enabled)", dists_cbf)]):
        ax = axes[0, ax_idx]
        ax.plot(reference[:, 0], reference[:, 1], 'g--', alpha=0.4, linewidth=1, label='Reference')
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Robot Path')
        ax.scatter(states[0, 0], states[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
        ax.scatter(states[-1, 0], states[-1, 1], c='blue', s=100, marker='s', zorder=5, label='End')
        for ox, oy, r in obstacles:
            ax.add_patch(patches.Circle((ox, oy), r, fill=True, facecolor='red', alpha=0.3,
                                         edgecolor='red', linewidth=2))
            ax.add_patch(patches.Circle((ox, oy), r + SAFETY_RADIUS, fill=False,
                                         edgecolor='orange', linestyle='--', linewidth=1))
        collisions = np.sum(min_dists < 0)
        ax.set_title(f'{label}\nCollisions: {collisions}', fontsize=11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # Min distance comparison
    ax = axes[1, 0]
    t = np.arange(len(dists_mpc)) * DT
    ax.plot(t, dists_mpc, 'r-', alpha=0.7, label='MPC Only')
    ax.plot(t, dists_cbf, 'b-', alpha=0.7, label='MPC + CBF')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Collision Boundary')
    ax.axhline(y=SAFETY_RADIUS, color='orange', linestyle='--', linewidth=1,
               label=f'Safety Margin ({SAFETY_RADIUS}m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Min Distance (m)')
    ax.set_title('Distance to Nearest Obstacle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Control inputs comparison
    ax = axes[1, 1]
    t_ctrl = np.arange(len(controls_mpc)) * DT
    ax.plot(t_ctrl, controls_mpc[:, 0], 'r-', alpha=0.5, label='v (MPC)')
    ax.plot(t_ctrl, controls_cbf[:, 0], 'b-', alpha=0.5, label='v (MPC+CBF)')
    ax.plot(t_ctrl, controls_mpc[:, 1], 'r--', alpha=0.5, label='w (MPC)')
    ax.plot(t_ctrl, controls_cbf[:, 1], 'b--', alpha=0.5, label='w (MPC+CBF)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.set_title('Control Inputs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save = save_path or 'mpc_comparison.png'
    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save}")


def main():
    parser = argparse.ArgumentParser(description="ECE 544 MPC Simulation")
    parser.add_argument("--mode", choices=["mpc", "mpc_cbf", "compare"], default="compare")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("ECE 544 — MPC Simulation (CasADi + IPOPT)")
    print("=" * 60)

    reference = generate_trajectory('line', num_points=100)
    obstacles = STATIC_OBSTACLES.copy()
    print(f"Reference: line from ({reference[0,0]:.1f}, {reference[0,1]:.1f}) "
          f"to ({reference[-1,0]:.1f}, {reference[-1,1]:.1f})")
    print(f"Obstacles: {len(obstacles)}")
    print(f"Horizon: {MPC_HORIZON}, dt: {DT}s")
    print()

    if args.mode == "compare":
        print("[1/2] Running MPC only (no safety)...")
        states_m, controls_m, dists_m, times_m = run_mpc_simulation(
            "mpc", args.num_steps, obstacles, reference)
        if states_m is None:
            return
        print(f"  Collisions: {np.sum(dists_m < 0)}, Min dist: {dists_m.min():.3f}m, "
              f"Avg solve: {times_m.mean()*1000:.1f}ms")
        print()

        print("[2/2] Running MPC + CBF (safety enabled)...")
        states_c, controls_c, dists_c, times_c = run_mpc_simulation(
            "mpc_cbf", args.num_steps, obstacles, reference)
        if states_c is None:
            return
        print(f"  Collisions: {np.sum(dists_c < 0)}, Min dist: {dists_c.min():.3f}m, "
              f"Avg solve: {times_c.mean()*1000:.1f}ms")
        print()

        print("=" * 60)
        print(f"{'Metric':<30} {'MPC Only':<15} {'MPC+CBF':<15}")
        print("-" * 60)
        print(f"{'Collision steps':<30} {np.sum(dists_m < 0):<15} {np.sum(dists_c < 0):<15}")
        print(f"{'Min distance (m)':<30} {dists_m.min():<15.3f} {dists_c.min():<15.3f}")
        print(f"{'Mean distance (m)':<30} {dists_m.mean():<15.3f} {dists_c.mean():<15.3f}")
        print(f"{'Avg solve time (ms)':<30} {times_m.mean()*1000:<15.1f} {times_c.mean()*1000:<15.1f}")
        print("=" * 60)

        plot_comparison(states_m, states_c, controls_m, controls_c,
                       dists_m, dists_c, reference, obstacles, args.save_plot)

    else:
        mode_label = "MPC + CBF" if args.mode == "mpc_cbf" else "MPC Only"
        print(f"Running {mode_label}...")
        states, controls, dists, times = run_mpc_simulation(
            args.mode, args.num_steps, obstacles, reference)
        if states is None:
            return
        print(f"\nCollisions: {np.sum(dists < 0)}, Min dist: {dists.min():.3f}m, "
              f"Avg solve: {times.mean()*1000:.1f}ms")
        plot_results(states, controls, dists, reference, obstacles,
                    title=mode_label, save_path=args.save_plot)

    print("\nDone!")


if __name__ == "__main__":
    main()
