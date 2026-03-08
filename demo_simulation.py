"""
ECE 544 — Demo Simulation (No CasADi Required)
=================================================

Run this FIRST to verify setup before tackling CasADi/MPC.

Usage:
    python demo_simulation.py
    python demo_simulation.py --mode compare
"""

import numpy as np
import argparse
import math
from typing import List, Tuple

from dynamics import RobotDynamics
from waypoint_planner import generate_trajectory, get_reference_window
from config import (DT, START_POSITION, GOAL_POSITION, STATIC_OBSTACLES,
                    MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY, SAFETY_RADIUS)


class ProportionalController:
    def __init__(self, kp_v=0.8, kp_omega=2.0, goal_threshold=0.15):
        self.kp_v = kp_v
        self.kp_omega = kp_omega
        self.goal_threshold = goal_threshold
        self.waypoint_index = 0

    def compute_control(self, state, reference):
        x, y, theta = state
        if self.waypoint_index < len(reference) - 1:
            wp = reference[self.waypoint_index]
            dist = math.sqrt((wp[0] - x)**2 + (wp[1] - y)**2)
            if dist < self.goal_threshold:
                self.waypoint_index = min(self.waypoint_index + 1, len(reference) - 1)

        target = reference[self.waypoint_index]
        dx = target[0] - x
        dy = target[1] - y
        dist = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)
        angle_error = angle_to_target - theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        v = self.kp_v * dist
        omega = self.kp_omega * angle_error
        if abs(angle_error) > 0.5:
            v *= 0.3
        v = np.clip(v, 0.0, MAX_LINEAR_VELOCITY)
        omega = np.clip(omega, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        return np.array([v, omega])


class ReactiveAvoidanceController(ProportionalController):
    def __init__(self, kp_v=0.8, kp_omega=2.0, avoidance_gain=1.5, avoidance_radius=0.8):
        super().__init__(kp_v, kp_omega)
        self.avoidance_gain = avoidance_gain
        self.avoidance_radius = avoidance_radius

    def compute_control_with_avoidance(self, state, reference, obstacles):
        base_control = self.compute_control(state, reference)
        v, omega = base_control
        x, y, theta = state
        repulsive_x = 0.0
        repulsive_y = 0.0

        for obs_x, obs_y, obs_r in obstacles:
            dx = x - obs_x
            dy = y - obs_y
            dist = math.sqrt(dx**2 + dy**2) - obs_r
            if dist < self.avoidance_radius and dist > 0.01:
                force = self.avoidance_gain / (dist**2)
                repulsive_x += force * dx / (dist + obs_r)
                repulsive_y += force * dy / (dist + obs_r)

        if abs(repulsive_x) > 0.01 or abs(repulsive_y) > 0.01:
            repulsive_angle = math.atan2(repulsive_y, repulsive_x)
            angle_diff = repulsive_angle - theta
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            omega += 2.0 * angle_diff
            repulsive_mag = math.sqrt(repulsive_x**2 + repulsive_y**2)
            v *= max(0.1, 1.0 - repulsive_mag * 0.5)

        v = np.clip(v, 0.0, MAX_LINEAR_VELOCITY)
        omega = np.clip(omega, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        return np.array([v, omega])


def run_simulation(controller_type="avoid", num_steps=200, obstacles=None, reference=None):
    dynamics = RobotDynamics(dt=DT)
    state = np.array(START_POSITION, dtype=float)
    if obstacles is None:
        obstacles = STATIC_OBSTACLES.copy()
    if reference is None:
        reference = generate_trajectory('line', num_points=100)

    if controller_type == "avoid":
        controller = ReactiveAvoidanceController()
    else:
        controller = ProportionalController()

    states = [state.copy()]
    controls = []
    min_distances = []

    for step in range(num_steps):
        if controller_type == "avoid":
            control = controller.compute_control_with_avoidance(state, reference, obstacles)
        else:
            control = controller.compute_control(state, reference)
        state = dynamics.discrete_step(state, control)
        states.append(state.copy())
        controls.append(control.copy())
        min_dist = float('inf')
        for ox, oy, r in obstacles:
            d = math.sqrt((state[0] - ox)**2 + (state[1] - oy)**2) - r
            min_dist = min(min_dist, d)
        min_distances.append(min_dist)

    return np.array(states), np.array(controls), np.array(min_distances)


def plot_comparison(states_track, states_avoid, controls_track, controls_avoid,
                    min_dist_track, min_dist_avoid, reference, obstacles, save_path=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ECE 544: Tracking Only vs Tracking + Avoidance', fontsize=14, fontweight='bold')

    for ax_idx, (states, label, min_dists) in enumerate([
        (states_track, "Tracking Only (MPC analog)", min_dist_track),
        (states_avoid, "Tracking + Avoidance (MPC+CBF analog)", min_dist_avoid)]):
        ax = axes[0, ax_idx]
        ax.plot(reference[:, 0], reference[:, 1], 'g--', alpha=0.4, linewidth=1, label='Reference')
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Robot Path')
        ax.scatter(states[0, 0], states[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
        ax.scatter(states[-1, 0], states[-1, 1], c='blue', s=100, marker='s', zorder=5, label='End')
        for ox, oy, r in obstacles:
            ax.add_patch(patches.Circle((ox, oy), r, fill=True, facecolor='red', alpha=0.3, edgecolor='red', linewidth=2))
            ax.add_patch(patches.Circle((ox, oy), r + SAFETY_RADIUS, fill=False, edgecolor='orange', linestyle='--', linewidth=1))
        collisions = np.sum(min_dists < 0)
        ax.set_title(f'{label}\nCollisions: {collisions}', fontsize=11)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.axis('equal')

    ax = axes[1, 0]
    t = np.arange(len(min_dist_track)) * DT
    ax.plot(t, min_dist_track, 'r-', alpha=0.7, label='Tracking Only')
    ax.plot(t, min_dist_avoid, 'b-', alpha=0.7, label='Tracking + Avoidance')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Collision Boundary')
    ax.axhline(y=SAFETY_RADIUS, color='orange', linestyle='--', linewidth=1, label=f'Safety Margin ({SAFETY_RADIUS}m)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Min Distance (m)')
    ax.set_title('Distance to Nearest Obstacle'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    t_ctrl = np.arange(len(controls_track)) * DT
    ax.plot(t_ctrl, controls_track[:, 0], 'r-', alpha=0.5, label='v (track)')
    ax.plot(t_ctrl, controls_avoid[:, 0], 'b-', alpha=0.5, label='v (avoid)')
    ax.plot(t_ctrl, controls_track[:, 1], 'r--', alpha=0.5, label='w (track)')
    ax.plot(t_ctrl, controls_avoid[:, 1], 'b--', alpha=0.5, label='w (avoid)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Velocity')
    ax.set_title('Control Inputs'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig('demo_comparison.png', dpi=150, bbox_inches='tight')
        print("Plot saved to: demo_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="ECE 544 Demo Simulation")
    parser.add_argument("--mode", choices=["track", "avoid", "compare"], default="compare")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("ECE 544 Demo Simulation")
    print("=" * 60)

    reference = generate_trajectory('line', num_points=100)
    obstacles = STATIC_OBSTACLES.copy()
    print(f"Reference: line from ({reference[0,0]:.1f}, {reference[0,1]:.1f}) to ({reference[-1,0]:.1f}, {reference[-1,1]:.1f})")
    print(f"Obstacles: {len(obstacles)}")

    if args.mode == "compare":
        print("[1/2] Running tracking only...")
        states_t, controls_t, dists_t = run_simulation("track", args.num_steps, obstacles, reference)
        print(f"  Collisions: {np.sum(dists_t < 0)}, Min dist: {dists_t.min():.3f}m")
        print("[2/2] Running tracking + avoidance...")
        states_a, controls_a, dists_a = run_simulation("avoid", args.num_steps, obstacles, reference)
        print(f"  Collisions: {np.sum(dists_a < 0)}, Min dist: {dists_a.min():.3f}m")
        print()
        print(f"{'Metric':<25} {'Track Only':<15} {'+ Avoidance':<15}")
        print("-" * 55)
        print(f"{'Collisions':<25} {np.sum(dists_t < 0):<15} {np.sum(dists_a < 0):<15}")
        print(f"{'Min distance (m)':<25} {dists_t.min():<15.3f} {dists_a.min():<15.3f}")
        save_path = args.save_plot or 'demo_comparison.png'
        plot_comparison(states_t, states_a, controls_t, controls_a,
                       dists_t, dists_a, reference, obstacles, save_path)
    else:
        states, controls, dists = run_simulation(args.mode, args.num_steps, obstacles, reference)
        print(f"Collisions: {np.sum(dists < 0)}, Min dist: {dists.min():.3f}m")

    print("\nDone!")


if __name__ == "__main__":
    main()