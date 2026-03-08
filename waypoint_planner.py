"""
ECE 544 Waypoint and Trajectory Planner
=======================================

Generates reference trajectories for MPC to track.
"""

import numpy as np
from typing import List, Tuple, Optional
import config


def generate_line_path(start: np.ndarray, goal: np.ndarray,
                       num_points: int = 50) -> np.ndarray:
    positions = np.linspace(start[:2], goal[:2], num_points)
    headings = []
    for i, pos in enumerate(positions):
        if i < num_points - 1:
            delta = positions[i + 1] - positions[i]
            theta = np.arctan2(delta[1], delta[0])
        else:
            theta = goal[2]
        headings.append(theta)
    path = np.column_stack([positions, headings])
    return path


def generate_circle_path(center: np.ndarray, radius: float,
                         num_points: int = 50,
                         start_angle: float = 0.0) -> np.ndarray:
    angles = np.linspace(start_angle, start_angle + 2*np.pi, num_points)
    positions = []
    headings = []
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        positions.append([x, y])
        theta = angle + np.pi / 2
        headings.append(theta)
    path = np.column_stack([positions, headings])
    return path


def generate_figure8_path(center: np.ndarray, scale: float = 1.0,
                          num_points: int = 50) -> np.ndarray:
    t = np.linspace(0, 1, num_points)
    x = center[0] + scale * np.sin(2 * np.pi * t)
    y = center[1] + scale * np.sin(4 * np.pi * t)
    dx_dt = 2 * np.pi * scale * np.cos(2 * np.pi * t)
    dy_dt = 4 * np.pi * scale * np.cos(4 * np.pi * t)
    headings = np.arctan2(dy_dt, dx_dt)
    path = np.column_stack([x, y, headings])
    return path


def generate_trajectory(trajectory_type: str = "line",
                       start: np.ndarray = None,
                       goal: np.ndarray = None,
                       num_points: int = 50) -> np.ndarray:
    if start is None:
        start = np.array(config.START_POSITION)
    if goal is None:
        goal = np.array(config.GOAL_POSITION)

    if trajectory_type == "line":
        return generate_line_path(start, goal, num_points)
    elif trajectory_type == "circle":
        center = (start[:2] + goal[:2]) / 2
        radius = np.linalg.norm(goal[:2] - start[:2]) / 2
        return generate_circle_path(center, radius, num_points)
    elif trajectory_type == "figure8":
        center = (start[:2] + goal[:2]) / 2
        scale = np.linalg.norm(goal[:2] - start[:2]) / 4
        return generate_figure8_path(center, scale, num_points)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")


def get_reference_window(full_trajectory: np.ndarray,
                         current_index: int,
                         horizon: int,
                         loop: bool = False) -> np.ndarray:
    N = full_trajectory.shape[0]
    current_index = max(0, min(current_index, N - 1))
    reference = []
    for i in range(horizon + 1):
        idx = current_index + i
        if idx >= N:
            if loop:
                idx = idx % N
            else:
                idx = N - 1
        reference.append(full_trajectory[idx])
    return np.array(reference)


def compute_trajectory_progress(robot_position: np.ndarray,
                                full_trajectory: np.ndarray) -> Tuple[int, float]:
    positions = full_trajectory[:, :2]
    distances = np.linalg.norm(positions - robot_position[:2], axis=1)
    closest_idx = np.argmin(distances)
    min_distance = distances[closest_idx]
    return closest_idx, min_distance


def interpolate_waypoints(waypoint1: np.ndarray, waypoint2: np.ndarray,
                         num_intermediate: int = 5) -> np.ndarray:
    t = np.linspace(0, 1, num_intermediate + 2)[1:-1]
    x = waypoint1[0] + t * (waypoint2[0] - waypoint1[0])
    y = waypoint1[1] + t * (waypoint2[1] - waypoint1[1])
    theta = waypoint1[2] + t * (waypoint2[2] - waypoint1[2])
    return np.column_stack([x, y, theta])


if __name__ == "__main__":
    print("=" * 60)
    print("Waypoint Planner Test")
    print("=" * 60)
    start = np.array([0, 0, 0])
    goal = np.array([2, 2, 0])
    line_path = generate_line_path(start, goal, num_points=11)
    print(f"Line: {line_path.shape[0]} pts, start={line_path[0]}, end={line_path[-1]}")
    circle_path = generate_circle_path(np.array([1, 1]), radius=0.5, num_points=20)
    print(f"Circle: {circle_path.shape[0]} pts")
    ref_window = get_reference_window(line_path, current_index=3, horizon=5)
    print(f"Reference window: {ref_window.shape}")