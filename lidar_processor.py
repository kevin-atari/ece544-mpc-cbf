"""
ECE 544 LiDAR Scan Processor
============================

Processes raw LiDAR data to extract obstacle distances and positions.
"""

import numpy as np
from typing import Tuple, List, Optional


def process_scan(ranges: np.ndarray,
                 angle_min: float,
                 angle_max: float,
                 angle_increment: float,
                 range_min: float = 0.12,
                 range_max: float = 3.5) -> Tuple[float, float, np.ndarray]:
    ranges = np.array(ranges, dtype=float)
    num_measurements = len(ranges)
    angles = np.linspace(angle_min, angle_max, num_measurements)

    valid_mask = np.isfinite(ranges)
    valid_mask &= (ranges >= range_min)
    valid_mask &= (ranges <= range_max)

    valid_ranges = ranges[valid_mask]
    valid_angles = angles[valid_mask]

    if len(valid_ranges) > 0:
        min_idx = np.argmin(valid_ranges)
        min_distance = valid_ranges[min_idx]
        min_angle = valid_angles[min_idx]
    else:
        min_distance = np.inf
        min_angle = 0.0

    return min_distance, min_angle, valid_ranges


def get_obstacle_positions(ranges: np.ndarray,
                          angle_min: float,
                          angle_max: float,
                          angle_increment: float,
                          robot_pose: np.ndarray,
                          range_min: float = 0.12,
                          range_max: float = 3.5) -> List[Tuple[float, float, float]]:
    ranges = np.array(ranges, dtype=float)
    x_robot, y_robot, theta_robot = robot_pose

    num_measurements = len(ranges)
    angles = np.linspace(angle_min, angle_max, num_measurements)

    valid_mask = np.isfinite(ranges)
    valid_mask &= (ranges >= range_min)
    valid_mask &= (ranges <= range_max)

    valid_ranges = ranges[valid_mask]
    valid_angles = angles[valid_mask]

    obstacles = []
    obstacle_radius = 0.05

    for range_val, bearing_robot in zip(valid_ranges, valid_angles):
        bearing_world = theta_robot + bearing_robot
        x_world = x_robot + range_val * np.cos(bearing_world)
        y_world = y_robot + range_val * np.sin(bearing_world)
        obstacles.append((x_world, y_world, obstacle_radius))

    return obstacles


def cluster_measurements(measurements: np.ndarray,
                        cluster_distance: float = 0.1) -> List[np.ndarray]:
    if len(measurements) == 0:
        return []

    measurements = np.array(measurements)
    visited = np.zeros(len(measurements), dtype=bool)
    clusters = []

    for i in range(len(measurements)):
        if visited[i]:
            continue
        cluster = [measurements[i]]
        visited[i] = True
        queue = [i]

        while queue:
            current_idx = queue.pop(0)
            current_pos = measurements[current_idx]
            for j in range(len(measurements)):
                if visited[j]:
                    continue
                dist = np.linalg.norm(measurements[j] - current_pos)
                if dist <= cluster_distance:
                    visited[j] = True
                    cluster.append(measurements[j])
                    queue.append(j)
        clusters.append(np.array(cluster))

    return clusters


def get_cluster_centroid(cluster: np.ndarray) -> Tuple[float, float]:
    return tuple(np.mean(cluster, axis=0))


def estimate_obstacle_radius(cluster: np.ndarray) -> float:
    if len(cluster) == 1:
        return 0.05
    distances = []
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            dist = np.linalg.norm(cluster[i] - cluster[j])
            distances.append(dist)
    return max(distances) / 2


if __name__ == "__main__":
    print("=" * 60)
    print("LiDAR Processor Test")
    print("=" * 60)
    num_rays = 360
    ranges = np.ones(num_rays) * 3.5
    ranges[180] = 0.5
    min_dist, min_angle, valid = process_scan(ranges, -np.pi, np.pi, 2*np.pi/num_rays)
    print(f"Min dist: {min_dist:.3f}m, angle: {np.degrees(min_angle):.1f} deg")