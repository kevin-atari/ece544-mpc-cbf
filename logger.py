"""
ECE 544 Data Logging Utility
=============================

Logs controller data to CSV for analysis.
"""

import csv
import numpy as np
from typing import Optional, List
from pathlib import Path


class DataLogger:

    def __init__(self, log_file_path: str = "/tmp/ece544_log.csv",
                 fieldnames: Optional[List[str]] = None):
        self.log_file_path = log_file_path
        self.data = []
        if fieldnames is None:
            self.fieldnames = [
                "timestamp", "x", "y", "theta", "v", "omega",
                "min_distance", "collision_flag", "solver_time",
            ]
        else:
            self.fieldnames = fieldnames

    def log_step(self, timestamp: float, x: float, y: float, theta: float,
                 v: float, omega: float, min_distance: float,
                 collision_flag: int, solver_time: float = 0.0):
        row = {
            "timestamp": timestamp, "x": x, "y": y, "theta": theta,
            "v": v, "omega": omega, "min_distance": min_distance,
            "collision_flag": collision_flag, "solver_time": solver_time,
        }
        self.data.append(row)

    def save(self, overwrite: bool = True) -> bool:
        try:
            Path(self.log_file_path).parent.mkdir(parents=True, exist_ok=True)
            mode = 'w' if overwrite else 'a'
            with open(self.log_file_path, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if mode == 'w':
                    writer.writeheader()
                writer.writerows(self.data)
            print(f"[OK] Logged {len(self.data)} steps to {self.log_file_path}")
            return True
        except Exception as e:
            print(f"[FAIL] Error saving log: {e}")
            return False

    def clear(self):
        self.data.clear()

    def get_size(self) -> int:
        return len(self.data)

    def get_data(self) -> List[dict]:
        return self.data.copy()


def plot_trajectory(log_file_path: str, show: bool = True):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(log_file_path)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.plot(df['x'], df['y'], 'b-', label='Trajectory')
        ax.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', s=100, label='Start')
        ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', s=100, label='End')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory'); ax.grid(True); ax.legend(); ax.axis('equal')

        ax = axes[0, 1]
        ax.plot(df['timestamp'], df['min_distance'], 'b-')
        ax.axhline(y=0.2, color='r', linestyle='--', label='Safety')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Min Dist (m)')
        ax.set_title('Obstacle Distance'); ax.grid(True); ax.legend()

        ax = axes[1, 0]
        ax.plot(df['timestamp'], df['v'], 'b-', label='v')
        ax.plot(df['timestamp'], df['omega'], 'r-', label='omega')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Vel')
        ax.set_title('Controls'); ax.grid(True); ax.legend()

        ax = axes[1, 1]
        collision_times = df[df['collision_flag'] == 1]['timestamp']
        ax.scatter(collision_times, [1]*len(collision_times), color='red', s=100)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Collision')
        ax.set_title('Collision Events'); ax.set_ylim([-0.5, 1.5]); ax.grid(True)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(log_file_path.replace('.csv', '.png'))
    except ImportError:
        print("Requires pandas and matplotlib")