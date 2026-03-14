"""
ECE 544 — Integration Tests
============================

Tests all modules work together correctly.

Usage:
    python test_integration.py
"""

import numpy as np
import math
import sys


def test_dynamics():
    """Test unicycle dynamics model."""
    from dynamics import RobotDynamics

    dt = 0.1
    dynamics = RobotDynamics(dt=dt)

    # Test: driving straight from origin
    state = np.array([0.0, 0.0, 0.0])
    control = np.array([1.0, 0.0])  # v=1, omega=0
    next_state = dynamics.discrete_step(state, control)

    assert abs(next_state[0] - 0.1) < 1e-10, f"x should be 0.1, got {next_state[0]}"
    assert abs(next_state[1] - 0.0) < 1e-10, f"y should be 0.0, got {next_state[1]}"
    assert abs(next_state[2] - 0.0) < 1e-10, f"theta should be 0.0, got {next_state[2]}"

    # Test: turning in place
    state = np.array([0.0, 0.0, 0.0])
    control = np.array([0.0, 1.0])  # v=0, omega=1
    next_state = dynamics.discrete_step(state, control)
    assert abs(next_state[0]) < 1e-10, "x should be 0 when turning in place"
    assert abs(next_state[1]) < 1e-10, "y should be 0 when turning in place"
    assert abs(next_state[2] - 0.1) < 1e-10, f"theta should be 0.1, got {next_state[2]}"

    # Test: trajectory simulation
    controls = np.array([[0.5, 0.0], [0.5, 0.5], [0.5, 0.0]])
    traj = dynamics.simulate_trajectory(np.array([0.0, 0.0, 0.0]), controls)
    assert traj.shape == (4, 3), f"Trajectory shape should be (4,3), got {traj.shape}"

    print("[PASS] dynamics")


def test_waypoint_planner():
    """Test trajectory generation and reference window."""
    from waypoint_planner import generate_trajectory, get_reference_window

    # Test line trajectory
    traj = generate_trajectory('line', num_points=50)
    assert traj.shape[0] == 50, f"Should have 50 points, got {traj.shape[0]}"
    assert traj.shape[1] >= 2, "Should have at least x, y columns"

    # Test that start and end are correct
    assert abs(traj[0, 0]) < 0.1, "Line should start near x=0"
    assert abs(traj[0, 1]) < 0.1, "Line should start near y=0"

    # Test reference window (current_index is an int)
    window = get_reference_window(traj, current_index=5, horizon=10)
    assert window.shape[0] == 11, f"Window should have 11 points, got {window.shape[0]}"

    # Test circle trajectory
    traj_circle = generate_trajectory('circle', num_points=50)
    assert traj_circle.shape[0] == 50, "Circle should have 50 points"

    print("[PASS] waypoint_planner")


def test_config():
    """Test config parameters are valid."""
    from config import (DT, MPC_HORIZON, MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY,
                       SAFETY_RADIUS, CBF_GAMMA, STATIC_OBSTACLES, START_POSITION,
                       GOAL_POSITION)

    assert DT > 0, "DT must be positive"
    assert MPC_HORIZON > 0, "Horizon must be positive"
    assert MAX_LINEAR_VELOCITY > 0, "Max velocity must be positive"
    assert MAX_ANGULAR_VELOCITY > 0, "Max angular velocity must be positive"
    assert SAFETY_RADIUS > 0, "Safety radius must be positive"
    assert 0 < CBF_GAMMA < 1, "CBF gamma must be between 0 and 1"
    assert len(STATIC_OBSTACLES) > 0, "Should have at least one obstacle"
    assert len(START_POSITION) == 3, "Start position should be [x, y, theta]"
    assert len(GOAL_POSITION) == 3, "Goal position should be [x, y, theta]"

    print("[PASS] config")


def test_lidar_processor():
    """Test LiDAR scan processing."""
    from lidar_processor import process_scan, get_obstacle_positions

    # Create a fake scan with an obstacle at 1m directly ahead
    num_readings = 360
    ranges = [float('inf')] * num_readings
    # Put obstacle at 0 degrees (directly ahead), 1 meter away
    for i in range(355, 360):
        ranges[i] = 1.0
    for i in range(0, 5):
        ranges[i] = 1.0

    angle_increment = 2 * math.pi / num_readings
    result = process_scan(ranges, angle_min=-math.pi, angle_max=math.pi,
                          angle_increment=angle_increment)
    assert result is not None, "process_scan should return a result"
    min_dist, min_angle, valid = result
    assert min_dist == 1.0, f"Min distance should be 1.0, got {min_dist}"

    print("[PASS] lidar_processor")


def test_logger():
    """Test data logging."""
    from logger import DataLogger
    import os
    import tempfile

    log_path = os.path.join(tempfile.gettempdir(), "test_ece544.csv")
    logger = DataLogger(log_path)

    # Log some data (method is log_step, not log)
    logger.log_step(0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0, 0.005)
    logger.log_step(0.1, 0.01, 0.0, 0.0, 0.1, 0.0, 0.9, 0, 0.004)
    logger.save()

    assert os.path.exists(log_path), "Log file should exist"
    os.remove(log_path)

    print("[PASS] logger")


def test_demo_simulation():
    """Test that demo simulation runs without errors."""
    from demo_simulation import run_simulation

    states, controls, min_dists = run_simulation("track", num_steps=20)
    assert states.shape[0] == 21, f"Should have 21 states for 20 steps, got {states.shape[0]}"
    assert controls.shape[0] == 20, f"Should have 20 controls, got {controls.shape[0]}"
    assert len(min_dists) == 20, f"Should have 20 distance entries, got {len(min_dists)}"

    states_a, controls_a, min_dists_a = run_simulation("avoid", num_steps=20)
    assert states_a.shape[0] == 21

    print("[PASS] demo_simulation")


def test_casadi_available():
    """Test CasADi installation."""
    try:
        import casadi as ca
        x = ca.MX.sym('x', 3)
        u = ca.MX.sym('u', 2)
        print(f"[PASS] casadi (version {ca.__version__})")
        return True
    except ImportError:
        print("[SKIP] casadi not installed — MPC tests skipped")
        return False


def test_mpc_controller():
    """Test MPC controller (requires CasADi)."""
    from mpc_controller import MPCController
    import config

    controller = MPCController(horizon=5, dt=0.1)
    state = np.array([0.0, 0.0, 0.0])
    control, success = controller.solve(state)

    assert success, "MPC should solve successfully"
    assert len(control) == 2, "Control should be [v, omega]"
    assert abs(control[0]) <= config.MAX_LINEAR_VELOCITY + 0.01, "v should be within bounds"
    assert abs(control[1]) <= config.MAX_ANGULAR_VELOCITY + 0.01, "omega should be within bounds"

    print("[PASS] mpc_controller")


def test_mpc_cbf_controller():
    """Test MPC+CBF controller (requires CasADi)."""
    from mpc_cbf_controller import MPCCBFController
    import config

    controller = MPCCBFController(horizon=5, dt=0.1, use_cbf=True)
    state = np.array([0.0, 0.0, 0.0])
    control, success = controller.solve(state)

    assert success, "MPC+CBF should solve successfully"
    assert len(control) == 2, "Control should be [v, omega]"

    print("[PASS] mpc_cbf_controller")


def main():
    print("=" * 60)
    print("ECE 544 — Integration Tests")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    skipped = 0

    # Core tests (no CasADi needed)
    core_tests = [
        test_config,
        test_dynamics,
        test_waypoint_planner,
        test_lidar_processor,
        test_logger,
        test_demo_simulation,
    ]

    for test in core_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    # CasADi tests
    print()
    has_casadi = test_casadi_available()
    if has_casadi:
        passed += 1
        casadi_tests = [test_mpc_controller, test_mpc_cbf_controller]
        for test in casadi_tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"[FAIL] {test.__name__}: {e}")
                failed += 1
    else:
        skipped += 3

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
