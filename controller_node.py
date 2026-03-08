"""
ECE 544 ROS2 MPC+CBF Controller Node
===================================

Main ROS2 node that integrates all components.
Subscribes to /odom and /scan, publishes to /cmd_vel.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from typing import Optional
import time

from mpc_controller import MPCController
from mpc_cbf_controller import MPCCBFController
from lidar_processor import process_scan, get_obstacle_positions
from waypoint_planner import (
    generate_trajectory,
    get_reference_window,
    compute_trajectory_progress,
)
from logger import DataLogger
import config


class MPCCBFControllerNode(Node):

    def __init__(self):
        super().__init__(config.NODE_NAME)

        self.declare_parameter("use_cbf", config.USE_CBF)
        self.declare_parameter("mpc_horizon", config.MPC_HORIZON)
        self.declare_parameter("control_freq", config.CONTROL_FREQ)

        use_cbf = self.get_parameter("use_cbf").value
        mpc_horizon = self.get_parameter("mpc_horizon").value
        control_freq = self.get_parameter("control_freq").value

        if use_cbf:
            self.controller = MPCCBFController(
                horizon=mpc_horizon, dt=1.0 / control_freq, use_cbf=True)
            self.get_logger().info("[MPC+CBF] Controller initialized")
        else:
            self.controller = MPCController(
                horizon=mpc_horizon, dt=1.0 / control_freq)
            self.get_logger().info("[MPC] Controller initialized")

        self.logger = DataLogger(log_file_path=config.LOG_FILE_PATH)
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.current_twist = np.array([0.0, 0.0])
        self.last_scan = None

        self.reference_trajectory = generate_trajectory(
            trajectory_type=config.TRAJECTORY_TYPE,
            num_points=config.NUM_WAYPOINTS)
        self.trajectory_index = 0

        self.odom_sub = self.create_subscription(
            Odometry, config.TOPIC_ODM, self.odom_callback,
            qos_profile=rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=config.ROS2_QUEUE_SIZE))

        self.scan_sub = self.create_subscription(
            LaserScan, config.TOPIC_SCAN, self.scan_callback,
            qos_profile=rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=config.ROS2_QUEUE_SIZE))

        self.cmd_vel_pub = self.create_publisher(
            Twist, config.TOPIC_CMD_VEL, config.ROS2_QUEUE_SIZE)

        control_period = 1.0 / control_freq
        self.timer = self.create_timer(control_period, self.control_loop_callback)
        self.get_logger().info(f"Node initialized (freq={control_freq} Hz)")

    def odom_callback(self, msg: Odometry):
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            quat = msg.pose.pose.orientation
            theta = 2 * np.arctan2(quat.z, quat.w)
            self.current_state = np.array([x, y, theta])
            v_lin = msg.twist.twist.linear.x
            v_ang = msg.twist.twist.angular.z
            self.current_twist = np.array([v_lin, v_ang])
        except Exception as e:
            self.get_logger().warn(f"odom_callback error: {e}")

    def scan_callback(self, msg: LaserScan):
        try:
            self.last_scan = msg
        except Exception as e:
            self.get_logger().warn(f"scan_callback error: {e}")

    def control_loop_callback(self):
        start_time = time.time()
        try:
            obstacles = []
            min_distance = float('inf')

            if self.last_scan is not None and config.USE_LIDAR:
                try:
                    min_distance, min_angle, _ = process_scan(
                        np.array(self.last_scan.ranges),
                        self.last_scan.angle_min,
                        self.last_scan.angle_max,
                        self.last_scan.angle_increment)
                    obstacles = get_obstacle_positions(
                        np.array(self.last_scan.ranges),
                        self.last_scan.angle_min,
                        self.last_scan.angle_max,
                        self.last_scan.angle_increment,
                        self.current_state)
                except Exception as e:
                    self.get_logger().warn(f"LiDAR error: {e}")

            try:
                idx, dist = compute_trajectory_progress(
                    self.current_state, self.reference_trajectory)
                self.trajectory_index = idx
                reference_window = get_reference_window(
                    self.reference_trajectory, idx,
                    config.MPC_HORIZON, loop=config.LOOP_TRAJECTORY)
            except Exception as e:
                self.get_logger().warn(f"Trajectory error: {e}")
                reference_window = None

            collision_flag = 0
            try:
                control, success = self.controller.solve(
                    self.current_state,
                    obstacles=obstacles if obstacles else None,
                    reference_trajectory=reference_window)
                if not success:
                    control = np.array([0.0, 0.0])
                if min_distance < config.SAFETY_RADIUS:
                    collision_flag = 1
            except Exception as e:
                self.get_logger().error(f"MPC solve error: {e}")
                control = np.array([0.0, 0.0])

            cmd_msg = Twist()
            cmd_msg.linear.x = float(control[0])
            cmd_msg.angular.z = float(control[1])
            self.cmd_vel_pub.publish(cmd_msg)

            if config.ENABLE_LOGGING:
                self.logger.log_step(
                    timestamp=self.get_clock().now().nanoseconds / 1e9,
                    x=self.current_state[0], y=self.current_state[1],
                    theta=self.current_state[2],
                    v=control[0], omega=control[1],
                    min_distance=min_distance, collision_flag=collision_flag)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")

    def shutdown(self):
        try:
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            if config.ENABLE_LOGGING:
                self.logger.save()
        except Exception as e:
            self.get_logger().error(f"Shutdown error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MPCCBFControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()