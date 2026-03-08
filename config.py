"""
ECE 544 MPC+CBF Controller Configuration
==========================================

This module contains all tunable parameters for the MPC and MPC+CBF controllers.
Modify these parameters to tune controller behavior, solver settings, and constraints.
"""

# ============================================================================
# MPC HORIZON & TIME PARAMETERS
# ============================================================================

MPC_HORIZON = 10  # steps
DT = 0.1  # seconds (corresponds to 10 Hz control loop)
MPC_T_PRED = MPC_HORIZON * DT  # seconds

# ============================================================================
# VEHICLE DYNAMICS PARAMETERS
# ============================================================================

MAX_LINEAR_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = 2.0  # rad/s
LINEAR_ACCEL_LIMIT = 1.0  # m/s^2
ANGULAR_ACCEL_LIMIT = 2.0  # rad/s^2
WHEELBASE = 0.16  # meters (approximate for TurtleBot3)

# ============================================================================
# MPC COST WEIGHTS
# ============================================================================

W_POSITION = 10.0
W_HEADING = 1.0
W_CONTROL = 0.1
W_TERMINAL = 50.0

# ============================================================================
# SAFETY PARAMETERS (CBF - Control Barrier Functions)
# ============================================================================

USE_CBF = True
SAFETY_RADIUS = 0.2  # meters
CBF_GAMMA = 0.1
MIN_DISTANCE_PENALTY = 100.0
CBF_ACTIVATION_DISTANCE = 1.5  # meters

STATIC_OBSTACLES = [
    (0.6, 0.5, 0.15),     # Obstacle near start of path
    (1.0, 1.2, 0.2),      # Obstacle in middle of path
    (1.5, 1.4, 0.15),     # Obstacle near end of path
    (1.8, 1.9, 0.1),      # Small obstacle near goal
]

USE_LIDAR = True
LIDAR_MIN_RANGE = 0.12  # meters
LIDAR_MAX_RANGE = 3.5   # meters

# ============================================================================
# REFERENCE TRAJECTORY PARAMETERS
# ============================================================================

REFERENCE_LINEAR_VEL = 0.3  # m/s
LOOP_TRAJECTORY = False

# ============================================================================
# SOLVER SETTINGS (CasADi + IPOPT)
# ============================================================================

IPOPT_TOL = 1e-4
IPOPT_MAX_ITER = 100
SOLVER_PRINT_LEVEL = 0  # 0 = quiet, 5 = verbose
HESSIAN_REGULARIZATION = 1e-6
SOLVER_TIME_LIMIT = 0.05  # seconds

# ============================================================================
# INITIAL CONDITIONS & GOAL
# ============================================================================

START_POSITION = [0.0, 0.0, 0.0]
GOAL_POSITION = [2.0, 2.0, 0.0]
NUM_WAYPOINTS = 50
TRAJECTORY_TYPE = "line"

# ============================================================================
# LOGGING PARAMETERS
# ============================================================================

ENABLE_LOGGING = True
LOG_FILE_PATH = "/tmp/ece544_controller_log.csv"
LOG_FREQUENCY = 1
LOG_FIELDS = [
    "timestamp", "x", "y", "theta", "v", "omega",
    "min_distance", "collision_flag", "solver_time",
]

# ============================================================================
# ROS2 NODE PARAMETERS
# ============================================================================

NODE_NAME = "mpc_cbf_controller"
CONTROL_FREQ = 10  # Hz
TOPIC_ODM = "/odom"
TOPIC_SCAN = "/scan"
TOPIC_CMD_VEL = "/cmd_vel"
ROS2_QUEUE_SIZE = 10