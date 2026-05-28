"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  system.launch.py  –  Full System Launch                                    ║
║  GPS / INS / Odometer Information Fusion Robot                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  LAUNCH ORDER  (phased to respect startup dependencies)                     ║
║  ────────────────────────────────────────────────────────                   ║
║  t = 0 s   PHASE 1 │ Simulation                                             ║
║                     │  • Gazebo  (indoor_outdoor.world)                     ║
║                     │  • robot_state_publisher (URDF → TF tree)             ║
║                     │  • spawn_entity  (TurtleBot3 Burger)                  ║
║                                                                              ║
║  t = ekf_delay      PHASE 2 │ Localization Stack                            ║
║      (default 8 s)  │  • ekf_local           /odometry/local  (KF1-local)  ║
║                     │  • ekf_filter_gps_imu  /odometry/global (KF1-global) ║
║                     │  • navsat_transform    /odometry/gps                  ║
║                     │  • ekf_local2          /odometry/local2 (KF2-local)  ║
║                     │  • ekf_filter_gps_enc  /odometry/global2(KF2-global) ║
║                     │  • navsat_transform2   /odometry/gps                  ║
║                                                                              ║
║  t = fusion_delay   PHASE 3 │ Fusion & Intelligence                         ║
║      (default 12 s) │  • complementary_filter  /odometry/fused              ║
║                     │  • trajectory_nn_node    /ann/trajectory              ║
║                     │  • trajectory_controller /cmd_vel                     ║
║                                                                              ║
║  t = bt_delay       PHASE 4 │ BT Orchestrator                               ║
║      (default 15 s) │  • bt_brain  /odometry/bt_fused                      ║
║                     │            /bt/fuzzy_weights                          ║
║                     │            /bt/status                                 ║
║                     │            /bt/indoor_detection                       ║
║                                                                              ║
║  OPTIONAL  │  RViz2 (launched at fusion_delay alongside PHASE 3)            ║
║                                                                              ║
║  ARGUMENTS                                                                  ║
║  ─────────                                                                  ║
║   use_rviz        true/false  (default: true)                               ║
║   use_sim_time    true/false  (default: true)                               ║
║   ekf_delay       seconds     (default: 8.0)                                ║
║   fusion_delay    seconds     (default: 12.0)                               ║
║   bt_delay        seconds     (default: 9.0)                                ║
║   log_level       DEBUG/INFO/WARN  (default: INFO)                          ║
║                                                                              ║
║  USAGE                                                                      ║
║  ─────                                                                      ║
║   ros2 launch bt_orchestrator_pkg system.launch.py                          ║
║   ros2 launch bt_orchestrator_pkg system.launch.py use_rviz:=false          ║
║   ros2 launch bt_orchestrator_pkg system.launch.py ekf_delay:=15.0          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


# ─────────────────────────────────────────────────────────────────────────────
#  Package share directories  (resolved at launch time)
# ─────────────────────────────────────────────────────────────────────────────
def get_pkg(name: str) -> str:
    return get_package_share_directory(name)


# ─────────────────────────────────────────────────────────────────────────────
def generate_launch_description() -> LaunchDescription:

    # ── Resolve package paths ─────────────────────────────────────────────────
    pkg_robot_desc   = get_pkg('robot_description_pkg')
    pkg_gps_ins      = get_pkg('gps_ins_pkg')
    pkg_gps_odom     = get_pkg('gps_odometry_pkg')
    pkg_control      = get_pkg('control_pkg')
    pkg_bt           = get_pkg('bt_orchestrator_pkg')

    cfg_ekf1         = os.path.join(pkg_gps_ins,  'config', 'ekf_1.yaml')
    cfg_ekf2         = os.path.join(pkg_gps_odom, 'config', 'ekf_2.yaml')
    cfg_waypoints    = os.path.join(pkg_control,  'config', 'waypoints.yaml')

    # ── Declare launch arguments ──────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz2 for live visualisation'),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use /clock from Gazebo instead of wall time'),

        DeclareLaunchArgument(
            'ekf_delay',
            default_value='8.0',
            description='Seconds to wait before starting EKF nodes '
                        '(allow Gazebo + sensors to come up)'),

        DeclareLaunchArgument(
            'fusion_delay',
            default_value='12.0',
            description='Seconds to wait before starting fusion / intelligence '
                        'nodes (allow EKFs to initialise)'),

        DeclareLaunchArgument(
            'bt_delay',
            default_value='9.0',
            description='Seconds to wait before starting the BT brain. '
                        'Must be > ekf_delay (default 8 s) so ekf_local has '
                        'time to publish odom→base_footprint before bt_brain '
                        'bootstraps the map→odom TF for navsat_transform.'),

        DeclareLaunchArgument(
            'log_level',
            default_value='INFO',
            description='ROS2 log level: DEBUG | INFO | WARN | ERROR'),
    ]

    # ── Shorthand references ──────────────────────────────────────────────────
    use_rviz     = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    ekf_delay    = LaunchConfiguration('ekf_delay')
    fusion_delay = LaunchConfiguration('fusion_delay')
    bt_delay     = LaunchConfiguration('bt_delay')
    log_level    = LaunchConfiguration('log_level')

    sim_time_param = {'use_sim_time': use_sim_time}

    # ═════════════════════════════════════════════════════════════════════════
    #  PHASE 1  –  SIMULATION  (t = 0 s)
    # ═════════════════════════════════════════════════════════════════════════
    phase1_log = LogInfo(msg='\n'
        '╔═══════════════════════════════════════╗\n'
        '║  PHASE 1 │ Starting Gazebo + Robot    ║\n'
        '╚═══════════════════════════════════════╝')

    # Reuse the existing spawn_robot.launch.py which starts:
    #   • gazebo_ros   (indoor_outdoor.world)
    #   • robot_state_publisher
    #   • spawn_entity (TurtleBot3 at x=5.0, y=6.5, z=0.05)
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_robot_desc, 'launch', 'spawn_robot.launch.py')
        )
        # Note: spawn_robot.launch.py does not declare use_sim_time yet;
        # robot_state_publisher inside it already receives use_sim_time=True
        # via its own parameters block.
    )

    # ═════════════════════════════════════════════════════════════════════════
    #  PHASE 2  –  LOCALIZATION STACK  (t = ekf_delay)
    #  Both EKF stacks are launched together; each yaml includes its own
    #  navsat_transform configuration so all 6 nodes start here.
    # ═════════════════════════════════════════════════════════════════════════
    phase2_log = LogInfo(msg='\n'
        '╔═══════════════════════════════════════════════════╗\n'
        '║  PHASE 2 │ Starting EKF1 + EKF2 stacks           ║\n'
        '╚═══════════════════════════════════════════════════╝')

    # ── KF1: GPS + IMU  EKF ──────────────────────────────────────────────────

    # Local EKF (continuous state, Odom+IMU)  → /odometry/local
    ekf1_local = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_local',
        output='screen',
        parameters=[cfg_ekf1, sim_time_param],
        remappings=[('odometry/filtered', '/odometry/local')],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # Global EKF (discrete GPS updates, GPS+IMU)  → /odometry/global
    ekf1_global = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_gps_imu',
        output='screen',
        parameters=[cfg_ekf1, sim_time_param],
        remappings=[('odometry/filtered', '/odometry/global')],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # NavSat Transform (GPS lat/lon → ENU odometry)  → /odometry/gps
    navsat1 = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        output='screen',
        parameters=[cfg_ekf1, sim_time_param],
        remappings=[
            ('imu',               '/imu/data'),
            ('gps/fix',           '/gps/fix'),
            ('gps/filtered',      '/gps/filtered'),
            ('odometry/gps',      '/odometry/gps'),
            ('odometry/filtered', '/odometry/global'),
        ],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── KF2: GPS + Odometry  EKF ─────────────────────────────────────────────

    # Local EKF (continuous, Odom+IMU)  → /odometry/local2
    ekf2_local = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_local2',
        output='screen',
        parameters=[cfg_ekf2, sim_time_param],
        remappings=[('odometry/filtered', '/odometry/local2')],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # Global EKF (GPS + Odom)  → /odometry/global2
    ekf2_global = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_gps_enc',
        output='screen',
        parameters=[cfg_ekf2, sim_time_param],
        remappings=[('odometry/filtered', '/odometry/global2')],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # FIX 3: navsat_transform2 has been removed.
    # Both navsat_transform instances were publishing to the same /odometry/gps
    # topic with different odometry sources, causing a non-deterministic race
    # condition: whichever node wrote last "won", corrupting the GPS→ENU
    # conversion that both EKF1 and EKF2 depend on.
    #
    # Solution: a single navsat_transform (navsat1, fed by EKF1's global
    # odometry for heading) publishes /odometry/gps once.  Both
    # ekf_filter_node_gps_imu and ekf_filter_node_gps_enc read the same
    # /odometry/gps topic — this is correct because the GPS hardware is one
    # physical sensor and its ENU projection does not change depending on
    # which EKF reads it.

    phase2 = TimerAction(
        period=ekf_delay,
        actions=[
            phase2_log,
            ekf1_local,
            ekf1_global,
            navsat1,    # Single navsat_transform feeds both EKF1 and EKF2
            ekf2_local,
            ekf2_global,
        ]
    )

    # ═════════════════════════════════════════════════════════════════════════
    #  PHASE 3  –  FUSION & INTELLIGENCE  (t = fusion_delay)
    # ═════════════════════════════════════════════════════════════════════════
    phase3_log = LogInfo(msg='\n'
        '╔══════════════════════════════════════════════════════════╗\n'
        '║  PHASE 3 │ Starting Fusion + ANN + Trajectory Control   ║\n'
        '╚══════════════════════════════════════════════════════════╝')

    # ── Complementary Filter ──────────────────────────────────────────────────
    # Subscribes : /odometry/global (KF1)  /odometry/global2 (KF2)
    # Publishes  : /odometry/fused  (fixed α=0.5 fusion, GPS mode reference)
    complementary_filter = Node(
        package='gps_ins_pkg',
        executable='complementary_filter',
        name='complementary_filter_node',
        output='screen',
        parameters=[sim_time_param],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── ANN Online Training Node ──────────────────────────────────────────────
    # Subscribes : /imu/data  /odometry/gps  /odom
    #              /odometry/global (KF1)  /odometry/global2 (KF2)
    #              /odometry/fused (training target)
    # Publishes  : /ann/trajectory  /ann/target_vis
    # Trains     : every 5 s in background thread (≥500 samples needed)
    ann_node = Node(
        package='robot_control_brain',
        executable='trajectory_nn_node',
        name='online_training_node',
        output='screen',
        parameters=[sim_time_param],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── Trajectory Controller ─────────────────────────────────────────────────
    # FIX 4: Remapped feedback from raw /odom → /odometry/global (EKF1 output).
    # Raw wheel odometry drifts freely in the odom frame.  The controller was
    # therefore computing waypoint errors in a frame that diverged from the map
    # frame over time, causing the robot to navigate to the wrong world positions.
    # /odometry/global is GPS+IMU corrected and lives in the map frame, so
    # waypoint coordinates (which are map-frame targets) are now consistent with
    # the robot's actual position estimate.
    #
    # NOTE: ekf_local (EKF1) starts at ekf_delay (default 8s), before this node
    # at fusion_delay (default 12s), so /odometry/global is guaranteed to be
    # publishing before the controller needs it.
    trajectory_controller = Node(
        package='control_pkg',
        executable='trajectory_controller',
        name='trajectory_controller',
        output='screen',
        parameters=[cfg_waypoints, sim_time_param],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── RViz2 (optional) ─────────────────────────────────────────────────────
    rviz_config = os.path.join(pkg_bt, 'rviz', 'system.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # Use config if it exists; otherwise RViz opens with defaults
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        parameters=[sim_time_param],
        condition=IfCondition(use_rviz),
    )

    phase3 = TimerAction(
        period=fusion_delay,
        actions=[
            phase3_log,
            complementary_filter,
            ann_node,
            trajectory_controller,
            rviz,
        ]
    )

    # ═════════════════════════════════════════════════════════════════════════
    #  PHASE 4  –  BT BRAIN  (t = bt_delay)
    #  Started last: the BT requires all sensor topics and EKF outputs.
    # ═════════════════════════════════════════════════════════════════════════
    phase4_log = LogInfo(msg='\n'
        '╔═════════════════════════════════════════════════════════════════╗\n'
        '║  PHASE 4 │ Starting BT Brain (GPS/INS/Odometer Orchestrator)  ║\n'
        '╚═════════════════════════════════════════════════════════════════╝')

    # ── BT Orchestrator ───────────────────────────────────────────────────────
    # Subscribes : /gps/fix  /imu/data  /odom  /odometry/gps
    #              /odometry/global  /odometry/global2  /ann/trajectory
    # Publishes  : /odometry/bt_fused   (authoritative fused position)
    #              /bt/fuzzy_weights    (α1, α2, slip_error diagnostic)
    #              /bt/status           (active branch label)
    #              /bt/indoor_detection (IndoorDetector diagnostic)
    # Groot2     : ZMQ port 1666/1667  (connect Groot2 to visualize live BT)
    bt_brain = Node(
        package='bt_orchestrator_pkg',
        executable='bt_brain',
        name='bt_brain',
        output='screen',
        parameters=[sim_time_param],
        arguments=['--ros-args', '--log-level', log_level],
    )

    phase4 = TimerAction(
        period=bt_delay,
        actions=[
            phase4_log,
            bt_brain,
        ]
    )

    # ═════════════════════════════════════════════════════════════════════════
    #  T2 FIX — set Gazebo /clock publish rate to 100 Hz
    # ═════════════════════════════════════════════════════════════════════════
    # Root cause (confirmed via /proc/<pid>/cmdline):
    #   gzserver.launch.py in Humble gazebo_ros_pkgs passes no --ros-args to
    #   gzserver.  The /gazebo ROS node declares 'publish_rate' (confirmed via
    #   ros2 param list) but the default of 10.0 Hz is never overridden at
    #   startup.  Passing --ros-args directly to gzserver via ExecuteProcess
    #   crashes it (Gazebo's own arg parser runs first and misinterprets the
    #   trailing args as a state-log filename).
    #
    # Fix: after gzserver is up (~5 s), call 'ros2 param set /gazebo
    # publish_rate 100.0'.  The gazebo_ros_init plugin registers a parameter
    # callback that cancels the old 10 Hz timer and creates a 100 Hz one.
    # Result: /clock → ~100 Hz → EKF sim-time timers unlock to 30 Hz.
    set_clock_rate = TimerAction(
        period=5.0,    # /gazebo node is reliably up by t=5 s
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'param', 'set', '/gazebo', 'publish_rate', '100.0'],
                output='screen',
            )
        ]
    )

    # ═════════════════════════════════════════════════════════════════════════
    #  ASSEMBLE
    # ═════════════════════════════════════════════════════════════════════════
    startup_log = LogInfo(msg='\n\n'
        '╔══════════════════════════════════════════════════════════════════╗\n'
        '║           GPS / INS / Odometer Fusion System Starting           ║\n'
        '╠══════════════════════════════════════════════════════════════════╣\n'
        '║  Phase 1 │ t=0 s    Gazebo + Robot                             ║\n'
        '║  Phase 2 │ t=ekf_delay   (default 8s)   EKF stacks             ║\n'
        '║  Phase 3 │ t=fusion_delay (default 12s)  Fusion + ANN + Nav    ║\n'
        '║  Phase 4 │ t=bt_delay   (default 9s)    BT Brain               ║\n'
        '║                                                                  ║\n'
        '║  Groot2: Connect on ZMQ port 1666 after Phase 4 starts          ║\n'
        '║  /odometry/bt_fused: authoritative position output              ║\n'
        '╚══════════════════════════════════════════════════════════════════╝\n')

    return LaunchDescription(
        args + [
            startup_log,

            # ── Phase 1: Simulation (immediate) ──────────────────────────────
            phase1_log,
            simulation,
            set_clock_rate,   # T2: set /gazebo publish_rate=100 Hz at t=5s

            # ── Phase 2: EKF stacks (delayed) ────────────────────────────────
            phase2,

            # ── Phase 3: Fusion + Intelligence (delayed) ─────────────────────
            phase3,

            # ── Phase 4: BT Brain (delayed, last) ────────────────────────────
            phase4,
        ]
    )