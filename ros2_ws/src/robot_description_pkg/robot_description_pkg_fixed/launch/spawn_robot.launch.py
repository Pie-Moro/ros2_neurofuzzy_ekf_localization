import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_name = 'robot_description_pkg'
    pkg_dir = get_package_share_directory(pkg_name)

    # 1. Launch Gazebo with the custom indoor/outdoor world
    world_file = os.path.join(pkg_dir, 'worlds', 'indoor_outdoor.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # 2. Resolve the xacro to a URDF string at launch time
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([FindPackageShare(pkg_name),
                              'urdf', 'tb3_custom.urdf.xacro'])
    ])

    # Force the parameter to be treated as a string (not a substitution)
    robot_description = ParameterValue(robot_description_content, value_type=str)

    # 3. Publish the TF tree
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }]
    )

    # 4. Spawn the robot in Gazebo.
    #
    # World wall envelope (world frame): X ∈ [-10.86, 11.20], Y ∈ [-4.15, 4.55].
    # Spawning at (5.0, 6.5) places the robot ~1.95 m north of the building —
    # clearly OUTDOORS, on open grass.
    #
    # z = 0.05 keeps the wheels just above the ground so they snap to contact
    # without a big drop. Dropping the robot from 0.3 m (previous value)
    # creates an initial angular impulse that can look like drift.
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'tb3_burger',
            '-x', '5.0', '-y', '6.5', '-z', '0.05',
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
