import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Đảm bảo Ignition Gazebo tìm thấy gz_ros2_control system plugins trong /opt/ros/humble/lib
    plugin_path = os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')
    if '/opt/ros/humble/lib' not in plugin_path:
        plugin_path = f"/opt/ros/humble/lib:{plugin_path}" if plugin_path else "/opt/ros/humble/lib"
    os.environ['IGN_GAZEBO_SYSTEM_PLUGIN_PATH'] = plugin_path

    ign_plugin_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
        value=plugin_path
    )

    rmw_env = SetEnvironmentVariable(
        name='RMW_IMPLEMENTATION',
        value='rmw_cyclonedds_cpp'
    )

    qt_qpa_env = SetEnvironmentVariable(
        name='QT_QPA_PLATFORM',
        value='xcb'
    )

    pkg_rua4 = get_package_share_directory('rua4_giongnoi')
    pkg_tb4_ignition = get_package_share_directory('turtlebot4_ignition_bringup')

    # Arguments
    world_arg = DeclareLaunchArgument(
        'world', default_value='warehouse',
        description='Môi trường Gazebo (warehouse, maze, depot)')
    model_arg = DeclareLaunchArgument(
        'model', default_value='standard',
        choices=['standard', 'lite'],
        description='Phiên bản robot: standard (có lidar & camera OAK-D) hoặc lite')
    spawn_dock_arg = DeclareLaunchArgument(
        'spawn_dock', default_value='false',
        choices=['true', 'false'],
        description='Có tạo trạm sạc Dock hay không (Mặc định: false để robot đứng trên sàn phẳng, không bị kẹt)')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        choices=['true', 'false'],
        description='Khởi chạy RViz2 trực quan hóa sensor & robot')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Sử dụng simulation clock')
    x_arg = DeclareLaunchArgument('x', default_value='0.0', description='Tọa độ x ban đầu')
    y_arg = DeclareLaunchArgument('y', default_value='0.0', description='Tọa độ y ban đầu')
    z_arg = DeclareLaunchArgument('z', default_value='0.0', description='Tọa độ z ban đầu')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0', description='Góc quay yaw ban đầu')

    # 1. Khởi chạy môi trường Gazebo Ignition (scoped để không làm nhiễm launch configurations)
    ignition_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_tb4_ignition, 'launch', 'ignition.launch.py'])
                ),
                launch_arguments={
                    'world': LaunchConfiguration('world'),
                    'model': LaunchConfiguration('model'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }.items()
            )
        ],
        scoped=True
    )

    # 2. Khởi chạy Spawn robot tự do trên sàn phẳng (không bị kẹt trên dock)
    robot_spawn_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_rua4, 'launch', 'turtlebot4_spawn.launch.py'])
                ),
                launch_arguments={
                    'namespace': '',
                    'model': LaunchConfiguration('model'),
                    'spawn_dock': LaunchConfiguration('spawn_dock'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'x': LaunchConfiguration('x'),
                    'y': LaunchConfiguration('y'),
                    'z': LaunchConfiguration('z'),
                    'yaw': LaunchConfiguration('yaw'),
                }.items()
            )
        ],
        scoped=True
    )

    # 3. RViz2 trực quan hóa đồng thời (Mô hình robot 3D, TF, /scan, /odom)
    rviz_config = PathJoinSubstitution([pkg_rua4, 'config', 'rviz', 'rua4_sim.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        additional_env={
            'QT_QPA_PLATFORM': 'xcb',
            'WAYLAND_DISPLAY': '',
        },
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    ld = LaunchDescription()
    ld.add_action(ign_plugin_path)
    ld.add_action(rmw_env)
    ld.add_action(qt_qpa_env)
    ld.add_action(world_arg)
    ld.add_action(model_arg)
    ld.add_action(spawn_dock_arg)
    ld.add_action(rviz_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(x_arg)
    ld.add_action(y_arg)
    ld.add_action(z_arg)
    ld.add_action(yaw_arg)
    ld.add_action(ignition_launch)
    ld.add_action(robot_spawn_launch)
    ld.add_action(rviz_node)

    return ld
