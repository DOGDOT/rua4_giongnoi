import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg_rua4 = get_package_share_directory('rua4_giongnoi')
    pkg_tb4_nav = get_package_share_directory('turtlebot4_navigation')

    launch_sim_str = LaunchConfiguration('launch_sim').perform(context).lower()
    use_sim = (launch_sim_str in ['true', '1'])

    sim_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_rua4, 'launch', 'sim.launch.py'])
                ),
                launch_arguments={
                    'world': LaunchConfiguration('world'),
                    'rviz': 'false',
                    'use_sim_time': LaunchConfiguration('use_sim_time')
                }.items(),
            )
        ],
        scoped=True,
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_tb4_nav, 'launch', 'localization.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'map': LaunchConfiguration('map'),
            'params': LaunchConfiguration('localization_params')
        }.items()
    )

    nav2_stack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_tb4_nav, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': LaunchConfiguration('nav2_params')
        }.items()
    )

    rviz_config = PathJoinSubstitution([pkg_rua4, 'config', 'rviz', 'rua4_nav2.rviz'])
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

    actions = []
    if use_sim:
        actions.append(sim_launch)
        # Cho Gazebo 6.0s để nạp world, spawn robot và ổn định trước khi localization/nav2 khởi động
        actions.append(TimerAction(period=6.0, actions=[localization_launch, nav2_stack_launch, rviz_node]))
    else:
        actions.extend([localization_launch, nav2_stack_launch, rviz_node])

    return actions


def generate_launch_description():
    pkg_rua4 = get_package_share_directory('rua4_giongnoi')

    launch_sim_arg = DeclareLaunchArgument(
        'launch_sim', default_value='false',
        choices=['true', 'false'],
        description='Tự động khởi chạy mô phỏng Gazebo nếu chưa chạy')
    world_arg = DeclareLaunchArgument(
        'world', default_value='warehouse',
        description='Môi trường mô phỏng (warehouse, maze, depot)')
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([pkg_rua4, 'maps', 'map.yaml']),
        description='Đường dẫn tuyệt đối đến file map.yaml')
    localization_params_arg = DeclareLaunchArgument(
        'localization_params',
        default_value=PathJoinSubstitution([pkg_rua4, 'config', 'localization.yaml']),
        description='File tham số Localization (AMCL & Map Server)')
    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params',
        default_value=PathJoinSubstitution([pkg_rua4, 'config', 'nav2.yaml']),
        description='File tham số Nav2 (Planner, Controller, Costmaps, BT)')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        choices=['true', 'false'],
        description='Mở RViz2 hiển thị Nav2 (Bản đồ, Costmaps, Path, Nav2 Goal tool)')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Sử dụng simulation clock')

    rmw_env = SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    qt_qpa_env = SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb')

    ld = LaunchDescription()
    ld.add_action(rmw_env)
    ld.add_action(qt_qpa_env)
    ld.add_action(launch_sim_arg)
    ld.add_action(world_arg)
    ld.add_action(map_arg)
    ld.add_action(localization_params_arg)
    ld.add_action(nav2_params_arg)
    ld.add_action(rviz_arg)
    ld.add_action(use_sim_time_arg)

    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld
