import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('gazebo', default_value='ignition',
                          choices=['classic', 'ignition'],
                          description='Which gazebo simulator to use'),
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
]


def generate_launch_description():
    pkg_self = get_package_share_directory('rua4_giongnoi')
    pkg_create3_common_bringup = get_package_share_directory('irobot_create_common_bringup')

    control_launch_file = PathJoinSubstitution(
        [pkg_self, 'launch', 'include', 'control.launch.py'])

    hazards_params_yaml_file = PathJoinSubstitution(
        [pkg_create3_common_bringup, 'config', 'hazard_vector_params.yaml'])
    ir_intensity_params_yaml_file = PathJoinSubstitution(
        [pkg_create3_common_bringup, 'config', 'ir_intensity_vector_params.yaml'])
    wheel_status_params_yaml_file = PathJoinSubstitution(
        [pkg_create3_common_bringup, 'config', 'wheel_status_params.yaml'])
    mock_params_yaml_file = PathJoinSubstitution(
        [pkg_create3_common_bringup, 'config', 'mock_params.yaml'])
    robot_state_yaml_file = PathJoinSubstitution(
        [pkg_create3_common_bringup, 'config', 'robot_state_params.yaml'])
    kidnap_estimator_yaml_file = PathJoinSubstitution(
        [pkg_create3_common_bringup, 'config', 'kidnap_estimator_params.yaml'])
    ui_mgr_params_yaml_file = PathJoinSubstitution(
        [pkg_create3_common_bringup, 'config', 'ui_mgr_params.yaml'])

    diffdrive_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([control_launch_file]),
        launch_arguments=[('namespace', LaunchConfiguration('namespace'))]
    )

    hazards_vector_node = Node(
        package='irobot_create_nodes',
        name='hazards_vector_publisher',
        executable='hazards_vector_publisher',
        parameters=[hazards_params_yaml_file, {'use_sim_time': True}],
        output='screen',
    )

    ir_intensity_vector_node = Node(
        package='irobot_create_nodes',
        name='ir_intensity_vector_publisher',
        executable='ir_intensity_vector_publisher',
        parameters=[ir_intensity_params_yaml_file, {'use_sim_time': True}],
        output='screen',
    )

    motion_control_node = Node(
        package='irobot_create_nodes',
        name='motion_control',
        executable='motion_control',
        parameters=[{'use_sim_time': True, 'safety_override': 'backup_only'}],
        output='screen',
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    wheel_status_node = Node(
        package='irobot_create_nodes',
        name='wheel_status_publisher',
        executable='wheel_status_publisher',
        parameters=[wheel_status_params_yaml_file, {'use_sim_time': True}],
        output='screen',
    )

    mock_topics_node = Node(
        package='irobot_create_nodes',
        name='mock_publisher',
        executable='mock_publisher',
        parameters=[mock_params_yaml_file, {'use_sim_time': True}],
        output='screen',
    )

    robot_state_node = Node(
        package='irobot_create_nodes',
        name='robot_state',
        executable='robot_state',
        parameters=[robot_state_yaml_file, {'use_sim_time': True}],
        output='screen',
    )

    kidnap_estimator_node = Node(
        package='irobot_create_nodes',
        name='kidnap_estimator_publisher',
        executable='kidnap_estimator_publisher',
        parameters=[kidnap_estimator_yaml_file, {'use_sim_time': True}],
        output='screen',
    )

    ui_mgr_node = Node(
        package='irobot_create_nodes',
        name='ui_mgr',
        executable='ui_mgr',
        parameters=[ui_mgr_params_yaml_file,
                    {'use_sim_time': True},
                    {'gazebo': LaunchConfiguration('gazebo')}],
        output='screen',
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(diffdrive_controller)
    ld.add_action(hazards_vector_node)
    ld.add_action(ir_intensity_vector_node)
    ld.add_action(motion_control_node)
    ld.add_action(wheel_status_node)
    ld.add_action(mock_topics_node)
    ld.add_action(robot_state_node)
    ld.add_action(kidnap_estimator_node)
    ld.add_action(ui_mgr_node)

    return ld
