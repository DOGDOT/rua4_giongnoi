import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
]


def launch_setup(context, *args, **kwargs):
    namespace = context.launch_configurations.get('namespace', '')
    pkg_create3_control = get_package_share_directory('irobot_create_control')
    control_params_file = os.path.join(pkg_create3_control, 'config', 'control.yaml')

    diffdrive_controller_node = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=['diffdrive_controller',
                   '--param-file', control_params_file,
                   '--controller-manager-timeout', '60.0',
                   '--service-call-timeout', '60.0',
                   '--switch-timeout', '60.0'],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=['joint_state_broadcaster',
                   '--controller-manager-timeout', '60.0',
                   '--service-call-timeout', '60.0',
                   '--switch-timeout', '60.0'],
        output='screen',
    )

    # Đảm bảo diffdrive_controller chạy sau khi joint_state_broadcaster kết thúc
    diffdrive_controller_callback = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diffdrive_controller_node],
        )
    )

    actions = [
        joint_state_broadcaster_spawner,
        diffdrive_controller_callback,
    ]

    if namespace:
        tf_namespaced_odom_publisher = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_namespaced_odom_publisher',
            arguments=['0', '0', '0',
                       '0', '0', '0',
                       'odom', f'{namespace}/odom'],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static')
            ],
            output='screen',
        )

        tf_namespaced_base_link_publisher = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_namespaced_base_link_publisher',
            arguments=['0', '0', '0',
                       '0', '0', '0',
                       f'{namespace}/base_link', 'base_link'],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static')
            ],
            output='screen',
        )
        actions.extend([tf_namespaced_odom_publisher, tf_namespaced_base_link_publisher])

    return actions


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
