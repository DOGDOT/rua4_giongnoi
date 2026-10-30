import os
from ament_index_python.packages import get_package_share_directory

from irobot_create_common_bringup.namespace import GetNamespacedName
from irobot_create_common_bringup.offset import OffsetParser

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node, PushRosNamespace

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'],
                          description='use_sim_time'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('spawn_dock', default_value='false',
                          choices=['true', 'false'],
                          description='Có tạo trạm sạc Dock hay không (Mặc định: false để tránh kẹt robot)'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():
    pkg_rua4 = get_package_share_directory('rua4_giongnoi')
    pkg_turtlebot4_ignition_bringup = get_package_share_directory(
        'turtlebot4_ignition_bringup')
    pkg_turtlebot4_description = get_package_share_directory(
        'turtlebot4_description')
    pkg_irobot_create_common_bringup = get_package_share_directory(
        'irobot_create_common_bringup')
    pkg_irobot_create_ignition_bringup = get_package_share_directory(
        'irobot_create_ignition_bringup')

    # Paths
    turtlebot4_ros_ign_bridge_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'ros_ign_bridge.launch.py'])
    turtlebot4_node_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'turtlebot4_nodes.launch.py'])
    
    # Dùng create3_nodes tùy biến của rua4 (đã tích hợp switch-timeout 60s tránh lỗi controller_manager)
    create3_nodes_launch = PathJoinSubstitution(
        [pkg_rua4, 'launch', 'include', 'create3_nodes.launch.py'])
    create3_ignition_nodes_launch = PathJoinSubstitution(
        [pkg_irobot_create_ignition_bringup, 'launch', 'create3_ignition_nodes.launch.py'])
    xacro_file = PathJoinSubstitution(
        [pkg_turtlebot4_description, 'urdf', LaunchConfiguration('model'), 'turtlebot4.urdf.xacro'])
    dock_description_launch = PathJoinSubstitution(
        [pkg_irobot_create_common_bringup, 'launch', 'dock_description.launch.py'])

    param_file_cmd = DeclareLaunchArgument(
        'param_file',
        default_value=PathJoinSubstitution(
            [pkg_turtlebot4_ignition_bringup, 'config', 'turtlebot4_node.yaml']),
        description='Turtlebot4 Robot param file')

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    x, y, z = LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')
    spawn_dock = LaunchConfiguration('spawn_dock')
    turtlebot4_node_yaml_file = LaunchConfiguration('param_file')

    robot_name = GetNamespacedName(namespace, 'turtlebot4')
    dock_name = GetNamespacedName(namespace, 'standard_dock')

    # Cho robot xuất hiện sát mặt sàn (1.5 cm) để bánh xe tiếp đất phẳng tự nhiên ngay lập tức
    z_robot = OffsetParser(z, 0.015)

    # Nếu người dùng muốn bật dock, đặt dock lùi ra sau robot 2m để không chạm bánh xe
    x_dock = OffsetParser(x, -2.0)
    y_dock = OffsetParser(y, 0.0)
    yaw_dock = OffsetParser(yaw, 3.1416)

    spawn_robot_group_action = GroupAction([
        PushRosNamespace(namespace),

        # Robot State Publisher (Chỉ khởi chạy robot_state_publisher, KHÔNG chạy joint_state_publisher
        # để tránh xung đột /joint_states với joint_state_broadcaster của ros2_control làm hỏng TF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_description': Command([
                    'xacro ', xacro_file, ' ',
                    'gazebo:=ignition ',
                    'namespace:=', namespace])},
            ],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static')
            ]
        ),

        # Dock description (chỉ tải khi spawn_dock=true)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([dock_description_launch]),
            launch_arguments={'gazebo': 'ignition'}.items(),
            condition=IfCondition(spawn_dock)
        ),

        # Spawn TurtleBot 4 trực tiếp trên sàn phẳng
        Node(
            package='ros_ign_gazebo',
            executable='create',
            arguments=['-name', robot_name,
                       '-x', x,
                       '-y', y,
                       '-z', z_robot,
                       '-Y', yaw,
                       '-topic', 'robot_description'],
            output='screen'
        ),

        # Spawn Dock (chỉ khi spawn_dock=true, đặt cách xa 2m)
        Node(
            package='ros_ign_gazebo',
            executable='create',
            arguments=['-name', dock_name,
                       '-x', x_dock,
                       '-y', y_dock,
                       '-z', z,
                       '-Y', yaw_dock,
                       '-topic', 'standard_dock_description'],
            output='screen',
            condition=IfCondition(spawn_dock)
        ),

        # ROS IGN bridge
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([turtlebot4_ros_ign_bridge_launch]),
            launch_arguments=[
                ('model', LaunchConfiguration('model')),
                ('robot_name', robot_name),
                ('dock_name', dock_name),
                ('namespace', namespace)]
        ),

        # TurtleBot 4 nodes
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([turtlebot4_node_launch]),
            launch_arguments=[('model', LaunchConfiguration('model')),
                              ('param_file', turtlebot4_node_yaml_file)]
        ),

        # Create 3 nodes (sử dụng cấu hình của rua4)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([create3_nodes_launch]),
            launch_arguments=[
                ('namespace', namespace)
            ]
        ),

        # Create 3 Ignition nodes
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([create3_ignition_nodes_launch]),
            launch_arguments=[
                ('robot_name', robot_name),
                ('dock_name', dock_name),
            ]
        ),

        # RPLIDAR static transforms
        Node(
            name='rplidar_stf',
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=[
                '0', '0', '0', '0', '0', '0.0',
                'rplidar_link', [robot_name, '/rplidar_link/rplidar']],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ]
        ),

        # OAKD static transform
        Node(
            name='camera_stf',
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=[
                '0', '0', '0',
                '1.5707', '-1.5707', '0',
                'oakd_rgb_camera_optical_frame',
                [robot_name, '/oakd_rgb_camera_frame/rgbd_camera']
            ],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ]
        ),
    ])

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(param_file_cmd)
    ld.add_action(spawn_robot_group_action)
    return ld
