from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    engine_arg = DeclareLaunchArgument(
        'engine', default_value='auto',
        choices=['auto', 'whisper', 'vosk', 'google', 'cli'],
        description='Bộ nhận dạng giọng nói: auto (ưu tiên Whisper -> Vosk -> CLI), whisper, vosk, google, cli'
    )
    model_size_arg = DeclareLaunchArgument(
        'whisper_model', default_value='base',
        description='Kích thước mô hình Whisper (tiny, base, small)'
    )
    show_gui_arg = DeclareLaunchArgument(
        'show_map_gui', default_value='true',
        choices=['true', 'false'],
        description='Hiển thị cửa sổ OpenCV trực quan các phân vùng ngữ nghĩa và đường đi dự kiến'
    )
    dry_run_arg = DeclareLaunchArgument(
        'dry_run', default_value='false',
        choices=['true', 'false'],
        description='Chế độ kiểm thử không cần đợi Nav2 Action Server (dùng khi test câu lệnh độc lập)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        choices=['true', 'false'],
        description='Sử dụng simulation clock'
    )

    rmw_env = SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    qt_qpa_env = SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb')

    voice_nav_node = Node(
        package='rua4_giongnoi',
        executable='voice_navigation_node',
        name='voice_navigation_node',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=[
            '--show-gui', LaunchConfiguration('show_map_gui'),
            '--dry-run', LaunchConfiguration('dry_run'),
        ]
    )

    speech_node = Node(
        package='rua4_giongnoi',
        executable='speech_recognition_node',
        name='speech_recognition_node',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=[
            '--engine', LaunchConfiguration('engine'),
            '--whisper-model', LaunchConfiguration('whisper_model'),
        ]
    )

    ld = LaunchDescription()
    ld.add_action(rmw_env)
    ld.add_action(qt_qpa_env)
    ld.add_action(engine_arg)
    ld.add_action(model_size_arg)
    ld.add_action(show_gui_arg)
    ld.add_action(dry_run_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(voice_nav_node)
    ld.add_action(speech_node)
    return ld
