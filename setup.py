import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rua4_giongnoi'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'launch', 'include'),
            glob(os.path.join('launch', 'include', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml')) +
            glob(os.path.join('config', '*.json'))),
        (os.path.join('share', package_name, 'config', 'rviz'),
            glob(os.path.join('config', 'rviz', '*.rviz'))),
        (os.path.join('share', package_name, 'maps'),
            glob(os.path.join('maps', '*.yaml')) +
            glob(os.path.join('maps', '*.pgm')) +
            glob(os.path.join('maps', '*.png'))),
        (os.path.join('share', package_name, 'results'),
            glob(os.path.join('results', '*.png')) +
            glob(os.path.join('results', '*.pgm')) +
            glob(os.path.join('results', '*.yaml')) +
            glob(os.path.join('results', '*.json')) +
            glob(os.path.join('results', '*.txt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Phạm Thành Đông',
    maintainer_email='23020735@vnu.edu.vn',
    description='Phân vùng bản đồ Occupancy Grid, Gắn nhãn ngữ nghĩa và Điều hướng TurtleBot 4 bằng giọng nói (ROS 2 Humble)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_segmenter = rua4_giongnoi.map_segmentation:main',
            'speech_recognition_node = rua4_giongnoi.speech_recognition_node:main',
            'voice_navigation_node = rua4_giongnoi.voice_navigation_node:main',
            'semantic_map_viewer = rua4_giongnoi.semantic_map_viewer:main',
        ],
    },
)
