from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get Hardware/Servo launch files from other packages
    gazebo_launch_path = os.path.join(
        get_package_share_directory('open_manipulator_x_bringup'),
        'launch',
        'gazebo.launch.py'
    )

    moveit_launch_path = os.path.join(
        get_package_share_directory('open_manipulator_x_moveit_config'),
        'launch',
        'moveit_core.launch.py'
    )

    servo_launch_path = os.path.join(
        get_package_share_directory('open_manipulator_x_moveit_config'),
        'launch',
        'servo.launch.py'
    )

    return LaunchDescription([     
        # Gazebo/Servo Launch (REQUIRED TO RUN ARM)
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gazebo_launch_path)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(moveit_launch_path)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(servo_launch_path)),

        # Safety controller
        Node(package='rbe575project', executable='safety_controller', output='screen'),
    ])
