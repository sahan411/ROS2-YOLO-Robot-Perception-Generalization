import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Force Qt to use offscreen platform to prevent Gazebo GUI crash over SSH
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # Declare launch argument
    save_dir_arg = DeclareLaunchArgument(
        'save_dir',
        default_value='/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization/datasets/env_a',
        description='Directory to save captured images'
    )

    # 1. Include turtlebot3_gazebo launch file
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    gazebo_launch_file = os.path.join(turtlebot3_gazebo_dir, 'launch', 'turtlebot3_world.launch.py')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'gui': 'false'}.items()
    )

    # 2. Robot driver node with 5 second delay
    robot_driver_node = Node(
        package='data_collection',
        executable='robot_driver',
        name='robot_driver',
        output='screen'
    )

    delayed_robot_driver = TimerAction(
        period=5.0,
        actions=[robot_driver_node]
    )

    # 3. Image capture node with 6 second delay
    image_capture_node = Node(
        package='data_collection',
        executable='image_capture',
        name='image_capture',
        parameters=[{'save_dir': LaunchConfiguration('save_dir')}],
        output='screen'
    )

    delayed_image_capture = TimerAction(
        period=6.0,
        actions=[image_capture_node]
    )

    return LaunchDescription([
        save_dir_arg,
        gazebo_launch,
        delayed_robot_driver,
        delayed_image_capture
    ])
