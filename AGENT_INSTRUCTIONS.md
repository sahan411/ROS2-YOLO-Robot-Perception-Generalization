## PROJECT CONTEXT — Read before writing any code

**Project:** ROS2 YOLO Robot Perception Generalization
**Repo:** ~/Sahan/ROS2-YOLO-Robot-Perception-Generalization/
**OS:** Ubuntu 24.04.4 LTS
**ROS2:** Jazzy (NOT Humble — APIs differ)
**Gazebo:** Harmonic version 8.11.0 (NOT Classic/Gazebo 11)
**Robot:** TurtleBot3 Waffle Pi (has camera)
**Python:** 3.12.3
**GPU:** RTX 5090, CUDA 13.1
**Access:** SSH only (no display available remotely)

---

## CRITICAL ROS2 JAZZY RULES — Never break these

1. cmd_vel uses `geometry_msgs/msg/TwistStamped` NOT `Twist`
   Always set header.stamp = self.get_clock().now().to_msg()

2. Camera topic is `/camera/image_raw` (sensor_msgs/msg/Image)

3. cv_bridge import: `from cv_bridge import CvBridge`
   Instance: `self.bridge = CvBridge()`
   Convert: `cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")`

4. Always destroy_node() and rclpy.shutdown() in finally block

5. ROS2 parameters declared BEFORE used:
   `self.declare_parameter('name', default_value)`
   `value = self.get_parameter('name').get_parameter_value().string_value`

6. Timer callback signature: `def callback(self):` — no arguments

7. Package dependencies go in BOTH package.xml AND setup.py

---

## WHAT TO AVOID

- Never use rospy (ROS1)
- Never use `Twist` for cmd_vel (use TwistStamped)
- Never use `time.sleep()` inside ROS2 nodes (use timer states)
- Never use global variables — use class attributes
- Never assume directory exists — always os.makedirs(exist_ok=True)
- Never use `ros::spin()` (C++ syntax)
- Never import opencv as `import cv2` without installing python3-opencv

---

## FOLDER STRUCTURE

```
ros2_ws/src/data_collection/
├── data_collection/
│   ├── __init__.py
│   ├── robot_drive_node.py
│   └── image_capture_node.py
├── package.xml
├── setup.py
└── setup.cfg
```

---

## CORRECT setup.py FORMAT

```python
from setuptools import find_packages, setup

package_name = 'data_collection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sdvn_hinder_metric',
    maintainer_email='sdvn_hinder_metric@todo.todo',
    description='Data collection nodes for YOLO generalization project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_driver = data_collection.robot_drive_node:main',
            'image_capture = data_collection.image_capture_node:main',
        ],
    },
)
```

---

## CORRECT package.xml DEPENDENCIES

```xml
<exec_depend>rclpy</exec_depend>
<exec_depend>sensor_msgs</exec_depend>
<exec_depend>geometry_msgs</exec_depend>
<exec_depend>cv_bridge</exec_depend>
<exec_depend>std_msgs</exec_depend>
```

---

## HOW THE TWO NODES CONNECT

```
Gazebo simulation
      │
      ▼
/camera/image_raw  ──►  image_capture_node  ──►  saves .jpg to datasets/env_a/
      
/cmd_vel  ◄──  robot_drive_node  (publishes TwistStamped)
      │
      ▼
TurtleBot3 moves in Gazebo
```

---

## DEVELOPMENT WORKFLOW & TESTING

- Build command is always:
  cd ~/Sahan/ROS2-YOLO-Robot-Perception-Generalization/ros2_ws
  colcon build --packages-select <package_name>
  
- After every build, source:
  source ~/Sahan/ROS2-YOLO-Robot-Perception-Generalization/ros2_ws/install/setup.bash

- Headless Gazebo launch (SSH only, no display):
  export QT_QPA_PLATFORM=offscreen
  ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py gui:=false

- Always test node registration before full run:
  ros2 run <package> <node> --help
