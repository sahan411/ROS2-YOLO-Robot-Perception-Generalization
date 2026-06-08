import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import yaml
import os
import time

class AutoLabelerNode(Node):
    def __init__(self):
        super().__init__('auto_labeler')
        
        # Declare parameters
        self.declare_parameter('save_dir', '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization/datasets/env_a')
        self.declare_parameter('objects_config', '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization/ros2_ws/src/data_collection/config/objects.yaml')
        self.declare_parameter('camera_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('world_frame', 'odom')
        
        self.save_dir = self.get_parameter('save_dir').value
        self.objects_config_path = self.get_parameter('objects_config').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        
        # Load objects config
        with open(self.objects_config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.objects = config.get('objects', [])
            self.class_names = config.get('class_names', {})
            
        os.makedirs(self.save_dir, exist_ok=True)
        
        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.bridge = CvBridge()
        
        self.intrinsics_ready = False
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.img_width = 0
        self.img_height = 0
        
        self.frame_counter = 1
        self.last_save_time = time.time()
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.cx = msg.k[2]
        self.fy = msg.k[4]
        self.cy = msg.k[5]
        self.img_width = msg.width
        self.img_height = msg.height
        self.intrinsics_ready = True
        self.get_logger().info(f"Camera intrinsics ready: fx={self.fx} fy={self.fy}")
        
        # Unsubscribe after getting intrinsics once
        self.destroy_subscription(self.camera_info_sub)
        
    def image_callback(self, msg):
        if not self.intrinsics_ready:
            return
            
        current_time = time.time()
        if current_time - self.last_save_time < 2.0:
            return
            
        try:
            # Look up transform from world_frame to camera_frame
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.world_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF failure: {str(e)}")
            return
            
        # Update throttle timer
        self.last_save_time = current_time
        
        # Extract translation and rotation
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z
        
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # Quaternion to Rotation Matrix
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[0, 1] = 2 * (qx*qy - qz*qw)
        R[0, 2] = 2 * (qx*qz + qy*qw)
        
        R[1, 0] = 2 * (qx*qy + qz*qw)
        R[1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[1, 2] = 2 * (qy*qz - qx*qw)
        
        R[2, 0] = 2 * (qx*qz - qy*qw)
        R[2, 1] = 2 * (qy*qz + qx*qw)
        R[2, 2] = 1 - 2 * (qx**2 + qy**2)
        
        translation = np.array([tx, ty, tz])
        
        yolo_labels = []
        
        for obj in self.objects:
            obj_pos = np.array(obj['position'])
            obj_size = obj['size']
            class_id = obj['class_id']
            
            # Transform object 3D center from world to camera optical frame
            obj_cam = R @ obj_pos + translation
            
            Z = obj_cam[2]
            if Z <= 0.05:
                continue
                
            X = obj_cam[0]
            Y = obj_cam[1]
            
            # Project to image pixels
            u = int(self.fx * X / Z + self.cx)
            v = int(self.fy * Y / Z + self.cy)
            
            # Visibility margin check
            margin = 20
            if u < margin or u > self.img_width - 1 - margin or v < margin or v > self.img_height - 1 - margin:
                continue
                
            # Compute bounding box
            half_w = self.fx * (obj_size[0] / 2.0) / Z
            half_h = self.fy * (obj_size[2] / 2.0) / Z
            
            x1 = max(0, int(u - half_w))
            y1 = max(0, int(v - half_h))
            x2 = min(self.img_width - 1, int(u + half_w))
            y2 = min(self.img_height - 1, int(v + half_h))
            
            area = (x2 - x1) * (y2 - y1)
            if area < 100:
                continue
                
            # YOLO normalized format
            cx_n = ((x1 + x2) / 2.0) / self.img_width
            cy_n = ((y1 + y2) / 2.0) / self.img_height
            w_n = (x2 - x1) / self.img_width
            h_n = (y2 - y1) / self.img_height
            
            yolo_labels.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            
        # Save image and label
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        img_filename = os.path.join(self.save_dir, f"frame_{self.frame_counter:05d}.jpg")
        txt_filename = os.path.join(self.save_dir, f"frame_{self.frame_counter:05d}.txt")
        
        cv2.imwrite(img_filename, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        with open(txt_filename, 'w') as f:
            if len(yolo_labels) > 0:
                f.write('\n'.join(yolo_labels) + '\n')
            else:
                f.write('') # Background image
                
        if self.frame_counter % 20 == 0:
            self.get_logger().info(f"Saved frame {self.frame_counter} | visible objects: {len(yolo_labels)}")
            
        self.frame_counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = AutoLabelerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
