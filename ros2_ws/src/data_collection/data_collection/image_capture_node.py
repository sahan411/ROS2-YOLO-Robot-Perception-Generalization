import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture')
        
        # Declare and get parameter for save directory
        self.declare_parameter('save_dir', '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization/datasets/env_a')
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'Images will be saved to: {self.save_dir}')

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
            
        self.cv_bridge = CvBridge()
        
        # State variables
        self.last_save_time = time.time()
        self.save_interval = 2.0  # seconds
        self.image_count = 0

    def image_callback(self, msg):
        current_time = time.time()
        
        # Time-based throttle: only save if 2 seconds have passed
        if (current_time - self.last_save_time) >= self.save_interval:
            self.last_save_time = current_time
            self.image_count += 1
            
            try:
                # Convert ROS Image to OpenCV format
                # Using 'bgr8' to ensure color channels are correct for cv2.imwrite
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                # Format filename: frame_00001.jpg
                filename = f"frame_{self.image_count:05d}.jpg"
                filepath = os.path.join(self.save_dir, filename)
                
                # Save image
                cv2.imwrite(filepath, cv_image)
                
                # Print progress every 50 images
                if self.image_count % 50 == 0:
                    self.get_logger().info(f'Saved {self.image_count} images to {self.save_dir}')
                    
            except Exception as e:
                self.get_logger().error(f'Failed to process image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
