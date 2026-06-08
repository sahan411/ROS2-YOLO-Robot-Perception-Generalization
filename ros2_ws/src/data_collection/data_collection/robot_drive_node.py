import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import random
import time

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        self.publisher_ = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.state = 'FORWARD'
        self.state_start_time = self.get_clock().now()
        self.turn_direction = 1.0
        
        self.get_logger().info('Robot Driver Node Started')

    def timer_callback(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.state_start_time).nanoseconds / 1e9

        if self.state == 'FORWARD':
            msg.twist.linear.x = 0.15
            msg.twist.angular.z = 0.0
            if elapsed_time >= 3.0:
                self.state = 'STOP'
                self.state_start_time = current_time
                self.get_logger().info('Transitioning to STOP')
                
        elif self.state == 'STOP':
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
            if elapsed_time >= 0.5:
                self.state = 'TURN'
                self.state_start_time = current_time
                self.turn_direction = random.choice([-1.0, 1.0])
                self.get_logger().info(f'Transitioning to TURN (dir: {self.turn_direction})')
                
        elif self.state == 'TURN':
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.5 * self.turn_direction
            if elapsed_time >= 2.0:
                self.state = 'FORWARD'
                self.state_start_time = current_time
                self.get_logger().info('Transitioning to FORWARD')

        self.publisher_.publish(msg)

    def stop_robot(self):
        self.get_logger().info('Stopping robot before exit...')
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.publisher_.publish(msg)
        time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    robot_driver = RobotDriver()
    try:
        rclpy.spin(robot_driver)
    except KeyboardInterrupt:
        robot_driver.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        robot_driver.stop_robot()
        robot_driver.destroy_node()
        # check if rclpy is okay to shutdown
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
