#!/usr/bin/env python3

import pprint
import math
import rclpy
import threading
import numpy
import time
import av
import tf2_ros
import cv2
import time
import yaml

from djitellopy import Tello

from rclpy.node import Node
from tello_msg.msg import TelloStatus, TelloID, TelloWifiConfig
from std_msgs.msg import Empty, UInt8, UInt8, Bool, String
from sensor_msgs.msg import Image, Imu, BatteryState, Temperature, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from tello_srv.srv import TelloMoveSrv
from cv_bridge import CvBridge
import ament_index_python

# Tello ROS node class, inherits from the Tello controller object.
#
# Can be configured to be used by multiple drones, publishes, all data collected from the drone and provides control using ROS messages.
class TelloNode():
    def __init__(self, node):
        # ROS node
        self.node = node

        # Declare parameters
        self.node.declare_parameter('simulate', False)
        self.node.declare_parameter('connect_timeout', 10.0)
        self.node.declare_parameter('tello_ip', '192.168.10.1')
        self.node.declare_parameter('tf_base', 'map')
        self.node.declare_parameter('tf_drone', 'drone')
        self.node.declare_parameter('tf_pub', False)
        self.node.declare_parameter('camera_info_file', '')

        # Get parameters
        self.simulate = bool(self.node.get_parameter('simulate').value)
        if self.simulate:
            self.node.get_logger().info('Node in simulation (no actual drone connection) mode')

        self.connect_timeout = float(self.node.get_parameter('connect_timeout').value)
        self.tello_ip = str(self.node.get_parameter('tello_ip').value)
        self.tf_base = str(self.node.get_parameter('tf_base').value)
        self.tf_drone = str(self.node.get_parameter('tf_drone').value)
        self.tf_pub = bool(self.node.get_parameter('tf_pub').value)
        self.camera_info_file = str(self.node.get_parameter('camera_info_file').value)

        # Camera information loaded from calibration yaml
        self.camera_info = None

        # Check if camera info file was received as argument
        if len(self.camera_info_file) == 0:
            share_directory = ament_index_python.get_package_share_directory('tello')
            self.camera_info_file = share_directory + '/ost.yaml'

        # Read camera info from YAML file
        with open(self.camera_info_file, 'r') as file:
            self.camera_info = yaml.load(file, Loader=yaml.FullLoader)
            # self.node.get_logger().info('Tello: Camera information YAML' + self.camera_info.__str__())

        # Configure drone connection
        Tello.TELLO_IP = self.tello_ip
        Tello.RESPONSE_TIMEOUT = int(self.connect_timeout)

        # Define default control topic
        self.control_gamepad = True
        self.control_commander = False

        # RC control or high level move_* commands mode enabled
        self.rc_control_enable = True

        # Connect to drone
        self.node.get_logger().info('Tello: Connecting to drone')

        self.tello = Tello()
        if not self.simulate:
            self.tello.connect()
        else:
            self.node.get_logger().info('Node is simulated. Skipping connection to the real drone')


        self.node.get_logger().info('Tello: Connected to drone')

        # Publishers and subscribers
        self.setup_publishers()
        self.setup_subscribers()

        # Processing threads
        self.start_video_capture()
        self.start_tello_status()
        self.start_tello_odom()

        self.node.get_logger().info('Tello: Driver node ready')

    # Setup ROS publishers of the node.
    def setup_publishers(self):
        self.pub_image_raw = self.node.create_publisher(Image, 'image_raw', 1)
        self.pub_camera_info = self.node.create_publisher(CameraInfo, 'camera_info', 1)
        self.pub_status = self.node.create_publisher(TelloStatus, 'status', 1)
        self.pub_id = self.node.create_publisher(TelloID, 'id', 1)
        self.pub_imu = self.node.create_publisher(Imu, 'imu', 1)
        self.pub_battery = self.node.create_publisher(BatteryState, 'battery', 1)
        self.pub_temperature = self.node.create_publisher(Temperature, 'temperature', 1)
        self.pub_odom = self.node.create_publisher(Odometry, 'odom', 1)

        # TF broadcaster
        if self.tf_pub:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)

    # Setup the topic subscribers of the node.
    def setup_subscribers(self):
        # Services instead of Empty topics
        self.srv_emergency = self.node.create_service(Trigger, 'emergency', self.cb_emergency_srv)
        self.srv_takeoff   = self.node.create_service(Trigger, 'takeoff', self.cb_takeoff_srv)
        self.srv_land      = self.node.create_service(Trigger, 'land', self.cb_land_srv)
        self.srv_move      = self.node.create_service(TelloMoveSrv, 'move', self.cb_movement_command)
        # Topics
        self.sub_control = self.node.create_subscription(Twist, 'control', self.cb_control, 1)
        self.sub_control_gamepad = self.node.create_subscription(Twist, 'control_gamepad', self.cb_control_gamepad, 1)
        self.sub_control_owner = self.node.create_subscription(String, '/control_owner', self.control_owner_callback, 1)
        self.sub_flip = self.node.create_subscription(String, 'flip', self.cb_flip, 1)
        self.sub_wifi_config = self.node.create_subscription(TelloWifiConfig, 'wifi_config', self.cb_wifi_config, 1)

    # Get the orientation of the drone as a quaternion
    def get_orientation_quaternion(self):
        deg_to_rad = math.pi / 180.0
        ret = euler_to_quaternion([
            self.tello.get_yaw() * deg_to_rad if not self.simulate else 0,
            self.tello.get_pitch() * deg_to_rad if not self.simulate else 0,
            self.tello.get_roll() * deg_to_rad if not self.simulate else 0
        ])
        return ret

    # Start drone info thread
    def start_tello_odom(self, rate=1.0/10.0):
        def status_odom():
            while rclpy.ok():
                # TF
                if self.tf_pub:
                    t = TransformStamped()
                    t.header.stamp = self.node.get_clock().now().to_msg()
                    t.header.frame_id = self.tf_base
                    t.child_frame_id = self.tf_drone
                    t.transform.translation.x = 0.0
                    t.transform.translation.y = 0.0
                    if not self.simulate:
                        t.transform.translation.z = (self.tello.get_distance_tof()) / 100.0
                    self.tf_broadcaster.sendTransform(t)

                # IMU
                if self.pub_imu.get_subscription_count() > 0:
                    q = self.get_orientation_quaternion()

                    msg = Imu()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.header.frame_id = self.tf_drone
                    if not self.simulate:
                        msg.linear_acceleration.x = self.tello.get_acceleration_x() / 100.0
                        msg.linear_acceleration.y = self.tello.get_acceleration_y() / 100.0
                        msg.linear_acceleration.z = self.tello.get_acceleration_z() / 100.0
                    msg.orientation.x = q[0]
                    msg.orientation.y = q[1]
                    msg.orientation.z = q[2]
                    msg.orientation.w = q[3]
                    self.pub_imu.publish(msg)

                # Odometry
                if self.pub_odom.get_subscription_count() > 0:
                    q = self.get_orientation_quaternion()

                    odom_msg = Odometry()
                    odom_msg.header.stamp = self.node.get_clock().now().to_msg()
                    odom_msg.header.frame_id = self.tf_base
                    odom_msg.pose.pose.orientation.x = q[0]
                    odom_msg.pose.pose.orientation.y = q[1]
                    odom_msg.pose.pose.orientation.z = q[2]
                    odom_msg.pose.pose.orientation.w = q[3]
                    if not self.simulate:
                        odom_msg.pose.pose.position.z = float(self.tello.get_distance_tof()) / 100.0
                        odom_msg.twist.twist.linear.x = float(self.tello.get_speed_x()) / 100.0
                        odom_msg.twist.twist.linear.y = float(self.tello.get_speed_y()) / 100.0
                        odom_msg.twist.twist.linear.z = float(self.tello.get_speed_z()) / 100.0
                    self.pub_odom.publish(odom_msg)

                time.sleep(rate)

        thread = threading.Thread(target=status_odom)
        thread.start()
        return thread

    # Start drone info thread
    def start_tello_status(self, rate=1.0/2.0):
        def status_loop():
            while rclpy.ok():
                # Battery
                if self.pub_battery.get_subscription_count() > 0:
                    msg = BatteryState()
                    if not self.simulate:
                        msg.header.frame_id = self.tf_drone
                        msg.percentage = float(self.tello.get_battery())
                        msg.voltage = 3.8
                        msg.design_capacity = 1.1
                        msg.present = True
                        msg.power_supply_technology = 2 # POWER_SUPPLY_TECHNOLOGY_LION
                        msg.power_supply_status = 2 # POWER_SUPPLY_STATUS_DISCHARGING
                    self.pub_battery.publish(msg)

                # Temperature
                if self.pub_temperature.get_subscription_count() > 0:
                    msg = Temperature()
                    if not self.simulate:
                        msg.header.frame_id = self.tf_drone
                        msg.temperature = self.tello.get_temperature()
                        msg.variance = 0.0
                    self.pub_temperature.publish(msg)

                # Tello Status
                if self.pub_status.get_subscription_count() > 0:
                    msg = TelloStatus()
                    if not self.simulate:
                        msg.acceleration.x = self.tello.get_acceleration_x()
                        msg.acceleration.y = self.tello.get_acceleration_y()
                        msg.acceleration.z = self.tello.get_acceleration_z()

                        msg.speed.x = float(self.tello.get_speed_x())
                        msg.speed.y = float(self.tello.get_speed_y())
                        msg.speed.z = float(self.tello.get_speed_z())

                        msg.pitch = self.tello.get_pitch()
                        msg.roll = self.tello.get_roll()
                        msg.yaw = self.tello.get_yaw()

                        msg.barometer = int(self.tello.get_barometer())
                        msg.distance_tof = self.tello.get_distance_tof()

                        msg.fligth_time = self.tello.get_flight_time()

                        msg.battery = self.tello.get_battery()

                        msg.highest_temperature = self.tello.get_highest_temperature()
                        msg.lowest_temperature = self.tello.get_lowest_temperature()
                        msg.temperature = self.tello.get_temperature()

                        # msg.wifi_snr = self.tello.query_wifi_signal_noise_ratio()

                    self.pub_status.publish(msg)

                # Tello ID
                if self.pub_id.get_subscription_count() > 0:
                    msg = TelloID()
                    if not self.simulate:
                        msg.sdk_version = self.tello.query_sdk_version()
                        msg.serial_number = self.tello.query_serial_number()
                    self.pub_id.publish(msg)

                # Camera info
                if self.pub_camera_info.get_subscription_count() > 0:
                    msg = CameraInfo()
                    if not self.simulate:
                        msg.height = self.camera_info.image_height
                        msg.width = self.camera_info.image_width
                        msg.distortion_model = self.camera_info.distortion_model
                        msg.D = self.camera_info.distortion_coefficients
                        msg.K = self.camera_info.camera_matrix
                        msg.R = self.camera_info.rectification_matrix
                        msg.P = self.camera_info.projection_matrix
                    self.pub_camera_info.publish(msg)

                # Sleep
                time.sleep(rate)

        thread = threading.Thread(target=status_loop)
        thread.start()
        return thread


    # Start video capture thread.
    def start_video_capture(self, rate=1.0/30.0):
        # Enable tello stream
        if not self.simulate:
            self.tello.streamon()

        # OpenCV bridge
        self.bridge = CvBridge()

        def video_capture_thread():
            if not self.simulate:
                frame_read = self.tello.get_frame_read()

            while rclpy.ok():
                msg = Image()
                if not self.simulate:
                    # Get frame from drone
                    frame = frame_read.frame

                    # Publish opencv frame using CV bridge
                    msg = self.bridge.cv2_to_imgmsg(numpy.array(frame), 'rgb8')
                    msg.header.frame_id = self.tf_drone
                self.pub_image_raw.publish(msg)

                time.sleep(rate)


        # We need to run the recorder in a seperate thread, otherwise blocking options would prevent frames from getting added to the video
        thread = threading.Thread(target=video_capture_thread)
        thread.start()
        return thread

    # Terminate the code and shutdown node.
    def terminate(self, err):
        self.node.get_logger().error(str(err))
        if not self.simulate:
            self.tello.end()
        rclpy.shutdown()

    def stop_rc(self):
        self.node.get_logger().info('stop_rc')
        self.rc_control_enable = False
        time.sleep(0.050)

    def resume_rc(self):
        self.rc_control_enable = True
        self.node.get_logger().info('resume_rc')
        time.sleep(0.050)

    def control_owner_callback(self, msg):
        if msg.data.lower() == "commander":
            self.control_gamepad = False
            self.control_commander = True
            self.node.get_logger().info('Commander only control channel enabled')
        elif msg.data.lower() == "gamepad":
            self.control_commander = False
            self.control_gamepad = True
            self.node.get_logger().info('Gamepad only control channel enabled')
        else:
            self.control_commander = False
            self.control_gamepad = True
            self.node.get_logger().info('Gamepad only control channel enabled')

    # Stop all movement in the drone
    def cb_emergency_srv(self, request, response):
        self.node.get_logger().info('Emergency')
        try:
            if not self.simulate:
                self.tello.emergency()
            response.success = True
            response.message = "Emergency stop sent"
        except Exception as e:
            response.success = False
            response.message = f"Emergency failed: {e}"
        return response

    # Drone takeoff message control
    def cb_takeoff_srv(self, request, response):
        self.node.get_logger().info('Takeoff')
        try:
            if not self.simulate:
                self.tello.takeoff()
            response.success = True
            response.message = "Takeoff sent"
        except Exception as e:
            response.success = False
            response.message = f"Takeoff failed: {e}"
        return response

    # Land the drone message callback
    def cb_land_srv(self, request, response):
        self.node.get_logger().info('Land')
        try:
            if not self.simulate:
                self.tello.land()
            response.success = True
            response.message = "Land sent"
        except Exception as e:
            response.success = False
            response.message = f"Land failed: {e}"
        return response

    # Move the drone message callback
    def cb_movement_command(self, request, response):
        self.stop_rc()
        cmd = request.movement_type.lower().strip()
        val = request.distance
        speed = request.speed
        self.node.get_logger().info(f'Command: {cmd} {val}')

        if self.simulate:
            time.sleep(1)
            response.success = True
            response.message = f"OK. Simulated {cmd} complete."
            return response

        try: # not sure if we should block all other tello actions in the meantime
            self.tello.set_speed(speed)
            if cmd == "rotate_right":
                self.tello.rotate_clockwise(int(val))
            elif cmd == "rotate_left":
                self.tello.rotate_counter_clockwise(int(val))
            elif cmd == "move_forward":
                self.tello.move_forward(int(val))
            elif cmd == "move_back":
                self.tello.move_back(int(val))
            elif cmd == "move_left":
                self.tello.move_left(int(val))
            elif cmd == "move_right":
                self.tello.move_right(int(val))
            elif cmd == "move_up":
                self.tello.move_up(int(val))
            elif cmd == "move_down":
                self.tello.move_down(int(val))
            else:
                raise ValueError(f"Unknown command: {cmd}")

            response.success = True
            response.message = f"OK. {cmd} complete."
        except Exception as e:
            response.success = False
            response.message = str(e)
        self.resume_rc()
        return response

    # Control messages received use to control the drone "analogically"
    #
    # This method of controls allow for more precision in the drone control.
    #
    # Receives the linear and angular velocities to be applied from -100 to 100.
    def cb_control(self, msg):
        if not self.simulate and self.rc_control_enable and self.control_commander:
            self.tello.send_rc_control(int(msg.linear.y), int(msg.linear.x), int(msg.linear.z), int(msg.angular.z))

    def cb_control_gamepad(self, msg):
        if not self.simulate and self.rc_control_enable and self.control_gamepad:
            self.tello.send_rc_control(int(msg.linear.y), int(msg.linear.x), int(msg.linear.z), int(msg.angular.z))

    # Configure the wifi credential that should be used by the drone.
    #
    # The drone will be restarted after the credentials are changed.
    def cb_wifi_config(self, msg):
        if not self.simulate:
            self.tello.set_wifi_credentials(msg.ssid, msg.password)

    # Perform a drone flip in a direction specified.
    #
    # Directions can be "r" for right, "l" for left, "f" for forward or "b" for backward.
    def cb_flip(self, msg):
        if not self.simulate:
            self.tello.flip(msg.data)

# Convert a rotation from euler to quaternion.
def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

# Convert rotation from quaternion to euler.
def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('tello')
    drone = TelloNode(node)

    rclpy.spin(node)

    drone.cb_shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
