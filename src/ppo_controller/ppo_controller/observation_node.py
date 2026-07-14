import pathlib

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from ppo_interfaces.msg import PPOObservation
from std_msgs.msg import Float64
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

# Self-contained numpy/scipy port of the training-time Track/CubicSplineND
# classes -- no jax, no network download. Vendor track_utils.py alongside
# this file in your ROS package (e.g. <pkg>/<pkg>/track_utils.py).
from .track_utils import Track

PACKAGE_NAME = "ppo_controller"

# Centerline CSV filename, installed via setup.py's data_files into
# share/<package>/data/ (see the data/*.csv -> share/<pkg>/data glob entry).
DEFAULT_CENTERLINE_FILE = "RRMini_centerline.csv"
LOOKAHEAD_DISTANCES = [1.0, 2.0, 4.0]

class ObservationNode(Node):

    def __init__(self):
        super().__init__("observation_node")

        # Either an absolute path to the centerline CSV, or just a bare
        # filename (resolved against share/<package>/data/, i.e. wherever
        # setup.py's data_files installed data/*.csv to).
        self.declare_parameter("centerline_file", DEFAULT_CENTERLINE_FILE)
        self.declare_parameter('steering_angle_to_servo_gain', -1.2135)
        self.declare_parameter('steering_angle_to_servo_offset', 0.3835123)

        centerline_param = self.get_parameter("centerline_file").get_parameter_value().string_value
        centerline_path = pathlib.Path(centerline_param)
        if not centerline_path.is_absolute():
            share_dir = pathlib.Path(get_package_share_directory(PACKAGE_NAME))
            centerline_path = share_dir / "data" / centerline_path

        self.get_logger().info(f"Loading centerline from: {centerline_path}")
        self.track = Track.from_centerline_file(centerline_path)

        self.servo_gain = float(self.get_parameter('steering_angle_to_servo_gain').value)
        self.servo_offset = float(self.get_parameter('steering_angle_to_servo_offset').value)

        # running guess for the arclength search, mirrors self.s_guess
        # inside the Track class so downstream logic can inspect it if
        # needed (the current calc_arclength does a full search each call,
        # same as training, so this is informational rather than a warm
        # start).
        self.s_guess = 0.0

        # Force the very first published observation to be s=0, ey=0,
        # epsi=0 -- i.e. assume the car starts at the beginning of the
        # centerline, rather than running cartesian_to_frenet's nearest-
        # waypoint search on the first odom message. Subsequent messages
        # fall back to the normal (searched) Frenet conversion.
        self._first_odom = True

        self.current_steering = 0.0

        self.obs_pub = self.create_publisher(
            PPOObservation,
            "/ppo_observation",
            10,
        )

        self.servo_sub = self.create_subscription(
            Float64, "/sensors/servo_position_command", self.servo_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info("Created odom subscriber")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # --- odometry velocities are typically in the WORLD/odom frame ---
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # twist is expressed in child_frame_id (base_link), i.e. already body-frame
        vx_body = msg.twist.twist.linear.x
        vy_body = msg.twist.twist.linear.y
        yaw_rate = msg.twist.twist.angular.z  # rotation about z is frame-independent anyway

        beta = np.arctan2(vy_body, vx_body)

        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        # # Rotate world-frame velocity into the vehicle body frame.
        # # The dynamics model (and therefore the trained policy) expects
        # # v = body-frame longitudinal velocity and beta = body-frame slip
        # # angle, NOT atan2 of raw world-frame vx/vy.
        # c = np.cos(yaw)
        # s = np.sin(yaw)
        # vx_body = c * vx_world + s * vy_world
        # vy_body = -s * vx_world + c * vy_world
        # beta = np.arctan2(vy_body, vx_body)

        if self._first_odom:
            # Skip the nearest-waypoint search entirely for the first
            # message -- assume the car is placed at the start of the
            # centerline.
            s_val, ey, epsi = 0.0, 0.0, 0.0            
            self._first_odom = False
            s_val, ey, epsi = self.track.cartesian_to_frenet(
                x, y, yaw, s_guess=self.s_guess
            )
        else:
            # --- exact Frenet conversion, identical math to training env ---
            s_val, ey, epsi = self.track.cartesian_to_frenet(
                x, y, yaw, s_guess=self.s_guess
            )
        self.s_guess = s_val

        lookahead_curvatures = [
            float(self.track.curvature((s_val + d) % self.track.s_frame_max))
            for d in LOOKAHEAD_DISTANCES
        ]

        # progress along center line, cross track error, heading error,
        # steering angle, body-frame longitudinal velocity, yaw rate,
        # body-frame slip angle -- matches:
        #   fre_state = [s, ey, epsi]
        #   cart_state = cartesian_states[(2,3,5,6)] = [delta, v, psi_dot, beta]
        state = [
            float(s_val),
            float(ey),
            float(epsi),
            float(self.current_steering),
            float(vx_body),
            float(yaw_rate),
            float(beta),
            *lookahead_curvatures
        ]

        track_yaw = self.track.centerline.calc_yaw(s_val)
        self.get_logger().info(f"vehicle_yaw={yaw:.3f} track_yaw={track_yaw:.3f}")

        self.get_logger().info(
            f"""
        s={state[0]:.3f}
        ey={state[1]:.3f}
        epsi={state[2]:.3f}
        delta={state[3]:.3f}
        v={state[4]:.3f}
        yawrate={state[5]:.3f}
        beta={state[6]:.3f}
        """
        )

        obs_msg = PPOObservation()
        obs_msg.state = state
        self.obs_pub.publish(obs_msg)

    def servo_callback(self, msg: Float64):
        # msg.data is the actual commanded servo position in servo units [0,1],
        # i.e. what ackermann_to_vesc_node produced from steering_angle. Invert
        # that same transform to recover radians.
        servo_value = msg.data
        self.delta_est_measured = (servo_value - self.servo_offset) / self.servo_gain


def main(args=None):
    rclpy.init(args=args)

    node = ObservationNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
