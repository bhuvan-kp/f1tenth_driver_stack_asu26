import rclpy
import torch
import torch.nn as nn
import numpy as np

from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from ppo_interfaces.msg import PPOObservation

from safetensors.torch import load_file

class Actor(nn.Module):

    def __init__(self, obs_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

class PolicyNode(Node):

    def __init__(self):

        super().__init__("policy_node")

        weights = load_file(
            "/f1tenth_ws/src/ppo_controller/models/RRMini_1_noscan_collision_progress+alive_velocity+steeringangle_10_v0_actor_params.safetensors"
        )

        obs_dim = weights["params,Dense_0,kernel"].shape[0]
        self.get_logger().info(f"Inferred observation dim from checkpoint: {obs_dim}")

        self.policy = Actor(obs_dim)

        self.policy.fc1.weight.data = weights["params,Dense_0,kernel"].T
        self.policy.fc1.bias.data = weights["params,Dense_0,bias"]

        self.policy.fc2.weight.data = weights["params,Dense_1,kernel"].T
        self.policy.fc2.bias.data = weights["params,Dense_1,bias"]

        self.policy.fc3.weight.data = weights["params,Dense_2,kernel"].T
        self.policy.fc3.bias.data = weights["params,Dense_2,bias"]

        self.policy.eval()

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            "/drive",
            10
        )

        self.obs_sub = self.create_subscription(
            PPOObservation,
            "/ppo_observation",
            self.obs_callback,
            10
        )

        self.get_logger().info("Policy loaded")

    def obs_callback(self, msg):

        obs = torch.tensor(
            [msg.state],
            dtype=torch.float32
        )

        with torch.no_grad():
            action = self.policy(obs)

        action = action.squeeze().numpy()

        # steering = float(
        #     np.clip(action[0], -0.34, 0.34)
        # )

        # steering = float(action[0])

        # speed = float(
        #    np.clip(action[1], 1.0, 5.0)
        # )

        steering = float(np.clip(action[0], -0.42, 0.42))

        speed = float(np.clip(action[1], -5.0, 20.0))

        drive_msg = AckermannDriveStamped()

        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed

        self.drive_pub.publish(drive_msg)

        self.get_logger().info(
            f"steer={steering:.3f}, speed={speed:.3f}"
        )

def main(args=None):

    rclpy.init(args=args)

    node = PolicyNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()

