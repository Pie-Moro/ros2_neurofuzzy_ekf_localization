#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node

# ROS 2 message types
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

# =============================================================================
# ANN ARCHITECTURE
# =============================================================================
class TrajectoryANN(nn.Module):
    def __init__(self, input_dim=15, output_dim=2):
        super(TrajectoryANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),  # Tansig in MATLAB corresponds to Hyperbolic Tangent
            nn.Linear(20, 15),
            nn.Tanh(),
            nn.Linear(15, output_dim)  # Purelin is linear activation (default)
        )
        
    def forward(self, x):
        return self.network(x)

# =============================================================================
# EXECUTION NODE (SUBSCRIBE & PUBLISH ONLY)
# =============================================================================
class TrajectoryNNNode(Node):
    def __init__(self):
        super().__init__('trajectory_nn_node')
        
        # Initialize Model and Normalization variables
        self.model = TrajectoryANN()
        self.input_mean = None
        self.input_std = None
        self.target_mean = None
        self.target_std = None
        
        # Load the pre-trained weights and normalization factors
        self.load_model()
        
        # Buffer for the latest received sensor values
        self.live_data = {
            'imu': None, 'gps': None, 'odom': None, 'kf1': None, 'kf2': None
        }
        
        # Setup Subscribers
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odometry/gps', self.gps_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/odometry/global', self.kf1_callback, 10)
        self.create_subscription(Odometry, '/odometry/global2', self.kf2_callback, 10)
        
        # Setup Publisher
        self.target_pub = self.create_publisher(Point, '/ann/target_trajectory', 10)
        
        # Timer at 100Hz to perform inference
        self.create_timer(0.01, self.inference_timer_callback)
        self.get_logger().info("Trajectory Execution Node started. Waiting for sensor data...")

    def load_model(self):
        model_path = 'trajectory_ann_model.pth'
        if os.path.exists(model_path):
            state = torch.load(model_path)
            self.model.load_state_dict(state['model_state'])
            self.input_mean = state['input_mean']
            self.input_std = state['input_std']
            self.target_mean = state['target_mean']
            self.target_std = state['target_std']
            self.model.eval()
            self.get_logger().info("Model loaded correctly. Ready for inference.")
        else:
            self.get_logger().error(f"Model file '{model_path}' not found! Inference will not run.")

    # =========================================================================
    # SENSOR CALLBACKS
    # =========================================================================
    def imu_callback(self, msg):
        self.live_data['imu'] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity.z]
        
    def gps_callback(self, msg):
        self.live_data['gps'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
    def odom_callback(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        t = msg.twist.twist
        self.live_data['odom'] = [p.x, p.y, o.z, t.linear.x, t.linear.y, t.angular.z]
        
    def kf1_callback(self, msg):
        self.live_data['kf1'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
    def kf2_callback(self, msg):
        self.live_data['kf2'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    # =========================================================================
    # INFERENCE & PUBLISHING LOOP
    # =========================================================================
    def inference_timer_callback(self):
        # Abort if the model isn't loaded or we are still missing data from some sensors
        if self.input_mean is None or any(v is None for v in self.live_data.values()):
            return
            
        # Construct the live input vector (15 elements)
        raw_input = np.concatenate([
            self.live_data['imu'], self.live_data['gps'], self.live_data['odom'],
            self.live_data['kf1'], self.live_data['kf2']
        ])
        
        # Normalize the live input using parameters calculated during training
        norm_input = (raw_input - self.input_mean) / self.input_std
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)  # Adds batch dimension
        
        # ANN Prediction
        with torch.no_grad():
            norm_output = self.model(input_tensor).numpy().flatten()
            
        # Denormalize the output (reverse map to std)
        raw_output = (norm_output * self.target_std) + self.target_mean
        
        # Construct and publish the Point message for the robot
        target_msg = Point()
        target_msg.x = float(raw_output[0])
        target_msg.y = float(raw_output[1])
        target_msg.z = 0.0
        
        self.target_pub.publish(target_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()