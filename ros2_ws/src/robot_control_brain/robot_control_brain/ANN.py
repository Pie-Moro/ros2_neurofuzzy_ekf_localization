#!/usr/bin/env python3
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

# =============================================================================
# ARCHITETTURA RETE NEURALE
# =============================================================================
class TrajectoryANN(nn.Module):
    def __init__(self, input_dim=15, output_dim=2):
        super(TrajectoryANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, 15),
            nn.Tanh(),
            nn.Linear(15, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# =============================================================================
# NODO DI ADDESTRAMENTO ED ESECUZIONE IN TEMPO REALE
# =============================================================================
class OnlineTrainingNode(Node):
    def __init__(self):
        super().__init__('online_training_node')
        
        # Modello e Ottimizzatore
        self.model = TrajectoryANN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        
        # Flag per capire se la rete ha iniziato ad essere abbastanza matura
        self.is_trained_at_least_once = False
        self.training_in_progress = False
        self.lock = threading.Lock() # Evita che il training modifichi i pesi mentre l'inference li legge
        
        # Buffer storici per l'addestramento (es. teniamo gli ultimi 3000 campioni = 30 secondi a 100Hz)
        self.max_buffer_size = 3000
        self.input_buffer = []
        self.target_buffer = []
        
        # Ultimo dato ricevuto dai sensori (Sincronizzazione sample-and-hold)
        self.live_sensors = {'imu': None, 'gps': None, 'odom': None, 'kf1': None, 'kf2': None}
        self.live_target = None
        
        # Sottoscrizioni ai Sensori
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odometry/gps', self.gps_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/odometry/global', self.kf1_callback, 10)
        self.create_subscription(Odometry, '/odometry/global2', self.kf2_callback, 10)
        # Il target reale che la ANN deve imparare a seguire
        self.create_subscription(Odometry, '/odometry/fused', self.target_callback, 10)
        
        # Publisher per vedere i risultati su RViz
        self.ann_pub = self.create_publisher(Point, '/ann/trajectory', 10)
        self.target_pub = self.create_publisher(Point, '/ann/target_vis', 10)
        
        # Loop Principale a 100Hz (Inference e riempimento Buffer)
        self.create_timer(0.1, self.control_loop_callback)
        
        # Timer per far partire l'addestramento in background ogni 5 secondi
        self.create_timer(5.0, self.trigger_background_training)
        
        self.get_logger().info("Node activated. Putting datas into the buffer...")

    # =========================================================================
    # CALLBACK SENSORI & CONVERSIONE ANGOLI
    # =========================================================================
    def euler_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def imu_callback(self, msg):
        self.live_sensors['imu'] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity.z]
        
    def gps_callback(self, msg):
        self.live_sensors['gps'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
    def odom_callback(self, msg):
        p = msg.pose.pose.position
        yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        t = msg.twist.twist
        self.live_sensors['odom'] = [p.x, p.y, yaw, t.linear.x, t.linear.y, t.angular.z]
        
    def kf1_callback(self, msg):
        self.live_sensors['kf1'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
    def kf2_callback(self, msg):
        self.live_sensors['kf2'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
    def target_callback(self, msg):
        self.live_target = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    # =========================================================================
    # LOOP PRINCIPALE (100 Hz)
    # =========================================================================
    def control_loop_callback(self):
        # 1. CONTROLLO DI DEBUG: Chi è che non sta pubblicando?
        missing_sensors = [k for k, v in self.live_sensors.items() if v is None]
        
        if missing_sensors or self.live_target is None:
            # Stampa un avviso ogni 2 secondi senza intasare il terminale
            target_status = "MANCA" if self.live_target is None else "OK"
            self.get_logger().warn(
                f"[DEBUG] Sensori mancanti: {missing_sensors} | Stato Target: {target_status}", 
                throttle_duration_sec=2.0
            )
            return  # Ritorna finché non risolvi il sensore specifico

        # --- Da qui in poi il codice per il training rimane identico ---
        raw_input = np.concatenate([
            self.live_sensors['imu'], self.live_sensors['gps'], self.live_sensors['odom'],
            self.live_sensors['kf1'], self.live_sensors['kf2']
        ])
        raw_target = np.array(self.live_target)

        self.input_buffer.append(raw_input)
        self.target_buffer.append(raw_target)
        if len(self.input_buffer) > self.max_buffer_size:
            self.input_buffer.pop(0)
            self.target_buffer.pop(0)

        tgt_msg = Point(x=float(raw_target[0]), y=float(raw_target[1]), z=0.0)
        self.target_pub.publish(tgt_msg)

        if self.is_trained_at_least_once:
            inputs_mat = np.array(self.input_buffer)
            targets_mat = np.array(self.target_buffer)
            
            in_mean, in_std = inputs_mat.mean(axis=0), inputs_mat.std(axis=0) + 1e-6
            tgt_mean, tgt_std = targets_mat.mean(axis=0), targets_mat.std(axis=0) + 1e-6
            
            norm_input = (raw_input - in_mean) / in_std
            input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)

            with self.lock:
                with torch.no_grad():
                    norm_output = self.model(input_tensor).numpy().flatten()

            raw_output = (norm_output * tgt_std) + tgt_mean
            ann_msg = Point(x=float(raw_output[0]), y=float(raw_output[1]), z=0.0)
            self.ann_pub.publish(ann_msg)

    # =========================================================================
    # GESTIONE TRAINING IN BACKGROUND
    # =========================================================================
    def trigger_background_training(self):
        # Evita di lanciare un altro training se quello precedente è ancora in corso
        # E aspetta che ci siano almeno 500 campioni nel buffer per avere senso pratico
        if self.training_in_progress or len(self.input_buffer) < 500:
            return
        
        # Estraiamo una copia dei dati correnti per non bloccare il loop a 100Hz
        inputs_copy = np.array(self.input_buffer)
        targets_copy = np.array(self.target_buffer)
        
        # Avviamo l'addestramento su un Thread separato di Python
        self.training_in_progress = True
        threading.Thread(target=self.background_training_worker, args=(inputs_copy, targets_copy)).start()

    def background_training_worker(self, inputs, targets):
        try:
            # Calcolo parametri di normalizzazione (Z-score come mapstd di MATLAB)
            in_mean, in_std = inputs.mean(axis=0), inputs.std(axis=0) + 1e-6
            tgt_mean, tgt_std = targets.mean(axis=0), targets.std(axis=0) + 1e-6
            
            norm_inputs = (inputs - in_mean) / in_std
            norm_targets = (targets - tgt_mean) / targets.std(axis=0) # evitiamo divisioni per zero
            
            X = torch.tensor(norm_inputs, dtype=torch.float32)
            Y = torch.tensor(norm_targets, dtype=torch.float32)
            
            # Mini-addestramento locale (es. 20 epoche rapide per aggiornare i pesi sui dati recenti)
            self.model.train()
            for epoch in range(20):
                self.optimizer.zero_grad()
                predictions = self.model(X)
                loss = self.criterion(predictions, Y)
                loss.backward()
                self.optimizer.step()
            
            # Fase di scrittura sicura sui pesi globali
            with self.lock:
                self.model.eval()
                self.is_trained_at_least_once = True
                
            self.get_logger().info(f"Training completed. Loss: {loss.item():.4f}")
        except Exception as e:
            self.get_logger().error(f"Errore durante il training: {str(e)}")
        finally:
            self.training_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    node = OnlineTrainingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()