#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Tipi di messaggi ROS 2 dedotti dallo script MATLAB
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

# =============================================================================
# 🧠 ARCHITETTURA DELLA ANN (Identica a MATLAB [20, 15], Tansig -> TanH)
# =============================================================================
class TrajectoryANN(nn.Module):
    def __init__(self, input_dim=15, output_dim=2):
        super(TrajectoryANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),  # Tansig in MATLAB corrisponde alla Tangente Iperbolica
            nn.Linear(20, 15),
            nn.Tanh(),
            nn.Linear(15, output_dim) # Purelin è l'attivazione lineare (default)
        )
        
    def forward(self, x):
        return self.network(x)

# =============================================================================
# 🤖 NODO ROS 2
# =============================================================================
class TrajectoryNNNode(Node):
    def __init__(self):
        super().__init__('trajectory_nn_node')
        
        # Dichiarazione parametri per flessibilità
        self.declare_parameter('bag_path', 'Tentativo11')
        self.declare_parameter('mode', 'train') # Può essere 'train' o 'live'
        
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.bag_path = self.get_parameter('bag_path').get_parameter_value().string_value
        
        # Inizializzazione variabili del Modello e Normalizzazione (ps_in, ps_out)
        self.model = TrajectoryANN()
        self.input_mean, self.input_std = None, None
        self.target_mean, self.target_std = None, None
        
        if self.mode == 'train':
            self.get_logger().info(f"Avvio in modalità TRAINING. Lettura bag da: {self.bag_path}")
            self.train_from_bag()
        else:
            self.get_logger().info("Avvio in modalità LIVE. Caricamento modello e attivazione sottoscrizioni...")
            self.load_model()
            self.setup_live_subscribers()

    # =========================================================================
    # 🔵 FASE OFFLINE: LETTURA BAG & TRAINING
    # =========================================================================
    def train_from_bag(self):
        import rosbag2_py
        
        # Dizionari per accumulare i dati grezzi e i loro timestamp
        data_store = {
            '/imu/data': {'t': [], 'val': []},
            '/odom': {'t': [], 'val': []},
            '/odometry/global': {'t': [], 'val': []},
            '/odometry/global2': {'t': [], 'val': []},
            '/odometry/gps': {'t': [], 'val': []},
            '/odometry/fused': {'t': [], 'val': []}
        }
        
        # Configurazione lettore rosbag2
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        )
        
        try:
            reader.open(storage_options, converter_options)
        except Exception as e:
            self.get_logger().error(f"Impossibile aprire la bag: {str(e)}")
            return

        # Leggiamo tutti i tipi di messaggi registrati nella bag per poterli deserializzare
        topic_types = reader.get_all_topics_and_types()
        type_map = {msg.name: msg.type for msg in topic_types}

        self.get_logger().info("Estrazione dati in corso (massima velocità)...")
        
        while reader.has_next():
            (topic, data, t_nanoseconds) = reader.read_next()
            if topic in data_store:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                
                # Conversione timestamp in secondi (punti decimali inclusi)
                t_sec = t_nanoseconds / 1e9
                data_store[topic]['t'].append(t_sec)
                
                # Estrazione specifica in base al tipo di topic (come facevi in MATLAB)
                if topic == '/imu/data':
                    data_store[topic]['val'].append([
                        msg.linear_acceleration.x, 
                        msg.linear_acceleration.y, 
                        msg.angular_velocity.z
                    ])
                elif topic in ['/odom', '/odometry/global', '/odometry/global2', '/odometry/gps', '/odometry/fused']:
                    # Struttura comune Odometry per la posizione
                    pos = msg.pose.pose.position
                    if topic == '/odom':
                        ori = msg.pose.pose.orientation
                        twist = msg.twist.twist
                        data_store[topic]['val'].append([
                            pos.x, pos.y, ori.z, twist.linear.x, twist.linear.y, twist.angular.z
                        ])
                    elif topic == '/odometry/fused': # Target Output
                        data_store[topic]['val'].append([pos.x, pos.y])
                    else: # global, global2, gps
                        data_store[topic]['val'].append([pos.x, pos.y])

        self.get_logger().info("Dati estratti. Allineamento temporale (100 Hz)...")
        
        # Trasformiamo in array numpy
        for topic in data_store:
            data_store[topic]['t'] = np.array(data_store[topic]['t'])
            data_store[topic]['val'] = np.array(data_store[topic]['val'])

        # Calcolo griglia temporale comune (min dei max, max dei min)
        t_start = max([data_store[top]['t'].min() for top in data_store])
        t_end = min([data_store[top]['t'].max() for top in data_store])
        
        fs = 100
        t_common = np.arange(t_start, t_end, 1/fs)
        
        # Interpolazione dei dati sulla griglia comune
        # IMU, Odom, Global, Global2, Target -> Lineare
        # GPS -> Previous (Zero-Order Hold per bassa frequenza)
        imu_interp = interp1d(data_store['/imu/data']['t'], data_store['/imu/data']['val'], axis=0, kind='linear')(t_common)
        gps_interp = interp1d(data_store['/odometry/gps']['t'], data_store['/odometry/gps']['val'], axis=0, kind='previous')(t_common)
        odom_interp = interp1d(data_store['/odom']['t'], data_store['/odom']['val'], axis=0, kind='linear')(t_common)
        kf1_interp = interp1d(data_store['/odometry/global']['t'], data_store['/odometry/global']['val'], axis=0, kind='linear')(t_common)
        kf2_interp = interp1d(data_store['/odometry/global2']['t'], data_store['/odometry/global2']['val'], axis=0, kind='linear')(t_common)
        tgt_interp = interp1d(data_store['/odometry/fused']['t'], data_store['/odometry/fused']['val'], axis=0, kind='linear')(t_common)
        
        # Composizione Input (15 colonne) e Target (2 colonne)
        X = np.hstack([imu_interp, gps_interp, odom_interp, kf1_interp, kf2_interp])
        Y = tgt_interp
        
        # Normalizzazione (Equivalente a mapstd di MATLAB)
        self.input_mean, self.input_std = X.mean(axis=0), X.std(axis=0)
        self.target_mean, self.target_std = Y.mean(axis=0), Y.std(axis=0)
        # Evitiamo divisioni per zero se qualche sensore ha varianza nulla
        self.input_std[self.input_std == 0] = 1.0
        self.target_std[self.target_std == 0] = 1.0
        
        X_norm = (X - self.input_mean) / self.input_std
        Y_norm = (Y - self.target_mean) / self.target_std
        
        # Conversione in Tensori PyTorch
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)
        
        # Divisione in Train (70%) e Validation (30% semplificato rispetto a MATLAB)
        dataset_size = len(X_tensor)
        train_size = int(0.85 * dataset_size)
        
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        Y_train, Y_val = Y_tensor[:train_size], Y_tensor[train_size:]
        
        # Allineamento Addestramento (Simile ai parametri MATLAB)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=0.01) # weight_decay = regolarizzazione
        
        self.get_logger().info("Inizio addestramento della ANN con PyTorch...")
        
        for epoch in range(1000):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, Y_val)
                    rmse_val = torch.sqrt(val_loss).item()
                self.get_logger().info(f"Epoca [{epoch+1}/1000] - Loss Train: {loss.item():.4f} - RMSE Val (Norm): {rmse_val:.4f}")
                
        self.get_logger().info("Training completato con successo! Salvataggio modello in corso...")
        self.save_model()

    def save_model(self):
        state = {
            'model_state': self.model.state_dict(),
            'input_mean': self.input_mean, 'input_std': self.input_std,
            'target_mean': self.target_mean, 'target_std': self.target_std
        }
        torch.save(state, 'trajectory_ann_model.pth')
        self.get_logger().info("Modello 'trajectory_ann_model.pth' salvato.")

    def load_model(self):
        if os.path.exists('trajectory_ann_model.pth'):
            state = torch.load('trajectory_ann_model.pth')
            self.model.load_state_dict(state['model_state'])
            self.input_mean = state['input_mean']
            self.input_std = state['input_std']
            self.target_mean = state['target_mean']
            self.target_std = state['target_std']
            self.model.eval()
            self.get_logger().info("Modello caricato correttamente da file.")
        else:
            self.get_logger().error("File modello non trovato! Impossibile andare in modalità Live.")

    # =========================================================================
    # 🟢 FASE ONLINE: RICEZIONE LIVE SENSORI & PREDIZIONE TRAGUARDO
    # =========================================================================
    def setup_live_subscribers(self):
        # Buffer per l'ultimo valore ricevuto da ogni sensore
        self.live_data = {
            'imu': None, 'gps': None, 'odom': None, 'kf1': None, 'kf2': None
        }
        
        # Sottoscrizioni ai 5 topic dei sensori live
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odometry/gps', self.gps_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/odometry/global', self.kf1_callback, 10)
        self.create_subscription(Odometry, '/odometry/global2', self.kf2_callback, 10)
        
        # Publisher per l'output della traiettoria calcolata dalla ANN
        self.target_pub = self.create_publisher(Point, '/ann/target_trajectory', 10)
        
        # Timer a 100Hz per fare inferenza (identico alla frequenza fs dello script)
        self.create_timer(0.01, self.inference_timer_callback)

    # Callback di archiviazione dati real-time
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

    def inference_timer_callback(self):
        # Controlliamo di aver ricevuto almeno un messaggio da ogni sensore per poter predire
        if any(v is None for v in self.live_data.values()):
            return
            
        # Costruiamo il vettore di input live (15 elementi)
        raw_input = np.concatenate([
            self.live_data['imu'], self.live_data['gps'], self.live_data['odom'],
            self.live_data['kf1'], self.live_data['kf2']
        ])
        
        # Normalizzazione dell'input live usando i parametri calcolati nel training
        norm_input = (raw_input - self.input_mean) / self.input_std
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0) # Aggiunge dimensione batch
        
        # Calcolo predizione ANN
        with torch.no_grad():
            norm_output = self.model(input_tensor).numpy().flatten()
            
        # Denormalizzazione dell'output (reverse mapstd)
        raw_output = (norm_output * self.target_std) + self.target_mean
        
        # Pubblicazione del Target per il robot
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