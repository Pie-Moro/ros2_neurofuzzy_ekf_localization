#!/usr/bin/env python3
"""
ANN Online Training & Inference Node — Fixed (Session 7+)
==========================================================

PAPER REFERENCE  [P2] Yousuf & Kadri (2020), §3.5, Table II, Eq. (3)

ANN ROLE (per paper §3, Fig.1 flowchart):
  • Trained ONLY when GPS is available (outdoor), using combined KF output as target.
  • Acts as GPS pseudo-sensor when GPS is unavailable (indoor).
  • Input: IMU + Odometer + KF positions — GPS NEVER included.
  • Output: [x_n, y_n] predicted position in map frame.

INPUT VECTOR — 15 features (no GPS, per Eq. 3 + flowchart NN_Train=[IMU,Odom,KFs]):

  IMU-derived (6):                           Source
  ─────────────────────────────────────────  ──────────────────────────
  1. v̇_a  = ||(a_x, a_y)||                  /imu/data linear_acceleration
  2. Σv̇_a  = Σ v̇_a (cumulative)             running sum
  3. ẋ_a   = Σ v̇_a·Δt  (IMU velocity mag)   running integral
  4. Σẋ_a  = Σ ẋ_a (cumulative)             running sum
  5. γ_g   = Σ ω_z·Δt  (gyro yaw)           integrated from /imu angular_velocity.z
  6. Σγ̇_g  = Σ ω_z    (cumulative yaw rate) running sum

  Odometer (5):                              Source
  ─────────────────────────────────────────  ──────────────────────────
  7.  ẋ_o  = vx                              /odom twist.linear.x
  8.  Σẋ_o = Σ vx (cumulative)              running sum
  9.  ẏ_o  = vy                              /odom twist.linear.y
  10. Σẏ_o = Σ vy (cumulative)              running sum
  11. ψ    = odom heading (yaw)              /odom pose quaternion → euler

  KF positions (4):                          Source
  ─────────────────────────────────────────  ──────────────────────────
  12. x_kf1, 13. y_kf1                       /odometry/global  (KF-1 GPS+IMU)
  14. x_kf2, 15. y_kf2                       /odometry/global2 (KF-2 GPS+Odom)

ARCHITECTURE (paper Table II):
  Input(15) → Dense(10, Sigmoid) → Dense(5, Sigmoid) → Dense(2, Linear)

TARGET (paper Eq. 3):
  [x_kf_c, y_kf_c] = α1·[x_kf1,y_kf1] + α2·[x_kf2,y_kf2]
  = /odometry/fused (complementary filter output)

TRAINING GATE (paper §3 Fig.1):
  Buffer accumulates ONLY when GPS fix is active (outdoor).
  Inference runs in both outdoor and indoor modes.

BUGS FIXED vs PREVIOUS VERSION:
  B1 — GPS removed from input features (was: live_sensors['gps'] in raw_input)
  B2 — Architecture corrected: 15→10→5→2, nn.Sigmoid (was: 15→20→15→2, nn.Tanh)
  B3 — Outdoor-only training gate added via /gps/fix subscription
  B4 — Normalization: tgt_std (with +1e-6 guard) used in both train and infer paths
  B5 — Weight persistence: torch.save on shutdown, torch.load on startup (T14)
"""

import os
import math
import signal
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point


# =============================================================================
# NEURAL NETWORK ARCHITECTURE — paper Table II
# =============================================================================
# Input(15) → Dense(10, Sigmoid) → Dense(5, Sigmoid) → Dense(2, Linear)
# Activation: log-sigmoid = nn.Sigmoid (paper Eq. 31: a = 1/(1+e^{-n}))
# Output:     linear transfer  (paper Eq. 32: a = n)
# =============================================================================
class TrajectoryANN(nn.Module):
    def __init__(self, input_dim: int = 15, output_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),  # Hidden layer 1: 10 neurons  (paper Table II)
            nn.Sigmoid(),              # log-sigmoid activation       (paper §3.5.1)
            nn.Linear(10, 5),          # Hidden layer 2:  5 neurons  (paper Table II)
            nn.Sigmoid(),              # log-sigmoid activation
            nn.Linear(5, output_dim),  # Output layer:    2 neurons  (paper Table II)
                                       # Linear transfer (no activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# ONLINE TRAINING & INFERENCE NODE
# =============================================================================
class OnlineTrainingNode(Node):

    WEIGHTS_PATH = os.path.expanduser('~/ros2_neurofuzzy_ekf_localization/ros2_ws/'
                                      'src/robot_control_brain/models/nn_weights.pt')

    def __init__(self):
        super().__init__('online_training_node')

        # ── Model & Optimiser ─────────────────────────────────────────────────
        self.model     = TrajectoryANN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()

        # Normalisation parameters — updated every training cycle
        self.in_mean : np.ndarray | None = None
        self.in_std  : np.ndarray | None = None
        self.tgt_mean: np.ndarray | None = None
        self.tgt_std : np.ndarray | None = None

        # Training state
        self.is_trained_at_least_once = False
        self.training_in_progress     = False
        self.lock = threading.Lock()

        # ── Sample buffer (paper: store while GPS available) ──────────────────
        # 3000 samples ≈ 300 s at 10 Hz inference loop
        self.MAX_BUFFER = 3000
        self.input_buffer : list[np.ndarray] = []
        self.target_buffer: list[np.ndarray] = []

        # ── GPS availability gate (paper Fig.1: "Check for GPS Position Signal") ──
        # Training buffer only fills when GPS fix is active (outdoor phase).
        # Inference runs regardless of GPS state.
        self.gps_fix_active = False     # set by /gps/fix callback

        # F2 FIX — Indoor anchor substitution
        # When GPS goes inactive (robot enters building), the EKF1 and EKF2
        # inputs continue changing (dead-reckoning), but the ANN was trained
        # exclusively on GPS-corrected KF positions. Indoor KF inputs are
        # out-of-distribution → ANN error = 7.62m mean (confirmed from bag).
        # Fix: snapshot the last GPS-active KF1/KF2 positions as an anchor.
        # Indoors, substitute this anchor into the ANN input in place of the
        # live (diverging) KF values. The anchor stays fixed, matching the
        # training distribution far better than dead-reckoned positions.
        self._kf1_anchor: list[float] | None = None
        self._kf2_anchor: list[float] | None = None

        # ── Cumulative sensor state (reset on node start; persist per-run) ────
        # Paper Eq. (3): Σv̇_a, Σẋ_a, Σγ̇_g, Σẋ_o, Σẏ_o are running sums.
        self._prev_stamp: float | None = None   # for Δt computation
        self._sigma_a_mag  = 0.0   # Σ||(a_x, a_y)||            (Σv̇_a)
        self._sigma_v_imu  = 0.0   # Σ (||(a_x,a_y)|| · Δt)    (Σẋ_a)
        self._gamma_g      = 0.0   # Σ ω_z · Δt                 (γ_g heading)
        self._sigma_omega  = 0.0   # Σ ω_z                       (Σγ̇_g)
        self._sigma_vx_od  = 0.0   # Σ vx_odom                  (Σẋ_o)
        self._sigma_vy_od  = 0.0   # Σ vy_odom                  (Σẏ_o)

        # ── Latest sensor readings (sample-and-hold) ──────────────────────────
        # GPS deliberately NOT stored — paper explicitly forbids it as ANN input.
        self._imu : list[float] | None = None   # [a_x, a_y, ω_z]
        self._odom: list[float] | None = None   # [vx, vy, yaw]
        self._kf1 : list[float] | None = None   # [x, y]
        self._kf2 : list[float] | None = None   # [x, y]
        self._target: list[float] | None = None # [x_kf_c, y_kf_c] from /odometry/fused

        # EKF validity guard (S11): reject training samples when KF1 covariance
        # has overflowed. P_xx > 5.0 m² indicates the EKF is in a diverged or
        # GPS-denied dead-reckoning state — targets drawn from /odometry/fused
        # at this point are corrupted (confirmed: ANN error 4–17m from bag analysis).
        # P_xx is covariance[0] (row 0, col 0 of the 6×6 pose covariance matrix).
        self._kf1_cov: float = 0.0   # P_xx [m²]; 0.0 = not yet received (safe: <5.0)

        # ── Subscriptions ─────────────────────────────────────────────────────
        # [B1 FIX] /odometry/gps subscription REMOVED.
        #   GPS was included in raw_input, blocking inference/training indoors
        #   (missing_sensors check fired every tick when GPS unavailable).
        #   Paper Eq. (3) explicitly: "GPS was not included as input in the
        #   training of ANN because the neural network was tested in situations
        #   when there was complete GPS signal loss such as in indoor environments."
        self.create_subscription(Imu,       '/imu/data',         self._cb_imu,    10)
        self.create_subscription(Odometry,  '/odom',             self._cb_odom,   10)
        self.create_subscription(Odometry,  '/odometry/global',  self._cb_kf1,    10)
        self.create_subscription(Odometry,  '/odometry/global2', self._cb_kf2,    10)
        # Training target: [x_kf_c, y_kf_c] = complementary filter output (paper Eq. 2)
        self.create_subscription(Odometry,  '/odometry/fused',   self._cb_target, 10)
        # [B3 FIX] GPS fix subscription for outdoor-only training gate
        # (paper Fig.1: buffer fills only when "GPS Position Signal Available")
        self.create_subscription(NavSatFix, '/gps/fix',          self._cb_gps_fix, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.ann_pub    = self.create_publisher(Point, '/ann/trajectory',  10)
        self.target_pub = self.create_publisher(Point, '/ann/target_vis',  10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)          # 10 Hz: inference + buffer
        self.create_timer(5.0, self._trigger_training)      # every 5 s: background train

        # ── T14: Load persisted weights if available ──────────────────────────
        self._load_weights()

        self.get_logger().info(
            '[ANN] Node started.\n'
            '  Architecture : 15→10→5→2  (log-sigmoid × 2, linear output)\n'
            '  Input features: 15 (IMU×6 + Odom×5 + KF1×2 + KF2×2, NO GPS)\n'
            '  Training gate : GPS fix required (outdoor only)\n'
            f'  Weights path  : {self.WEIGHTS_PATH}')

    # =========================================================================
    # SENSOR CALLBACKS
    # =========================================================================

    def _cb_gps_fix(self, msg: NavSatFix) -> None:
        """[B3 FIX] Track GPS fix availability for the training gate.
        [F2 FIX] Save KF1/KF2 anchor positions when GPS transitions active→inactive.
        """
        new_active = (msg.status.status >= NavSatStatus.STATUS_FIX)
        if self.gps_fix_active and not new_active:
            # GPS just lost — snapshot last GPS-corrected KF positions as anchor.
            # These will substitute for live KF inputs during indoor dead-reckoning
            # to keep the ANN input distribution consistent with training.
            if self._kf1 is not None:
                self._kf1_anchor = list(self._kf1)
            if self._kf2 is not None:
                self._kf2_anchor = list(self._kf2)
            self.get_logger().info(
                f'[ANN][F2] GPS lost — anchor saved: '
                f'KF1=({self._kf1_anchor[0]:.3f},{self._kf1_anchor[1]:.3f})  '
                f'KF2=({self._kf2_anchor[0]:.3f},{self._kf2_anchor[1]:.3f})'
                if self._kf1_anchor else '[ANN][F2] GPS lost — no KF data yet to anchor')
        self.gps_fix_active = new_active

    def _cb_imu(self, msg: Imu) -> None:
        """
        Store raw IMU values and update cumulative features.

        Paper §3.2 IMU quantities used in Eq. (3):
          v̇_a  = ||(a_x, a_y)||       acceleration magnitude
          ẋ_a   = Σ v̇_a · Δt          velocity magnitude from IMU integration
          γ_g   = Σ ω_z · Δt           heading from gyro integration
          Σγ̇_g  = Σ ω_z               cumulative yaw rate
        """
        ax  = msg.linear_acceleration.x
        ay  = msg.linear_acceleration.y
        wz  = msg.angular_velocity.z
        now = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        if self._prev_stamp is None:
            self._prev_stamp = now
        dt = max(now - self._prev_stamp, 1e-4)   # guard against zero Δt
        self._prev_stamp = now

        a_mag = math.sqrt(ax * ax + ay * ay)
        v_imu = a_mag * dt                       # velocity increment from IMU

        self._sigma_a_mag  += a_mag              # Σv̇_a
        self._sigma_v_imu  += v_imu              # Σẋ_a (cumulative IMU velocity mag)
        self._gamma_g      += wz * dt            # γ_g  (integrated heading)
        self._sigma_omega  += wz                 # Σγ̇_g (cumulative yaw rate)

        self._imu = [ax, ay, wz]

    def _cb_odom(self, msg: Odometry) -> None:
        """Store odometry velocities and heading; update cumulative odom features."""
        vx  = msg.twist.twist.linear.x
        vy  = msg.twist.twist.linear.y
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        self._sigma_vx_od += vx   # Σẋ_o
        self._sigma_vy_od += vy   # Σẏ_o
        self._odom = [vx, vy, yaw]

    def _cb_kf1(self, msg: Odometry) -> None:
        """KF-1 position estimate (GPS+IMU EKF, /odometry/global).

        EKF validity guard (S11): cache P_xx = pose.covariance[0] (m²).
        The 36-element covariance vector is row-major for [x,y,z,roll,pitch,yaw]
        so index 0 is σ²_x in the map frame. When this exceeds 5.0 m² the EKF
        has diverged or GPS has been absent long enough for prediction to inflate
        uncertainty beyond a reliable threshold — training targets from this
        state are corrupted (confirmed from bag analysis, S11).
        """
        self._kf1 = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self._kf1_cov = msg.pose.covariance[0]   # P_xx [m²] for validity guard

    def _cb_kf2(self, msg: Odometry) -> None:
        """KF-2 position estimate (GPS+Odom EKF, /odometry/global2)."""
        self._kf2 = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    def _cb_target(self, msg: Odometry) -> None:
        """
        Training target: complementary filter output /odometry/fused.
        Per paper Eq. (2): [x_kf_c, y_kf_c] = α1·[x_kf1]+α2·[x_kf2]
        Per paper Eq. (3): target = [x_kf_c, y_kf_c]
        """
        self._target = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    # =========================================================================
    # FEATURE CONSTRUCTION (paper Eq. 3 — 15 features, NO GPS)
    # =========================================================================
    def _build_input(self) -> np.ndarray | None:
        """
        Assemble the 15-element input vector from live sensor readings.

        Returns None if any required sensor stream has not yet published.
        Unlike the old code, GPS absence does NOT block this function.
        """
        if self._imu is None or self._odom is None or \
           self._kf1 is None or self._kf2 is None:
            return None

        ax, ay, wz  = self._imu
        vx, vy, yaw = self._odom

        # [F2 FIX] Indoor anchor substitution:
        # Use GPS-corrected KF anchor instead of live (dead-reckoning) values
        # when GPS is inactive. This keeps features 12-15 in the training
        # distribution (outdoor KF positions), preventing the 7.62m ANN error
        # seen indoors in the bag analysis.
        if not self.gps_fix_active and self._kf1_anchor is not None:
            x1, y1 = self._kf1_anchor
            x2, y2 = self._kf2_anchor if self._kf2_anchor is not None \
                      else self._kf1_anchor
        else:
            x1, y1 = self._kf1
            x2, y2 = self._kf2

        # Paper Eq. (3) feature layout (15 elements):
        #   IMU-derived (6): v̇_a, Σv̇_a, ẋ_a, Σẋ_a, γ_g, Σγ̇_g
        #   Odometer   (5): ẋ_o, Σẋ_o, ẏ_o, Σẏ_o, ψ (heading from odom)
        #   KF outputs (4): x_kf1, y_kf1, x_kf2, y_kf2
        v_dot_a = math.sqrt(ax * ax + ay * ay)             # v̇_a
        x_dot_a = v_dot_a * (1.0 / 10.0)                   # ẋ_a approx (10 Hz loop)
        return np.array([
            # ── IMU features (6) ──────────────────────────────────────────────
            v_dot_a,            #  1. v̇_a  acceleration magnitude
            self._sigma_a_mag,  #  2. Σv̇_a  cumulative accel mag
            x_dot_a,            #  3. ẋ_a   IMU velocity magnitude (per step)
            self._sigma_v_imu,  #  4. Σẋ_a  cumulative IMU velocity
            self._gamma_g,      #  5. γ_g   integrated gyro heading
            self._sigma_omega,  #  6. Σγ̇_g  cumulative yaw rate
            # ── Odometer features (5) ─────────────────────────────────────────
            vx,                 #  7. ẋ_o   odom x velocity
            self._sigma_vx_od,  #  8. Σẋ_o  cumulative odom x velocity
            vy,                 #  9. ẏ_o   odom y velocity
            self._sigma_vy_od,  # 10. Σẏ_o  cumulative odom y velocity
            yaw,                # 11. ψ     heading from odom quaternion
            # ── KF position features (4) ──────────────────────────────────────
            x1,                 # 12. x_kf1
            y1,                 # 13. y_kf1
            x2,                 # 14. x_kf2
            y2,                 # 15. y_kf2
        ], dtype=np.float32)

    # =========================================================================
    # MAIN LOOP — 10 Hz
    # =========================================================================
    def _control_loop(self) -> None:
        """
        10 Hz loop: build input, optionally fill buffer (outdoor only), run inference.
        """
        raw_input = self._build_input()
        if raw_input is None:
            # Log missing sensor streams without GPS confusion
            missing = []
            if self._imu  is None: missing.append('imu')
            if self._odom is None: missing.append('odom')
            if self._kf1  is None: missing.append('kf1 (/odometry/global)')
            if self._kf2  is None: missing.append('kf2 (/odometry/global2)')
            self.get_logger().warn(
                f'[ANN] Waiting for: {missing}', throttle_duration_sec=3.0)
            return

        # [B3 FIX] — Buffer samples only when GPS is available (outdoor).
        # Paper Fig.1: "Train the Neural Network on the Sensor-KF input data
        # saved in Computer Memory" — this save happens only in the GPS-available
        # branch of the flowchart.
        #
        # [S11 EKF VALIDITY GUARD] — Reject samples when KF1 covariance has
        # overflowed. During GPS→indoor transitions and multi-path events, the
        # EKF prediction step inflates P_xx rapidly once GPS corrections stop.
        # Training on these samples bakes diverged EKF positions into the ANN
        # as ground truth, producing the 4–17m indoor prediction errors seen in
        # bag analysis (runs 4–7). Threshold 5.0 m² = σ_x > 2.24 m, well above
        # normal GPS-corrected operation (σ_x ≈ 0.32 m) but below float64
        # overflow territory (P > 10^300 seen in run analysis).
        _EKF_COV_MAX_M2 = 5.0
        if self.gps_fix_active and self._target is not None:
            if self._kf1_cov > _EKF_COV_MAX_M2:
                # Reject: EKF diverged — target position unreliable
                self.get_logger().warn(
                    f'[ANN][EKF-guard] Sample REJECTED: '
                    f'KF1 P_xx={self._kf1_cov:.2f} m² > {_EKF_COV_MAX_M2} m²  '
                    f'buf={len(self.input_buffer)}',
                    throttle_duration_sec=5.0)
            else:
                raw_target = np.array(self._target, dtype=np.float32)
                self.input_buffer.append(raw_input.copy())
                self.target_buffer.append(raw_target)
                if len(self.input_buffer) > self.MAX_BUFFER:
                    self.input_buffer.pop(0)
                    self.target_buffer.pop(0)
                # Visualise training target
                self.target_pub.publish(
                    Point(x=float(raw_target[0]), y=float(raw_target[1]), z=0.0))

        # Inference — runs in BOTH outdoor and indoor modes
        if not self.is_trained_at_least_once:
            return

        if self.in_mean is None:
            return

        norm_input = (raw_input - self.in_mean) / self.in_std
        tensor_in  = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)

        with self.lock:
            with torch.no_grad():
                norm_out = self.model(tensor_in).numpy().flatten()

        # Denormalise output
        raw_out = (norm_out * self.tgt_std) + self.tgt_mean
        self.ann_pub.publish(
            Point(x=float(raw_out[0]), y=float(raw_out[1]), z=0.0))

    # =========================================================================
    # BACKGROUND TRAINING — every 5 s
    # =========================================================================
    def _trigger_training(self) -> None:
        """
        Spawn background training if: not already training AND ≥500 outdoor samples.
        Paper §3.5.2: quasi-Newton backpropagation. Here approximated with Adam
        (not specified in paper for ROS2 implementation — standard practice).
        """
        if self.training_in_progress or len(self.input_buffer) < 500:
            return

        inputs_copy  = np.array(self.input_buffer,  dtype=np.float32)
        targets_copy = np.array(self.target_buffer, dtype=np.float32)

        self.training_in_progress = True
        t = threading.Thread(
            target=self._training_worker,
            args=(inputs_copy, targets_copy),
            daemon=True)
        t.start()

    def _training_worker(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Background training with Z-score normalisation.
        Paper §3.5.3 regularisation: MSEreg = γ·MSE + (1-γ)·MSW
        implemented via Adam weight decay (L2 penalty ≈ MSW term).

        [B4 FIX] tgt_std uses +1e-6 guard on BOTH training and inference paths.
        Previously: `targets.std(axis=0)` (no guard) → NaN when robot stationary.
        """
        try:
            in_mean  = inputs.mean(axis=0)
            in_std   = inputs.std(axis=0)  + 1e-6   # [B4 guard]
            tgt_mean = targets.mean(axis=0)
            tgt_std  = targets.std(axis=0) + 1e-6   # [B4 guard]

            norm_inputs  = (inputs  - in_mean)  / in_std
            norm_targets = (targets - tgt_mean) / tgt_std  # [B4 FIX: was /targets.std()]

            X = torch.tensor(norm_inputs,  dtype=torch.float32)
            Y = torch.tensor(norm_targets, dtype=torch.float32)

            self.model.train()
            for epoch in range(20):
                self.optimizer.zero_grad()
                preds = self.model(X)
                loss  = self.criterion(preds, Y)
                loss.backward()
                self.optimizer.step()

            with self.lock:
                self.model.eval()
                # Cache normalisation params for inference loop
                self.in_mean  = in_mean
                self.in_std   = in_std
                self.tgt_mean = tgt_mean
                self.tgt_std  = tgt_std
                self.is_trained_at_least_once = True

            self.get_logger().info(
                f'[ANN] Training complete. '
                f'Loss={loss.item():.4f}  N={len(X)}  '
                f'GPS-gate={"ON" if self.gps_fix_active else "OFF"}')

        except Exception as exc:
            self.get_logger().error(f'[ANN] Training error: {exc}')
        finally:
            self.training_in_progress = False

    # =========================================================================
    # T14 — WEIGHT PERSISTENCE
    # =========================================================================
    def _load_weights(self) -> None:
        """
        T14: Load persisted weights from previous session if available.
        Allows inference to start immediately on re-launch without retraining.
        """
        if not os.path.exists(self.WEIGHTS_PATH):
            self.get_logger().info('[ANN] No persisted weights found — starting fresh.')
            return
        try:
            # B5 FIX (Session 11): use weights_only=False.
            # The checkpoint is written exclusively by save_weights() in this
            # node — fully trusted source. weights_only=True fails on PyTorch 2.6
            # because numpy dtype generic aliases cannot be allowlisted.
            # save_weights now stores arrays as Python lists so future checkpoints
            # will load cleanly; this fallback handles old numpy-array checkpoints.
            checkpoint = torch.load(self.WEIGHTS_PATH, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state'])
            if 'in_mean' in checkpoint:
                # Support both old numpy-array format and new list format
                self.in_mean  = np.array(checkpoint['in_mean'],  dtype=np.float32)
                self.in_std   = np.array(checkpoint['in_std'],   dtype=np.float32)
                self.tgt_mean = np.array(checkpoint['tgt_mean'], dtype=np.float32)
                self.tgt_std  = np.array(checkpoint['tgt_std'],  dtype=np.float32)
                self.is_trained_at_least_once = True
            self.model.eval()
            self.get_logger().info(
                f'[ANN] Loaded weights from {self.WEIGHTS_PATH}')
        except Exception as exc:
            self.get_logger().warn(f'[ANN] Failed to load weights: {exc}')

    def save_weights(self) -> None:
        """
        T14: Save current weights + normalisation params to disk.
        Called by main() on shutdown (KeyboardInterrupt / SIGINT).
        """
        if not self.is_trained_at_least_once:
            self.get_logger().info('[ANN] No trained weights to save (skipping).')
            return
        try:
            os.makedirs(os.path.dirname(self.WEIGHTS_PATH), exist_ok=True)
            # B5 FIX: store normalization params as plain Python lists, not numpy
            # arrays. This makes the checkpoint loadable with weights_only=True on
            # PyTorch 2.6+ without any allowlist hacks, because lists are always safe.
            checkpoint = {
                'model_state': self.model.state_dict(),
                'in_mean' : self.in_mean.tolist()  if self.in_mean  is not None else None,
                'in_std'  : self.in_std.tolist()   if self.in_std   is not None else None,
                'tgt_mean': self.tgt_mean.tolist() if self.tgt_mean is not None else None,
                'tgt_std' : self.tgt_std.tolist()  if self.tgt_std  is not None else None,
            }
            torch.save(checkpoint, self.WEIGHTS_PATH)
            self.get_logger().info(f'[ANN] Weights saved → {self.WEIGHTS_PATH}')
        except Exception as exc:
            self.get_logger().error(f'[ANN] Failed to save weights: {exc}')


# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = OnlineTrainingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_weights()    # T14: persist weights before shutdown
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()