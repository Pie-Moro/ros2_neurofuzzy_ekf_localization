# ROS 2 Hybrid Indoor/Outdoor Localization with Neuro-Fuzzy EKF

A complete ROS 2 package for seamless mobile robot localization across indoor and outdoor environments, combining dual Extended Kalman Filters, an online-trained Artificial Neural Network, and a Fuzzy Logic System for adaptive sensor fusion.

**Platform:** TurtleBot3 Burger · **Simulator:** Gazebo · **ROS 2 Distro:** Humble

---

## Overview

Accurate robot localization breaks down at the boundary between GPS-available (outdoor) and GPS-denied (indoor) environments. This project implements the hybrid fusion architecture proposed by Yousuf & Kadri (2020) in a full ROS 2 simulation stack:

- **Outdoors:** Two parallel EKFs fuse GPS, IMU, and wheel odometry to produce a drift-bounded position estimate.
- **Indoors:** A neural network, trained online during outdoor operation, takes over as a GPS pseudo-sensor. A fuzzy logic system continuously weights the ANN and EKF outputs based on detected wheel slippage.
- **Transitions:** A geofence detector triggers seamless mode-switching with no position jumps or filter divergence.

The system has been validated on a 131-waypoint trajectory covering all 6 rooms of a custom indoor building plus three outdoor loops — completing the full route in 14.2 minutes with zero stuck events and zero recovery cycles.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SENSORS                          │
│  GPS (5 Hz)   IMU (100 Hz)   Wheel Odometry (50 Hz) │
└──────┬───────────┬────────────────┬─────────────────┘
       │           │                │
       ▼           ▼                ▼
┌──────────────────────────────────────────┐
│           navsat_transform               │  GPS → ENU
│        /odometry/gps  (5 Hz)            │
└──────────┬───────────────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌─────────┐
│  KF-1   │  │  KF-2   │   Dual parallel EKFs (robot_localization)
│ GPS+IMU │  │GPS+Odom │
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           ▼
┌──────────────────────┐     ┌─────────────────────┐
│  complementary_filter │     │   ANN (online train) │
│  α1·KF1 + α2·KF2     │     │   GPS pseudo-sensor  │
│  /odometry/fused      │     │   /ann/trajectory    │
└──────────────────────┘     └─────────────────────┘
           │                           │
           └──────────┬────────────────┘
                      ▼
           ┌─────────────────────┐
           │      BT Brain       │   Behavior Tree Orchestrator
           │  IndoorDetector     │   + Fuzzy Logic weighting
           │  /odometry/bt_fused │   + TF publisher (map→odom)
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │ trajectory_controller│   Waypoint follower
           │     /cmd_vel         │   + Door guard v4
           └─────────────────────┘
```

**Outdoor mode:** BT brain blends KF-1 and KF-2 via FLS → publishes `/odometry/bt_fused`

**Indoor mode:** BT brain blends ANN output and KF-2 via FLS → publishes `/odometry/bt_fused`

---

## Package Structure

```
ros2_ws/src/
├── bt_orchestrator_pkg/        C++   BT brain, IndoorDetector, TF publisher, system.launch.py
├── gps_ins_pkg/                YAML  KF-1 (GPS+IMU EKF) + complementary_filter.py
├── gps_odometry_pkg/           YAML  KF-2 (GPS+Odometer EKF)
├── robot_control_brain/        Python  Online ANN training & inference
├── control_pkg/                Python  Waypoint trajectory controller + waypoints.yaml
└── robot_description_pkg/      URDF  TurtleBot3 model + indoor_outdoor.world
```

---

## Key Features

**Dual EKF backbone**
Two parallel `robot_localization` EKF instances run in the map frame. KF-1 fuses GPS+IMU; KF-2 fuses GPS+Odometry. Both use empirically tuned process noise Q=0.05 (required for Gazebo GPS σ=0.316m) and are guarded against covariance blowup via `smooth_lagged_data`, `initial_estimate_covariance`, and carefully chosen `odom_config` vectors.

**Online ANN pseudo-sensor**
A 15→10→5→2 feedforward network (log-sigmoid activations, per Yousuf & Kadri Table II) trains online during GPS-available segments. Input features: IMU×6 + Odometer×5 + KF positions×4 — GPS is deliberately excluded so the network runs when GPS is unavailable. Weights persist across restarts via `torch.save/load`.

**Fuzzy Logic adaptive weighting**
An expert FLS monitors the velocity discrepancy between IMU and odometer to detect wheel slippage, dynamically adjusting the α1/α2 blending coefficients between ANN and KF-2 outputs.

**Door guard v4 + position jump filter**
The trajectory controller implements a per-door alignment guard that requires both physical wall crossing *and* lateral alignment before advancing to the next waypoint. A position jump filter (JUMP_REJECT_M = 1.0m) rejects ANN oscillation spikes (observed up to 1.25m) while passing legitimate GPS transitions (≤0.5m).

**Two-phase stuck recovery**
If a door guard stays active for >8s, the robot executes a two-phase recovery: (1) 1.5s backward to clear the wall, (2) 1.5s in-place rotation to face the target waypoint directly — guaranteeing guard clearance on re-approach.

---

## Prerequisites

- ROS 2 Humble
- Gazebo (Classic)
- [`robot_localization`](https://github.com/cra-ros-pkg/robot_localization) package
- [`BehaviorTree.CPP`](https://github.com/BehaviorTree/BehaviorTree.CPP) v4
- Python 3.10+, PyTorch ≥ 2.0
- TurtleBot3 packages (`turtlebot3`, `turtlebot3_simulations`)

```bash
sudo apt install ros-humble-robot-localization ros-humble-turtlebot3*
pip install torch
```

---

## Build

```bash
cd ~/ros2_neurofuzzy_ekf_localization/ros2_ws

# Full build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Or build only changed packages
colcon build --packages-select bt_orchestrator_pkg \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

---

## Run

```bash
source install/setup.bash
ros2 launch bt_orchestrator_pkg system.launch.py 2>&1 | tee ~/launch_log.txt
```

The launch sequence is time-phased:
| Time | What starts |
|---|---|
| t=0s | Gazebo + robot spawn |
| t=8s | All EKF instances + navsat_transform |
| t=9s | BT brain (immediately bootstraps TF tree) |
| t=12s | complementary_filter + ANN + trajectory_controller |

---

## Record a Bag

```bash
ros2 bag record \
  /odometry/global /odometry/global2 /odom /cmd_vel \
  /bt/indoor_detection /odometry/gps /gps/fix /imu/data \
  /odometry/bt_fused /odometry/fused /ann/trajectory \
  -o run_bag
```

---

## Results

**Session 17 — First Clean Full Run (2026-06-01)**

| Metric | Value |
|---|---|
| Waypoints completed | 131 / 131 |
| Total navigation time | 14.2 min |
| Stuck events | 0 |
| Recovery cycles triggered | 0 |
| ANN training samples | 3000 (capped, Outdoor Pass 2) |
| Final ANN loss | 0.0000 – 0.0001 |
| KF1 P_xx max | 0.56 m² (0% overflow) |
| KF2 P_xx max | 0.22 m² (0% overflow) |
| GPS ENU offset (dx_gps) | −0.266m (varies ±0.5m run-to-run) |

All 7 door guards cleared on first attempt. The s_exit guard (south building exit) exhibited an expected 2–3 cycle oscillation due to ANN lateral noise near the alignment threshold — resolving within 2 seconds without triggering stuck recovery.

---

## Environment

The simulation uses a custom 6-room building (`indoor_outdoor.world`) with the following layout:

```
     NORTH
┌─────────────┬───────────────┬─────────────┐
│ North-West  │  North-Center │  North-East │
│  [Door A2↕] │ via C2 and C1 │  [Door B2↕] │
│  [Door C2←→]│               │  [Door C1←→]│
├─────────────┼───────────────┼─────────────┤
│ South-West  │  South-Center │  South-East │
│  [Door A←→] │ [MAIN ENTRY↕] │  [Door B←→] │
└─────────────┴───────────────┴─────────────┘
     SOUTH
```

The center E-W wall (Wall_1) is solid — NC is only reachable via SW→NW→NC or SE→NE→NC. The trajectory covers all 6 rooms across 3 indoor passes separated by outdoor loops.

---

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `Q_x/Q_y` (both global EKFs) | 0.05 | Empirically required for Gazebo GPS σ=0.316m |
| `linear_speed` | 0.50 m/s | |
| `angular_gain` (K_ω) | 0.50 | K=0.8 caused overshoot crash |
| `max_angular_speed` | 0.50 rad/s | Spin event threshold = 0.60 rad/s |
| `distance_threshold` | 0.45 m | ≥ GPS σ to avoid spurious captures |
| `angle_threshold` | 0.70 rad | +5° margin above GPS noise floor |
| `JUMP_REJECT_M` | 1.00 m | ANN oscillation amplitude = 1.25m |
| `GUARD_STUCK_TIMEOUT` | 8.0 s | |
| ANN architecture | 15→10→5→2 | Log-sigmoid hidden, linear output |
| Geofence source | `/odom` | NOT `/odometry/global` — GPS bias too variable |

---

## References

**[P1]** T. Moore and D. Stouch, *"A Generalized Extended Kalman Filter Implementation for the Robot Operating System,"* in Proc. 5th International Conference on Intelligent Systems and Applications (INTELLI), 2014.

**[P2]** S. Yousuf and M. B. Kadri, *"Information Fusion of GPS, INS and Odometer Sensors for Improving Localization Accuracy of Mobile Robots in Indoor and Outdoor Applications,"* Robotica, vol. 38, no. 9, pp. 1–27, 2020. doi:[10.1017/S0263574720000351](https://doi.org/10.1017/S0263574720000351)
