# ros2_neurofuzzy_ekf_localization
ROS 2 Project for hybrid mobile robot localization . It utilizes dual Extended Kalman Filters for outdoor GPS-INS-odometry fusion . During GPS outages, it transitions to an ANN for position prediction, utilizing a Fuzzy Logic System to compute dynamic blending parameters and mitigate wheel slippage .

## EKF Evaluation

To calculate trajectory and RMSE of the EKF filters on recorded rosbag files, use the `evaluate_ekf.py` script located in the `ros2_ws` directory.

Example:

```bash
cd /home/pietro/ros2_neurofuzzy_ekf_localization/ros2_ws
python3 evaluate_ekf.py --bag Traiettoria7 --ref /odom --topics /odometry/global /odometry/global2 /odometry/fused --plot
```

- `--bag` specifies the rosbag2 directory (`Traiettoria7`, `Traiettoria6`, ...)
- `--ref` specifies the reference topic for comparison (default `/odom`)
- `--topics` specifies the EKF output topics to evaluate
- `--plot` displays a graph comparing the trajectories
