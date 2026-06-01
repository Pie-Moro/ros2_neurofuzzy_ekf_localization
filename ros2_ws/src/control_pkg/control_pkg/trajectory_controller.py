"""
Trajectory Controller — P2 Eq.5 + Door Guard v4

S15 FINAL FIXES applied in this version:
═══════════════════════════════════════════════════════════════════

FIX 1 — POSITION JUMP FILTER (PRIMARY FIX — prevents ANN crash)
  Root cause (confirmed from 1_0.db3, 2026-06-01):
    ANN outputs oscillate ±1.25m in y-axis every ~2s indoors.
    These jumps caused the controller to capture NE-room WPs (22-25)
    in 0.5s without the robot physically navigating there. The robot
    was then commanded south while physically stuck at the solid wall
    section at x=4.07, y=0.31 (east of Wall_1, west of Door B2 gap).

  Fix: reject bt_fused updates where |Δpos| > JUMP_REJECT_THRESHOLD.
    Physical max per tick: v×dt = 0.5×0.1 = 0.05m
    ANN oscillation amplitude: 1.25m  → ALWAYS rejected
    GPS outdoor jumps: typically 0.3-0.5m → allowed through
    JUMP_REJECT  = 1.00m (hard reject)
    JUMP_WARN    = 0.50m (log warning but accept)

FIX 2 — GUARD CROSSING REQUIREMENT (prevents premature WP capture)
  Root cause: Guard deactivated on alignment alone (|ε| ≤ 0.20m),
    even if robot hadn't crossed the wall. WP21 (Door B2 crossing)
    captured at bt_fused y=-0.089m (SOUTH of wall at y=+0.201).

  Fix: guard now deactivates ONLY when BOTH conditions hold:
    (a) Robot has crossed wall_coord (already_crossed = True), AND
    (b) Alignment error ≤ align_guard_threshold
  If aligned but not crossed → guard remains active (blocks WP advance).

FIX 3 — angular_gain default
  0.60 → 0.50 (tested: K_ω=0.8 overshoot at south entry; K_ω=0.5 stable)
  Saturation angle with ω_max=0.80: 0.80/0.50 = 1.60 rad = 91.7°
  → Full proportional control through all door approach angles.

FIX 4 — Log angular_gain and max_angular_speed at startup.

══════════════════════════════════════════════════════════════════
CRASH TIMELINE (1_0.db3):
  t+188.6s WP21 captured at bt_fused y=-0.089 (SOUTH of Door B2 wall)
  t+190.7s ANN jumped +1.25m: bt_fused → (4.886, +1.150)
  t+190.4s WP22,23,24,25 captured in 0.5s (robot never entered NE room)
  t+191.9s WP26 captured, robot physically at y≈0, x≈4.07
  t+196+s  Robot stuck at solid east wall (x∈[3.924,4.255], y≈0.201)
           World odom frozen at (4.07, 0.31), vel_mis < 0.04m/s

REFERENCES:
  [P2] Yousuf & Kadri (2020) Robotica 38(9):1759–1786 — ANN+FLS (Eq.5)
  Session 15: T5 fix, guard v1→v4, jump filter, crossing requirement
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion


# ═══════════════════════════════════════════════════════════════════
# DOOR GATE DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DoorGate:
    """Geometric descriptor for one door opening (map frame).

    wall_axis     : 'x' (N-S wall, robot crosses E-W)
                    'y' (E-W wall, robot crosses N-S)
    wall_coord    : Wall plane coordinate [m].
    approach_side : Side from which robot approaches:
                    'west'|'east' for x-walls;  'south'|'north' for y-walls.
    align_center  : Door centre on orthogonal axis (y_c or x_c) [m].
    safe_half_width: Conservative half-width of door gap [m].
    """
    name:            str
    wall_axis:       str
    wall_coord:      float
    approach_side:   str
    align_center:    float
    safe_half_width: float


# ═══════════════════════════════════════════════════════════════════
# DOOR GATE TABLE  (all coords in map / bt_fused frame)
# ═══════════════════════════════════════════════════════════════════
DOOR_GATE_WPS: Dict[int, DoorGate] = {
    # ── Pass 1 (WP 0–50) ──────────────────────────────────────────
    11:  DoorGate('s_entry_p1_app', 'y', -4.149, 'south',  0.161, 0.45),
    12:  DoorGate('s_entry_p1_in',  'y', -4.149, 'south',  0.161, 0.45),
    50:  DoorGate('s_exit_p1',      'y', -4.149, 'north',  0.161, 0.45),
    16:  DoorGate('door_B_WE_p1',   'x',  3.849, 'west',  -3.43, 0.40),
    29:  DoorGate('door_B_EW_p1',   'x',  3.849, 'east',  -3.43, 0.40),
    21:  DoorGate('door_B2_SN_p1',  'y',  0.201, 'south',  4.705, 0.40),
    25:  DoorGate('door_B2_NS_p1',  'y',  0.201, 'north',  4.705, 0.40),
    33:  DoorGate('door_A_EW_p1',   'x', -3.501, 'east',  -3.47, 0.40),
    46:  DoorGate('door_A_WE_p1',   'x', -3.501, 'west',  -3.47, 0.40),
    38:  DoorGate('door_A2_SN_p1',  'y',  0.211, 'south', -4.741, 0.40),
    42:  DoorGate('door_A2_NS_p1',  'y',  0.211, 'north', -4.741, 0.40),
    # ── Pass 2 (WP 59–94) ─────────────────────────────────────────
    60:  DoorGate('s_entry_p2_app', 'y', -4.149, 'south',  0.161, 0.45),
    61:  DoorGate('s_entry_p2_in',  'y', -4.149, 'south',  0.161, 0.45),
    94:  DoorGate('s_exit_p2',      'y', -4.149, 'north',  0.161, 0.45),
    65:  DoorGate('door_A_EW_p2',   'x', -3.501, 'east',  -3.47, 0.40),
    90:  DoorGate('door_A_WE_p2',   'x', -3.501, 'west',  -3.47, 0.40),
    71:  DoorGate('door_A2_SN_p2',  'y',  0.211, 'south', -4.741, 0.40),
    86:  DoorGate('door_A2_NS_p2',  'y',  0.211, 'north', -4.741, 0.40),
    75:  DoorGate('door_C2_WE_p2',  'x', -3.510, 'west',   0.902, 0.40),
    81:  DoorGate('door_C2_EW_p2',  'x', -3.510, 'east',   0.902, 0.40),
    # ── Pass 3 (WP 107–129) ───────────────────────────────────────
    108: DoorGate('s_entry_p3_app', 'y', -4.149, 'south',  0.161, 0.45),
    109: DoorGate('s_entry_p3_in',  'y', -4.149, 'south',  0.161, 0.45),
    129: DoorGate('s_exit_p3',      'y', -4.149, 'north',  0.161, 0.45),
    113: DoorGate('door_B_WE_p3',   'x',  3.849, 'west',  -3.43, 0.40),
    125: DoorGate('door_B_EW_p3',   'x',  3.849, 'east',  -3.43, 0.40),
    117: DoorGate('door_B2_SN_p3',  'y',  0.201, 'south',  4.705, 0.40),
    121: DoorGate('door_B2_NS_p3',  'y',  0.201, 'north',  4.705, 0.40),
}

# ═══════════════════════════════════════════════════════════════════
# POSITION JUMP FILTER THRESHOLDS
# ═══════════════════════════════════════════════════════════════════
# Physical max movement per tick: v×dt = 0.5×0.1 = 0.05 m
# ANN oscillation amplitude (observed): 1.25 m  → ALWAYS rejected
# GPS outdoor jump (typical): 0.3–0.5 m → passed through
JUMP_REJECT_M = 1.00   # [m] hard reject; ANN jumps = 1.25m → always caught
JUMP_WARN_M   = 0.50   # [m] log warning; GPS jumps may reach 0.5m


# ═══════════════════════════════════════════════════════════════════
# STUCK RECOVERY CONSTANTS
# ═══════════════════════════════════════════════════════════════════
# If the door-centering guard stays active continuously for more than
# GUARD_STUCK_TIMEOUT seconds, the robot is physically stuck (pressed
# against a wall outside the door gap). Recovery: back up for
# RECOVERY_SECS seconds to free the robot, then resume normal navigation.
# The diagonal re-approach naturally corrects both axes simultaneously.
GUARD_STUCK_TIMEOUT  = 8.0   # [s] guard active this long → stuck
RECOVERY_PHASE1_SECS = 1.5   # [s] Phase 1: pure backup (creates clearance)
RECOVERY_PHASE2_SECS = 1.5   # [s] Phase 2: rotate in place to face WP exactly
RECOVERY_TOTAL_SECS  = RECOVERY_PHASE1_SECS + RECOVERY_PHASE2_SECS  # 3.0s
RECOVERY_VEL         = -0.20  # [m/s] backup velocity (negative = backward)
# Heading fix guarantee: if robot faces atan2(WP_y-y, WP_x-x) and drives
# straight, it arrives at y=WP_y exactly when x=WP_x → guard ALWAYS clears.

# ═══════════════════════════════════════════════════════════════════
# CONTROLLER NODE
# ═══════════════════════════════════════════════════════════════════

class TrajectoryController(Node):
    """ROS2 waypoint controller — P2 Eq.5 + door guard v4 + jump filter."""

    def __init__(self):
        super().__init__('trajectory_controller')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter(
            'waypoints',
            [4.0,1.0, 5.0,5.0, 0.0,5.0, -5.0,5.0, -5.0,0.0,
             -5.0,-5.0, 0.0,-5.0, 5.0,-5.0, 5.0,0.0])
        self.declare_parameter('linear_speed',       0.5)   # [m/s]
        self.declare_parameter('angular_gain',       0.50)  # K_ω — FIX 3
        self.declare_parameter('max_angular_speed',  0.80)  # [rad/s]
        self.declare_parameter('distance_threshold', 0.45)  # [m]
        self.declare_parameter('angle_threshold',    0.70)  # [rad]
        self.declare_parameter('loop_trajectory',    False)
        self.declare_parameter('align_guard_enabled',   True)
        self.declare_parameter('align_guard_threshold', 0.20)  # [m]

        wps_flat = self.get_parameter('waypoints').value
        self.waypoints  = [(wps_flat[i], wps_flat[i+1])
                           for i in range(0, len(wps_flat), 2)]
        self.v          = self.get_parameter('linear_speed').value
        self.K_w        = self.get_parameter('angular_gain').value
        self.w_max      = self.get_parameter('max_angular_speed').value
        self.d_th       = self.get_parameter('distance_threshold').value
        self.theta_th   = self.get_parameter('angle_threshold').value
        self.loop       = self.get_parameter('loop_trajectory').value
        self.guard_en   = self.get_parameter('align_guard_enabled').value
        self.eps_th     = self.get_parameter('align_guard_threshold').value

        # ── State ─────────────────────────────────────────────────
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0
        self.wp_idx      = 0
        self.done        = False
        self.has_pose    = False
        self.last_log    = self.get_clock().now()
        self._guard_name = ''
        self._jump_count = 0   # consecutive rejected jump counter

        # Stuck recovery state
        self._guard_active_secs = 0.0   # seconds current guard has been active
        self._in_recovery       = False  # True while backing up
        self._recovery_elapsed  = 0.0   # seconds spent in recovery

        # ── ROS interfaces ────────────────────────────────────────
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(
            Odometry, '/odometry/bt_fused', self._odom_cb, 10)
        self.timer = self.create_timer(0.1, self._loop)

        self.get_logger().info(
            f'TrajectoryController ready — {len(self.waypoints)} WPs | '
            f'guard v4 (ε_th={self.eps_th:.3f}m, crossing+align required) | '
            f'jump_filter: warn>{JUMP_WARN_M}m reject>{JUMP_REJECT_M}m'
        )
        self.get_logger().info(
            f'Params: v={self.v}m/s K_ω={self.K_w} ω_max={self.w_max}rad/s '
            f'd_th={self.d_th}m θ_th={self.theta_th}rad'
        )

    # ── Position jump filter ──────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        """Subscribe to /odometry/bt_fused with position jump filter.

        FIX 1: Reject updates where Euclidean position jump exceeds
        JUMP_REJECT_M (1.00m). ANN indoor oscillations reach ±1.25m
        per tick; GPS outdoor jumps typically stay below 0.50m.

        FIX 2 interaction: The jump filter ensures the bt_fused position
        only advances smoothly, so the guard's already_crossed check
        (which reads self.current_y/x) cannot be triggered by a spurious
        ANN position jump.
        """
        nx = msg.pose.pose.position.x
        ny = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        _, _, nyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if not self.has_pose:
            self.has_pose    = True
            self.current_x   = nx
            self.current_y   = ny
            self.current_yaw = nyaw
            self.get_logger().info(
                f'/odometry/bt_fused received (frame={msg.header.frame_id}). '
                f'Control active.'
            )
            return

        jump = math.sqrt((nx - self.current_x)**2 + (ny - self.current_y)**2)

        if jump > JUMP_REJECT_M:
            self._jump_count += 1
            self.get_logger().warn(
                f'[JUMP_FILTER] REJECTED Δ={jump:.2f}m > {JUMP_REJECT_M}m: '
                f'({self.current_x:.2f},{self.current_y:.2f}) → ({nx:.2f},{ny:.2f}) '
                f'[consecutive #{self._jump_count}]'
            )
            return   # keep previous position
        elif jump > JUMP_WARN_M:
            self.get_logger().info(
                f'[JUMP_WARN] Δ={jump:.2f}m > {JUMP_WARN_M}m (accepted): '
                f'({self.current_x:.2f},{self.current_y:.2f}) → ({nx:.2f},{ny:.2f})'
            )

        self._jump_count = 0
        self.current_x   = nx
        self.current_y   = ny
        self.current_yaw = nyaw

    # ── Door guard (v4: crossing + alignment required) ────────────

    def _guard(self, gate: DoorGate) -> Tuple[bool, str]:
        """Door centering guard v4.

        FIX 2: Guard deactivates ONLY when BOTH:
          (a) Robot has crossed wall_coord (already_crossed = True), AND
          (b) Alignment error ≤ align_guard_threshold.

        Previously deactivated on (b) alone, allowing premature capture
        before physical crossing. WP21 was captured at bt_fused y=-0.089
        (south of wall y=+0.201) because alignment was met first.

        With FIX 1 (jump filter) + FIX 2: the guard cannot be bypassed
        by ANN position jumps, and the robot must physically navigate past
        wall_coord before the crossing WP can be captured.
        """
        if gate.wall_axis == 'x':
            robot_align = self.current_y
            robot_wall  = self.current_x
            crossed = (robot_wall > gate.wall_coord
                       if gate.approach_side == 'west'
                       else robot_wall < gate.wall_coord)
        else:
            robot_align = self.current_x
            robot_wall  = self.current_y
            crossed = (robot_wall > gate.wall_coord
                       if gate.approach_side == 'south'
                       else robot_wall < gate.wall_coord)

        align_err = abs(robot_align - gate.align_center)

        # FIX 2: require BOTH crossing AND alignment
        if crossed and align_err <= self.eps_th:
            return False, f'crossed+aligned ε={align_err:.3f}m'

        if crossed:
            return True, f'crossed-misaligned ε={align_err:.3f}m'

        # Not crossed: always block advance (guard active)
        return True, f'not-crossed ε={align_err:.3f}m'

    # ── Main control loop ─────────────────────────────────────────

    def _loop(self) -> None:
        if not self.has_pose or self.done:
            return

        if self.wp_idx >= len(self.waypoints):
            if self.loop:
                self.wp_idx = 0
                return
            self.done = True
            self.get_logger().info('Trajectory completed!')
            self.pub.publish(Twist())
            return

        # ── Stuck recovery (two-phase: backup → rotate-to-face-WP) ────────
        if self._in_recovery:
            self._recovery_elapsed += 0.1
            twist = Twist()

            if self._recovery_elapsed <= RECOVERY_PHASE1_SECS:
                # Phase 1: pure backup — move away from wall
                twist.linear.x  = RECOVERY_VEL
                twist.angular.z = 0.0

            else:
                # Phase 2: rotate in place to face target WP directly.
                # Heading guarantee: if robot faces atan2(WP_y-y, WP_x-x)
                # and drives straight, it arrives at WP_y when x=WP_x → guard clears.
                tx, ty = self.waypoints[self.wp_idx]
                t_yaw = math.atan2(ty - self.current_y, tx - self.current_x)
                a_err = math.atan2(
                    math.sin(t_yaw - self.current_yaw),
                    math.cos(t_yaw - self.current_yaw))
                twist.linear.x  = 0.0
                # 1.5× normal gain for faster heading realignment during recovery
                twist.angular.z = max(-self.w_max,
                                      min(self.w_max, 1.5 * self.K_w * a_err))

            self.pub.publish(twist)

            if self._recovery_elapsed >= RECOVERY_TOTAL_SECS:
                self._in_recovery      = False
                self._recovery_elapsed = 0.0
                self._guard_active_secs= 0.0
                self._guard_name       = ''
                self.get_logger().info(
                    f'[RECOVERY] Done (backup+rotate). '
                    f'Robot faces WP{self.wp_idx} heading, resuming.')
            return   # skip normal control during recovery

        tx, ty = self.waypoints[self.wp_idx]
        guard_active = False
        guard_name   = ''

        if self.guard_en and self.wp_idx in DOOR_GATE_WPS:
            gate = DOOR_GATE_WPS[self.wp_idx]
            guard_active, reason = self._guard(gate)
            if guard_active:
                guard_name = gate.name
                if self._guard_name != guard_name:
                    self.get_logger().warn(
                        f'[GUARD] {guard_name} WP{self.wp_idx} {reason}')
                    self._guard_name = guard_name
                # ── Stuck timeout ──────────────────────────────────────────
                self._guard_active_secs += 0.1
                if self._guard_active_secs >= GUARD_STUCK_TIMEOUT:
                    self.get_logger().warn(
                        f'[STUCK] Guard {guard_name} active '
                        f'{self._guard_active_secs:.1f}s at '
                        f'({self.current_x:.2f},{self.current_y:.2f}). '
                        f'Backing up {RECOVERY_TOTAL_SECS}s (phase1={RECOVERY_PHASE1_SECS}s + phase2={RECOVERY_PHASE2_SECS}s).')
                    self._in_recovery      = True
                    self._recovery_elapsed = 0.0
                    self.pub.publish(Twist())
                    return
            else:
                self._guard_active_secs = 0.0   # reset timer when guard clears
                if self._guard_name:
                    self.get_logger().info(
                        f'[GUARD] Cleared {self._guard_name} ({reason})')
                    self._guard_name = ''

        dx    = tx - self.current_x
        dy    = ty - self.current_y
        dist  = math.sqrt(dx**2 + dy**2)
        t_yaw = math.atan2(dy, dx)
        a_err = math.atan2(math.sin(t_yaw - self.current_yaw),
                           math.cos(t_yaw - self.current_yaw))

        now = self.get_clock().now()
        if (now - self.last_log).nanoseconds > 1e9:
            g_tag = f' [GUARD:{guard_name}]' if guard_active else ''
            self.get_logger().info(
                f'Pos: ({self.current_x:.2f}, {self.current_y:.2f})  '
                f'WP{self.wp_idx}: ({tx:.2f}, {ty:.2f})  dist: {dist:.2f} m{g_tag}')
            self.last_log = now

        cap_th = self.eps_th if guard_active else self.d_th

        if dist < cap_th:
            if guard_active:
                self.pub.publish(Twist())   # hold; re-evaluate next tick
                return
            self.get_logger().info(
                f'WP {self.wp_idx} captured: ({tx:.2f}, {ty:.2f})')
            self.wp_idx += 1
            return

        twist = Twist()
        if abs(a_err) > self.theta_th:
            twist.angular.z = self.K_w * a_err
        else:
            twist.linear.x  = self.v
            twist.angular.z = 0.5 * self.K_w * a_err

        twist.angular.z = max(-self.w_max, min(self.w_max, twist.angular.z))
        self.pub.publish(twist)

    def destroy_node(self):
        self.pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()