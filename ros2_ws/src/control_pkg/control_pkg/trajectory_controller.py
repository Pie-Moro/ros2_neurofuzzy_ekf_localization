"""
Trajectory Controller — Optimal Architecture  (Session 20 final)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE: DUAL-SOURCE POSITION — NO OFFSET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Position sources:
  Outdoor  →  /odometry/bt_fused  (GPS-corrected map frame)
  Indoor   →  /odom directly      (wheel encoder, map frame)

WHY /odom directly — not bt_fused, not odom+offset:

  1. Map frame = odom frame (same origin).
     Both share spawn position = GPS datum = Gazebo scene origin.
     Odom (x, y) IS map frame (x, y) — no transformation needed.

  2. The "odom-to-map offset" (bt_fused − odom) IS the GPS bias.
     GPS bias in this setup: 0.25–0.99 m in x, 0.15–0.46 m in y,
     unstable within a single second (confirmed from ODOM_OFF logs).
     Adding this offset to odom INTRODUCES error, not corrects it.

  3. Fatal consequence of off_y > 0 at Door B2 (confirmed bug):
       Guard fires when py > y_wall = 0.201
       py = odom_y + off_y > 0.201  →  odom_y > 0.201 − off_y
       With off_y = +0.25: guard fires at odom_y = −0.049
       Robot is physically 0.25 m SOUTH of Door B2 — not crossed.
     Fatal consequence of off_x > 0 at Door B2:
       off_x = +0.50 → physical odom_x = 4.67 − 0.50 = 4.17 m
       Door B2 opening: x ∈ [4.255, 5.155]
       Physical x = 4.17 < 4.255 → OUTSIDE door, robot hits wall frame.
     Result: robot jammed against Door B2 south face indefinitely.

  4. Gazebo /odom is perfect (zero encoder noise, zero drift).
     TurtleBot3 Burger encoder resolution → sub-mm in simulation.
     Using /odom directly gives exact physical position — better than
     GPS+EKF+ANN for indoor navigation in every metric.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTROL LAW: ALWAYS-FORWARD PURSUIT (no pure-rotation mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ω = clip( Kω × θ_err, ±ω_max )
  v = v_max × max( cos(θ_err), V_MIN_FACTOR )

  Robot drives forward at all times. Speed reduces with misalignment.
  No pure-rotation mode → no spinning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOOR GUARD: GATES CAPTURE ONLY, NOT MOTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Guard v4 (crossing + alignment): WP captured only after physical
  wall crossing AND lateral alignment with door centre.

  With /odom, the guard checks the ACTUAL physical position against
  the ACTUAL wall coordinate. No offset error possible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTDOOR → INDOOR TRANSITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  At wp_idx = 11/60/108, controller switches from bt_fused to /odom.
  GPS bias may shift perceived position by 0.3–0.9 m at transition.
  The always-forward pursuit law corrects this gradually during
  approach — no abrupt stop or recovery needed. South entry door
  opening is 1.39 m wide; even with 0.5 m x-bias at transition,
  the robot remains within the door corridor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ELIMINATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  odom-to-map offset  → map frame = odom frame; offset = GPS noise
  OFFSET_CAPTURE_WPS  → no offset to capture
  Guard hold          → guard gates capture, never motion
  South wall brake    → /odom cannot drift into walls

REFERENCES:
  [P2] Yousuf & Kadri (2020) Robotica 38(9):1759-1786 (Eq.5)
  Sessions 15-20: guard v4, always-forward law, offset removal
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
    approach_side : 'west'|'east' for x-walls; 'south'|'north' for y-walls.
    align_center  : Door centre on orthogonal axis [m].
    safe_half_width: Conservative half-width of door gap [m].
    """
    name:            str
    wall_axis:       str
    wall_coord:      float
    approach_side:   str
    align_center:    float
    safe_half_width: float


# ═══════════════════════════════════════════════════════════════════
# DOOR GATE TABLE  (map frame coords — unchanged from S15)
# ═══════════════════════════════════════════════════════════════════

DOOR_GATE_WPS: Dict[int, DoorGate] = {
    # ── Pass 1 (WP 11–50) ────────────────────────────────────────
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
    # ── Pass 2 (WP 60–94) ────────────────────────────────────────
    60:  DoorGate('s_entry_p2_app', 'y', -4.149, 'south',  0.161, 0.45),
    61:  DoorGate('s_entry_p2_in',  'y', -4.149, 'south',  0.161, 0.45),
    94:  DoorGate('s_exit_p2',      'y', -4.149, 'north',  0.161, 0.45),
    65:  DoorGate('door_A_EW_p2',   'x', -3.501, 'east',  -3.47, 0.40),
    90:  DoorGate('door_A_WE_p2',   'x', -3.501, 'west',  -3.47, 0.40),
    71:  DoorGate('door_A2_SN_p2',  'y',  0.211, 'south', -4.741, 0.40),
    86:  DoorGate('door_A2_NS_p2',  'y',  0.211, 'north', -4.741, 0.40),
    75:  DoorGate('door_C2_WE_p2',  'x', -3.510, 'west',   0.902, 0.40),
    81:  DoorGate('door_C2_EW_p2',  'x', -3.510, 'east',   0.902, 0.40),
    # ── Pass 3 (WP 108–129) ──────────────────────────────────────
    108: DoorGate('s_entry_p3_app', 'y', -4.149, 'south',  0.161, 0.45),
    109: DoorGate('s_entry_p3_in',  'y', -4.149, 'south',  0.161, 0.45),
    129: DoorGate('s_exit_p3',      'y', -4.149, 'north',  0.161, 0.45),
    113: DoorGate('door_B_WE_p3',   'x',  3.849, 'west',  -3.43, 0.40),
    125: DoorGate('door_B_EW_p3',   'x',  3.849, 'east',  -3.43, 0.40),
    117: DoorGate('door_B2_SN_p3',  'y',  0.201, 'south',  4.705, 0.40),
    121: DoorGate('door_B2_NS_p3',  'y',  0.201, 'north',  4.705, 0.40),
}


# ═══════════════════════════════════════════════════════════════════
# INDOOR RANGES
# ═══════════════════════════════════════════════════════════════════

# WP index ranges that use /odom directly as map-frame position.
# Map frame = odom frame (same origin = spawn = GPS datum).
# NO offset applied: GPS bias (off = bt_fused − odom ≈ 0.3–0.9 m)
# must NOT be added — it degrades accuracy and causes early guard fire.
INDOOR_RANGES: Tuple[Tuple[int, int], ...] = ((11, 50), (60, 94), (108, 129))


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# bt_fused jump filter (outdoor — /odom needs no filter)
JUMP_REJECT_M = 1.00
JUMP_WARN_M   = 0.50
# Sustained-reject fallback: after this many consecutive rejects the
# filter force-accepts the incoming GPS value.  Rationale:
#   • Transient ANN indoor spikes resolve in <1s (~10 rejects at 10 Hz).
#   • Permanent GPS EKF drift (e.g. post-indoor re-acquisition) never
#     resolves → infinite cascade without this escape hatch.
# 50 rejects = 5 s at 10 Hz — catches drift, not spikes.
JUMP_REJECT_FALLBACK_COUNT = 50

# Always-forward control: minimum speed fraction (prevents spinning)
# v = v_max * max(cos(theta_err), V_MIN_FACTOR)
V_MIN_FACTOR = 0.15   # 15% of v_max = 0.075 m/s floor

# Two-phase recovery
RECOVERY_PHASE1_SECS = 1.5   # pure backup
RECOVERY_PHASE2_SECS = 1.5   # rotate to face WP
RECOVERY_TOTAL_SECS  = RECOVERY_PHASE1_SECS + RECOVERY_PHASE2_SECS
RECOVERY_VEL         = -0.20  # [m/s] backward

# Unified progress tracker
PROGRESS_DELTA_M      = 0.08  # [m]  min improvement to count as progress
PROGRESS_TIMEOUT_SECS = 8.0   # [s]  no progress => trigger recovery


# ═══════════════════════════════════════════════════════════════════
# CONTROLLER NODE
# ═══════════════════════════════════════════════════════════════════

class TrajectoryController(Node):
    """ROS 2 waypoint controller with dual-source positioning.

    Outdoor: bt_fused (GPS-corrected map frame)
    Indoor:  /odom + captured offset (clean dead-reckoning in map frame)
    """

    # ── Init ──────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('trajectory_controller')

        # Parameters
        self.declare_parameter('waypoints',
            [4.0,1.0, 5.0,5.0, 0.0,5.0, -5.0,5.0, -5.0,0.0,
             -5.0,-5.0, 0.0,-5.0, 5.0,-5.0, 5.0,0.0])
        self.declare_parameter('linear_speed',        0.5)
        self.declare_parameter('angular_gain',        0.50)
        self.declare_parameter('max_angular_speed',   0.50)
        self.declare_parameter('distance_threshold',  0.45)
        self.declare_parameter('angle_threshold',     0.70)
        self.declare_parameter('loop_trajectory',     False)
        self.declare_parameter('align_guard_enabled', True)
        self.declare_parameter('align_guard_threshold', 0.20)

        flat = self.get_parameter('waypoints').value
        self.waypoints  = [(flat[i], flat[i+1]) for i in range(0, len(flat), 2)]
        self._v         = self.get_parameter('linear_speed').value
        self._K_w       = self.get_parameter('angular_gain').value
        self._w_max     = self.get_parameter('max_angular_speed').value
        self._d_th      = self.get_parameter('distance_threshold').value
        self._theta_th  = self.get_parameter('angle_threshold').value
        self._loop      = self.get_parameter('loop_trajectory').value
        self._guard_en  = self.get_parameter('align_guard_enabled').value
        self._eps_th    = self.get_parameter('align_guard_threshold').value

        # bt_fused state (outdoor primary)
        self._bt_x    = 0.0
        self._bt_y    = 0.0
        self._bt_yaw  = 0.0
        self._has_bt  = False
        self._jump_n  = 0
        # Set to True when a WP advances from indoor→outdoor range.
        # The NEXT bt_fused message is force-accepted, overriding the
        # stale pre-indoor GPS anchor that has been frozen since indoor
        # pass entry.  Without this, the ~8 m gap between the frozen
        # outdoor-GPS last_pos and the post-indoor GPS re-acquisition
        # value triggers an infinite JUMP_REJECT cascade on the first
        # outdoor WP (confirmed: Test 3, WP130, 333+ consecutive rejects).
        self._pending_bt_reset: bool = False

        # /odom state (indoor primary — exact map-frame position, no offset)
        self._odom_x   = 0.0
        self._odom_y   = 0.0
        self._odom_yaw = 0.0
        self._has_odom = False

        # Navigation state
        self.wp_idx     = 0
        self._done      = False
        self._last_log  = self.get_clock().now()
        self._guard_name = ''

        # Recovery state
        self._in_recovery      = False
        self._recovery_elapsed = 0.0

        # Unified progress tracker
        self._prog_min_dist = math.inf
        self._prog_last_t   = None

        # ROS interfaces
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odometry/bt_fused', self._bt_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_timer(0.1, self._loop_cb)

        self.get_logger().info(
            f'TrajectoryController ready — {len(self.waypoints)} WPs | '
            f'guard v4 (capture-gate only) | eps={self._eps_th}m\n'
            f'  Outdoor: /odometry/bt_fused  |  Indoor: /odom  (map=odom frame, NO offset)\n'
            f'  Control: v*max(cos(θ),{V_MIN_FACTOR}) — always-forward, no spinning\n'
            f'  Stuck  : δ={PROGRESS_DELTA_M}m T={PROGRESS_TIMEOUT_SECS}s\n'
            f'  Params : v={self._v} K_w={self._K_w} w_max={self._w_max} d_th={self._d_th}')

    # ═══════════════════════════════════════════════════════════════
    # POSITION PROPERTIES
    # ═══════════════════════════════════════════════════════════════
    # Indoor  → /odom directly (map frame = odom frame, NO offset).
    #           Offset = GPS bias 0.3–0.9 m; adding it introduces error.
    # Outdoor → bt_fused (GPS-corrected, map frame).
    # ═══════════════════════════════════════════════════════════════

    @property
    def _indoor(self) -> bool:
        return any(lo <= self.wp_idx <= hi for lo, hi in INDOOR_RANGES)

    @property
    def px(self) -> float:
        if self._indoor and self._has_odom:
            return self._odom_x          # exact physical position, no GPS noise
        return self._bt_x

    @property
    def py(self) -> float:
        if self._indoor and self._has_odom:
            return self._odom_y
        return self._bt_y

    @property
    def pyaw(self) -> float:
        if self._indoor and self._has_odom:
            return self._odom_yaw
        return self._bt_yaw

    # ═══════════════════════════════════════════════════════════════
    # SENSOR CALLBACKS
    # ═══════════════════════════════════════════════════════════════

    def _bt_cb(self, msg: Odometry) -> None:
        nx  = msg.pose.pose.position.x
        ny  = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        _, _, nyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if not self._has_bt:
            self._bt_x, self._bt_y, self._bt_yaw = nx, ny, nyaw
            self._has_bt = True
            self.get_logger().info(
                f'/odometry/bt_fused received '
                f'(frame={msg.header.frame_id}). Controller armed.')
            return

        jump = math.hypot(nx - self._bt_x, ny - self._bt_y)

        # ── Force-accept path A: indoor→outdoor WP transition ────────────────
        # When wp_idx crosses from an INDOOR_RANGE to an outdoor WP, the
        # bt_fused jump-filter last_pos is STALE — it was frozen at the last
        # accepted outdoor GPS value from *before* the indoor pass started.
        # The entire indoor segment accumulates JUMP_REJECTs (ANN noise >1 m),
        # so last_pos never updates.  When GPS re-acquires on outdoor re-entry
        # the new GPS position can be 6–9 m from the frozen last_pos, causing
        # an infinite cascade.  Force-accepting on the first callback after the
        # transition resets last_pos to the current GPS position.
        if self._pending_bt_reset:
            self._pending_bt_reset = False
            old_x, old_y = self._bt_x, self._bt_y
            self._jump_n = 0
            self._bt_x, self._bt_y, self._bt_yaw = nx, ny, nyaw
            self.get_logger().info(
                f'[JUMP_RESET] Indoor→outdoor transition: '
                f'({old_x:.2f},{old_y:.2f})→({nx:.2f},{ny:.2f}) '
                f'Δ={jump:.2f}m (stale anchor cleared)')
            return

        # ── Force-accept path B: sustained consecutive rejection ─────────────
        # Transient ANN spikes resolve in <1 s (~10 rejects); GPS drift is
        # permanent.  After FALLBACK_COUNT consecutive rejects the filter
        # force-accepts, breaking the infinite cascade.
        if self._jump_n >= JUMP_REJECT_FALLBACK_COUNT:
            n = self._jump_n
            old_x, old_y = self._bt_x, self._bt_y
            self._jump_n = 0
            self._bt_x, self._bt_y, self._bt_yaw = nx, ny, nyaw
            self.get_logger().warn(
                f'[JUMP_RESET] {n} consecutive rejects — '
                f'sustained GPS drift accepted: '
                f'({old_x:.2f},{old_y:.2f})→({nx:.2f},{ny:.2f}) '
                f'Δ={jump:.2f}m')
            return

        # ── Normal path: jump filter ──────────────────────────────────────────
        if jump > JUMP_REJECT_M:
            self._jump_n += 1
            self.get_logger().warn(
                f'[JUMP_REJECT] delta={jump:.2f}m '
                f'({self._bt_x:.2f},{self._bt_y:.2f})->'
                f'({nx:.2f},{ny:.2f}) [#{self._jump_n}]')
            return
        if jump > JUMP_WARN_M:
            self.get_logger().info(
                f'[JUMP_WARN] delta={jump:.2f}m accepted '
                f'({self._bt_x:.2f},{self._bt_y:.2f})->'
                f'({nx:.2f},{ny:.2f})')

        self._jump_n = 0
        self._bt_x, self._bt_y, self._bt_yaw = nx, ny, nyaw

    def _odom_cb(self, msg: Odometry) -> None:
        # No filter — wheel odometry is inherently smooth
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._odom_x   = msg.pose.pose.position.x
        self._odom_y   = msg.pose.pose.position.y
        self._odom_yaw = yaw
        self._has_odom = True

    # ═══════════════════════════════════════════════════════════════
    # DOOR GUARD  v4  (crossing + alignment required)
    # ═══════════════════════════════════════════════════════════════

    def _guard(self, gate: DoorGate) -> Tuple[bool, str]:
        """Guard deactivates only when crossed AND aligned.
        Uses self.px/py (odom-compensated indoors) for clean detection."""
        if gate.wall_axis == 'x':
            robot_wall  = self.px
            robot_align = self.py
            crossed = (robot_wall > gate.wall_coord
                       if gate.approach_side == 'west'
                       else robot_wall < gate.wall_coord)
        else:
            robot_wall  = self.py
            robot_align = self.px
            crossed = (robot_wall > gate.wall_coord
                       if gate.approach_side == 'south'
                       else robot_wall < gate.wall_coord)

        align_err = abs(robot_align - gate.align_center)

        if crossed and align_err <= self._eps_th:
            return False, f'crossed+aligned eps={align_err:.3f}m'
        if crossed:
            return True, f'crossed-misaligned eps={align_err:.3f}m'
        return True, f'not-crossed eps={align_err:.3f}m'

    # ═══════════════════════════════════════════════════════════════
    # PROGRESS TRACKER
    # ═══════════════════════════════════════════════════════════════

    def _reset_progress(self) -> None:
        self._prog_min_dist = math.inf
        self._prog_last_t   = None

    def _check_progress(self, dist: float) -> bool:
        """Returns True when stuck (no progress for PROGRESS_TIMEOUT_SECS)."""
        now = self.get_clock().now()
        if self._prog_last_t is None:
            self._prog_last_t   = now
            self._prog_min_dist = dist
            return False

        if dist < self._prog_min_dist - PROGRESS_DELTA_M:
            self._prog_min_dist = dist
            self._prog_last_t   = now
            return False

        stall = (now - self._prog_last_t).nanoseconds / 1e9
        if stall > PROGRESS_TIMEOUT_SECS:
            src = 'odom' if (self._indoor and self._has_odom) else 'bt'
            self.get_logger().warn(
                f'[STUCK] WP{self.wp_idx}: stalled {stall:.1f}s at '
                f'({self.px:.2f},{self.py:.2f}) [{src}]  '
                f'dist={dist:.2f}m  min_seen={self._prog_min_dist:.2f}m. '
                f'Triggering recovery.')
            return True
        return False

    # ═══════════════════════════════════════════════════════════════
    # RECOVERY  (two-phase: backup → rotate to face WP)
    # ═══════════════════════════════════════════════════════════════

    def _run_recovery(self) -> None:
        self._recovery_elapsed += 0.1
        twist = Twist()

        if self._recovery_elapsed <= RECOVERY_PHASE1_SECS:
            # Phase 1: pure backup — create physical clearance
            twist.linear.x  = RECOVERY_VEL
            twist.angular.z = 0.0
        else:
            # Phase 2: rotate to face WP — guarantees clean approach
            tx, ty = self.waypoints[self.wp_idx]
            t_yaw  = math.atan2(ty - self.py, tx - self.px)
            a_err  = math.atan2(math.sin(t_yaw - self.pyaw),
                                math.cos(t_yaw - self.pyaw))
            twist.angular.z = max(-self._w_max,
                                  min(self._w_max, 1.5 * self._K_w * a_err))

        self._pub.publish(twist)

        if self._recovery_elapsed >= RECOVERY_TOTAL_SECS:
            self._in_recovery      = False
            self._recovery_elapsed = 0.0
            self._guard_name       = ''
            self._reset_progress()
            self.get_logger().info(
                f'[RECOVERY] Done. Faces WP{self.wp_idx}, resuming.')

    # ═══════════════════════════════════════════════════════════════
    # MAIN CONTROL LOOP  (10 Hz)
    # ═══════════════════════════════════════════════════════════════

    def _loop_cb(self) -> None:
        if not self._has_bt or self._done:
            return

        # Recovery mode — run state machine, skip normal control
        if self._in_recovery:
            self._run_recovery()
            return

        # Trajectory complete
        if self.wp_idx >= len(self.waypoints):
            if self._loop:
                self.wp_idx = 0
            else:
                self._done = True
                self.get_logger().info('Trajectory completed!')
                self._pub.publish(Twist())
            return

        tx, ty = self.waypoints[self.wp_idx]

        # ── Door guard ────────────────────────────────────────────
        guard_active = False
        guard_name   = ''

        if self._guard_en and self.wp_idx in DOOR_GATE_WPS:
            gate = DOOR_GATE_WPS[self.wp_idx]
            guard_active, reason = self._guard(gate)
            if guard_active:
                guard_name = gate.name
                if self._guard_name != guard_name:
                    self.get_logger().warn(
                        f'[GUARD] {guard_name} WP{self.wp_idx} {reason}')
                    self._guard_name = guard_name
            else:
                if self._guard_name:
                    self.get_logger().info(
                        f'[GUARD] Cleared {self._guard_name} ({reason})')
                    self._guard_name = ''

        # ── Distance and heading ──────────────────────────────────
        dx   = tx - self.px
        dy   = ty - self.py
        dist = math.hypot(dx, dy)

        t_yaw = math.atan2(dy, dx)
        a_err = math.atan2(math.sin(t_yaw - self.pyaw),
                           math.cos(t_yaw - self.pyaw))

        # ── Unified stuck detection ───────────────────────────────
        if self._check_progress(dist):
            self._in_recovery      = True
            self._recovery_elapsed = 0.0
            self._reset_progress()
            self._pub.publish(Twist())
            return

        # ── Throttled position log ────────────────────────────────
        now = self.get_clock().now()
        if (now - self._last_log).nanoseconds > 1e9:
            src  = 'odom' if (self._indoor and self._has_odom) else 'bt'
            gtag = f' [GUARD:{guard_name}]' if guard_active else ''
            self.get_logger().info(
                f'[{src}] ({self.px:.2f},{self.py:.2f})  '
                f'WP{self.wp_idx}:({tx:.2f},{ty:.2f})  '
                f'dist:{dist:.2f}m  th:{math.degrees(a_err):.1f}deg{gtag}')
            self._last_log = now

        # ── WP capture ────────────────────────────────────────────
        # INVARIANT: guard gates WP capture only — NEVER robot motion.
        # Robot drives forward continuously through all door WPs.
        # Guard clears when crossed+aligned; WP sits on wall plane
        # → dist ≈ 0 at crossing → captured same tick guard clears.
        if dist < self._d_th and not guard_active:
            self.get_logger().info(
                f'WP {self.wp_idx} captured: ({tx:.2f},{ty:.2f})')
            # Detect indoor→outdoor WP transition BEFORE incrementing
            # wp_idx so both the old and new range membership are visible.
            was_indoor = self._indoor
            self.wp_idx += 1
            if was_indoor and not self._indoor:
                # First outdoor WP after an indoor segment: the bt_fused
                # jump-filter last_pos is stale (frozen since indoor entry).
                # Arm the force-accept flag so the next bt_fused callback
                # resets last_pos to the current GPS re-acquisition value.
                self._pending_bt_reset = True
                self.get_logger().info(
                    f'[JUMP_RESET_ARM] WP{self.wp_idx-1}→WP{self.wp_idx}: '
                    f'indoor→outdoor transition detected. '
                    f'bt_fused filter will reset on next callback.')
            self._reset_progress()
            return

        # ── Always-forward pursuit law ────────────────────────────
        #
        #   omega = clip(K_w * theta_err, ±w_max)
        #   v     = v_max * max(cos(theta_err), V_MIN_FACTOR)
        #
        # The cos(theta_err) term:
        #   aligned (0 deg)   → v = v_max          (full speed)
        #   45 deg            → v = 0.707 * v_max  (moderate)
        #   90 deg+           → v = V_MIN_FACTOR*v_max  (creep while turning)
        #
        # Eliminates pure-rotation mode. With clean /odom position
        # indoors, theta_err is smooth → control is smooth → no spinning.
        twist = Twist()
        twist.angular.z = max(-self._w_max,
                              min(self._w_max, self._K_w * a_err))
        twist.linear.x  = self._v * max(math.cos(a_err), V_MIN_FACTOR)
        self._pub.publish(twist)

    def destroy_node(self):
        self._pub.publish(Twist())
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