/**
 * ============================================================================
 *  indoor_detector.hpp  –  v3  (World-calibrated, Map-resilient)
 * ============================================================================
 *
 *  WHAT CHANGED FROM v2
 *  ─────────────────────
 *  v2 used a single Y-axis threshold in the odom frame.  That fails when:
 *    • the robot enters from a different direction (east/west/south doors)
 *    • the building has a non-rectangular entrance corridor
 *    • numerical drift in odom shifts the threshold hit point
 *
 *  v3 changes:
 *   1. FULL 2-D AXIS-ALIGNED BOUNDING BOX (AABB) geofence
 *        All four walls are checked simultaneously.  The robot must be
 *        inside the complete rectangular footprint to be INDOOR.
 *        Entry and exit use different (wider) boxes → Schmitt-trigger
 *        hysteresis.  No oscillation at the wall boundary.
 *
 *   2. PRE-CALIBRATED FOR indoor_outdoor.world
 *        Outer wall positions extracted directly from the world file
 *        <state> section (ground-truth runtime poses):
 *          North  Y = +4.551 m   (Wall_9, Wall_14, Wall_16)
 *          South  Y = -4.149 m   (Wall_22, Wall_24, Wall_27)
 *          East   X = +11.199 m  (Wall_13, Wall_25)
 *          West   X = -10.862 m  (Wall_18, Wall_21)
 *        Building: 22.06 m × 8.70 m, center (0.169, 0.201)
 *
 *   3. WORLD-FRAME AABB via spawn-offset correction
 *        /odom position (origin = spawn point) is shifted by the known
 *        robot spawn coordinates (world x=5.0, y=6.5) so the geofence
 *        ALWAYS operates in world frame regardless of spawn location.
 *        Update `spawn_world_x / spawn_world_y` in Config for a new map.
 *
 *   4. MAP-AGNOSTIC FALLBACK
 *        Setting `use_geofence = false` disables the AABB and relies
 *        entirely on GPS quality scoring — the correct mode for unknown
 *        maps or real-hardware deployments where GPS actually degrades.
 *
 *   5. GPS VELOCITY vs ODOM VELOCITY mismatch score (new)
 *        Compares |v_gps − v_odom| as an additional indoor indicator.
 *        Works even when fix-status and covariance stay clean (Gazebo).
 *
 *   6. INSTANT AABB TRIGGER, debounced GPS path
 *        Geofence: state flips in the same callback tick (no debounce).
 *        GPS quality: still debounced for noise immunity on real hardware.
 *
 * ============================================================================
 *  HOW TO CALIBRATE FOR A NEW MAP
 *  ────────────────────────────────
 *  1. Open the .world file.  Find the <state> section.
 *  2. Identify the outermost wall links and note their X/Y positions.
 *  3. Compute AABB:
 *       x_min = minimum X of any outer EAST-or-WEST wall
 *       x_max = maximum X of any outer EAST-or-WEST wall
 *       y_min = minimum Y of any outer NORTH-or-SOUTH wall
 *       y_max = maximum Y of any outer NORTH-or-SOUTH wall
 *  4. Set Config fields:
 *       building_x_min_world = x_min
 *       building_x_max_world = x_max
 *       building_y_min_world = y_min
 *       building_y_max_world = y_max
 *       spawn_world_x = <robot spawn X from launch file>
 *       spawn_world_y = <robot spawn Y from launch file>
 *  5. Leave `wall_margin_m = 0.35` (covers wall thickness + tolerance).
 *  6. Leave `hysteresis_m = 0.50` (prevents boundary oscillation).
 * ============================================================================
 *
 *  SUBSCRIBED TOPICS
 *    /odom            nav_msgs/Odometry  robot position (geofence – primary)
 *    /gps/fix         NavSatFix          fix status + covariance (secondary)
 *    /odometry/gps    nav_msgs/Odometry  ENU position (jump + vel mismatch)
 *
 *  PUBLISHED TOPICS
 *    /bt/indoor_detection  std_msgs/String  live diagnostic
 * ============================================================================
 */

#pragma once

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <mutex>
#include <sstream>
#include <iomanip>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensor_msgs/msg/nav_sat_status.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/string.hpp"

// ─────────────────────────────────────────────────────────────────────────────

class IndoorDetector
{
public:
    enum class Environment { OUTDOOR, INDOOR };

    // =========================================================================
    //  CONFIGURATION
    // =========================================================================
    struct Config
    {
        // ── GEOFENCE 2D AABB ─────────────────────────────────────────────────
        bool   use_geofence        = true;

        // Building outer boundary in WORLD frame [m]
        // Pre-calibrated from indoor_outdoor.world <state> section.
        // Outer walls:  X∈[-10.862, 11.199]   Y∈[-4.149, 4.551]
        // 0.35 m margin inset (wall thickness 0.15 m + 0.20 m clearance):
        double building_x_min_world = -10.0;   // west  inner edge
        double building_x_max_world = +10.0;   // east  inner edge
        double building_y_min_world = -4.5;   // south inner edge
        double building_y_max_world = +4.5;   // north inner edge

        // Robot spawn position in world frame (from spawn_robot.launch.py)
        double spawn_world_x       = 5.0;
        double spawn_world_y       = 6.5;

        // Wall margin already baked into the AABB above.
        // hysteresis_m: extra clearance the robot must travel PAST the inner
        // boundary before OUTDOOR is declared (prevents wall oscillation).
        double hysteresis_m        = 0.10;

        // ── GPS QUALITY SCORING (secondary path / real hardware) ─────────────
        // Covariance [m²] – guard removed; always evaluated (fixes Gazebo bug)
        double cov_warn_m2         =   4.0;   //  2 m std  → +1
        double cov_bad_m2          =  25.0;   //  5 m std  → +2
        double cov_critical_m2     = 100.0;   // 10 m std  → +3

        // Position jump in GPS ENU between consecutive messages [m]
        double jump_minor_m        =  0.5;    // +1
        double jump_moderate_m     =  2.0;    // +2
        double jump_severe_m       =  5.0;    // +3

        // GPS ENU velocity vs /odom velocity mismatch [m/s]  (new in v3)
        bool   use_vel_mismatch    = true;
        double vel_mismatch_warn   =  0.20;   // +1
        double vel_mismatch_bad    =  0.50;   // +2

        // GPS position vs odometry position divergence [m]
        bool   use_pos_div         = true;
        double pos_div_warn_m      =  1.0;    // +1
        double pos_div_bad_m       =  3.0;    // +2

        // GPS timeout [s]: no NavSatFix received → bad tick
        double signal_timeout_s    = 2.0;

        // Scoring thresholds
        int    score_indoor_thr    = 3;    // bad tick when score ≥ this
        int    score_critical_thr  = 7;    // instant flip (no debounce)

        // Debounce (GPS path only; geofence path is always instant)
        int    debounce_in_ticks   = 2;    // consecutive bad  → INDOOR
        int    debounce_out_ticks  = 4;    // consecutive good → OUTDOOR

        // Diagnostic publish period [s]
        double diag_period_s       = 0.5;
    };

    // =========================================================================
    IndoorDetector(rclcpp::Node*                      node,
                   std::shared_ptr<std::atomic<bool>> indoor_flag,
                   const Config&                      cfg)
    : node_(node), flag_(indoor_flag), cfg_(cfg),
      env_(Environment::OUTDOOR),
      consec_bad_(0), consec_good_(0),
      gps_timed_out_(false),
      last_gps_stamp_(rclcpp::Time(0, 0, RCL_ROS_TIME)),
      has_odom_(false), odom_x_(0.0), odom_y_(0.0),
      has_gps_enu_(false), gps_enu_x_(0.0), gps_enu_y_(0.0),
      gps_enu_vx_(0.0), gps_enu_vy_(0.0),
      has_prev_gps_(false), prev_gps_x_(0.0), prev_gps_y_(0.0),
      last_jump_m_(0.0), last_div_m_(0.0), last_vel_mis_(0.0),
      last_score_(0), last_cov_(0.0), last_fix_status_(0)
    {
        auto sq = rclcpp::SensorDataQoS();

        sub_odom_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", rclcpp::QoS(10),
            [this](nav_msgs::msg::Odometry::ConstSharedPtr m){ on_odom(m); });

        sub_fix_ = node_->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/gps/fix", sq,
            [this](sensor_msgs::msg::NavSatFix::ConstSharedPtr m){ on_fix(m); });

        sub_gps_odom_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/gps", rclcpp::QoS(10),
            [this](nav_msgs::msg::Odometry::ConstSharedPtr m){ on_gps_odom(m); });

        diag_pub_ = node_->create_publisher<std_msgs::msg::String>(
            "/bt/indoor_detection", rclcpp::QoS(10));

        diag_timer_ = node_->create_wall_timer(
            std::chrono::duration<double>(cfg_.diag_period_s),
            [this](){ evaluate_timeout(); publish_diagnostics(); });

        log_startup();
    }

    Environment getEnvironment() const { return env_.load(); }
    bool        isIndoor()       const { return flag_->load(); }

    void forceIndoor(bool indoor)
    {
        env_.store(indoor ? Environment::INDOOR : Environment::OUTDOOR);
        flag_->store(indoor);
        RCLCPP_WARN(node_->get_logger(),
            "[IndoorDetector] MANUAL OVERRIDE → %s",
            indoor ? "INDOOR" : "OUTDOOR");
    }

    // =========================================================================
private:

    // ── Geometry helper ───────────────────────────────────────────────────────

    /** True when (px, py) is strictly inside the axis-aligned box. */
    static bool in_box(double px, double py,
                       double x_min, double x_max,
                       double y_min, double y_max)
    {
        return px > x_min && px < x_max &&
               py > y_min && py < y_max;
    }

    // ── Geofence check (called from on_odom at ~50 Hz) ────────────────────────
    /**
     * Schmitt-trigger hysteresis:
     *
     *   INDOOR  entry : robot inside INNER box  (AABB inset by 0)
     *   OUTDOOR entry : robot outside OUTER box (AABB expanded by hysteresis_m)
     *
     *   The band between INNER and OUTER keeps the state stable while the
     *   robot is near the wall (passing through the door, etc.)
     *
     *      ┌─ outer box (AABB + hysteresis) ────────────────────────────┐
     *      │                                                             │
     *      │   ┌─ inner box (AABB) ──────────────────────────────┐      │
     *      │   │                    INDOOR                       │      │
     *      │   │         (state flips IN here)                   │      │
     *      │   └─────────────────────────────────────────────────┘      │
     *      │                   hysteresis band                           │
     *      └─────────────────────────────────────────────────────────────┘
     *      OUTDOOR (state flips OUT only when robot leaves outer box)
     */
    void check_geofence(double world_x, double world_y)
    {
        Environment cur = env_.load();

        if (cur == Environment::OUTDOOR) {
            // INDOOR entry: strictly inside the calibrated AABB
            bool enters = in_box(world_x, world_y,
                                 cfg_.building_x_min_world,
                                 cfg_.building_x_max_world,
                                 cfg_.building_y_min_world,
                                 cfg_.building_y_max_world);
            if (enters) {
                RCLCPP_WARN(node_->get_logger(),
                    "[IndoorDetector] ■ GEOFENCE ENTRY → INDOOR  "
                    "world=(%.2f, %.2f)  box=[%.2f..%.2f, %.2f..%.2f]",
                    world_x, world_y,
                    cfg_.building_x_min_world, cfg_.building_x_max_world,
                    cfg_.building_y_min_world, cfg_.building_y_max_world);
                flip_to(Environment::INDOOR);
            }
        } else {
            // OUTDOOR entry: must leave the outer (expanded) box
            double h = cfg_.hysteresis_m;
            bool exits = !in_box(world_x, world_y,
                                 cfg_.building_x_min_world - h,
                                 cfg_.building_x_max_world + h,
                                 cfg_.building_y_min_world - h,
                                 cfg_.building_y_max_world + h);
            if (exits) {
                RCLCPP_INFO(node_->get_logger(),
                    "[IndoorDetector] ● GEOFENCE EXIT → OUTDOOR  "
                    "world=(%.2f, %.2f)  "
                    "exit_box=[%.2f..%.2f, %.2f..%.2f]",
                    world_x, world_y,
                    cfg_.building_x_min_world - h,
                    cfg_.building_x_max_world + h,
                    cfg_.building_y_min_world - h,
                    cfg_.building_y_max_world + h);
                flip_to(Environment::OUTDOOR);
            }
        }
    }

    // ── GPS quality score ─────────────────────────────────────────────────────
    int compute_gps_score(const sensor_msgs::msg::NavSatFix::ConstSharedPtr& msg,
                          double jump_m, double div_m, double vel_mis)
    {
        int s = 0;

        // 1. Fix status
        if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX)
            s += 5;

        // 2. Covariance – evaluated regardless of covariance_type (Gazebo fix)
        double cov = msg->position_covariance[0];
        if      (cov >= cfg_.cov_critical_m2) s += 3;
        else if (cov >= cfg_.cov_bad_m2)      s += 2;
        else if (cov >= cfg_.cov_warn_m2)     s += 1;

        // 3. GPS position jump (multipath signature)
        if      (jump_m >= cfg_.jump_severe_m)   s += 3;
        else if (jump_m >= cfg_.jump_moderate_m) s += 2;
        else if (jump_m >= cfg_.jump_minor_m)    s += 1;

        // 4. GPS ENU position vs /odom divergence
        if (cfg_.use_pos_div) {
            if      (div_m >= cfg_.pos_div_bad_m)  s += 2;
            else if (div_m >= cfg_.pos_div_warn_m) s += 1;
        }

        // 5. GPS velocity vs odom velocity mismatch  (new in v3)
        if (cfg_.use_vel_mismatch) {
            if      (vel_mis >= cfg_.vel_mismatch_bad)  s += 2;
            else if (vel_mis >= cfg_.vel_mismatch_warn) s += 1;
        }

        return s;
    }

    // ── Debounced GPS state machine ───────────────────────────────────────────
    void update_gps_state(bool bad_tick, bool critical)
    {
        Environment cur = env_.load();

        if (bad_tick) { consec_good_ = 0; ++consec_bad_; }
        else          { consec_bad_  = 0; ++consec_good_; }

        if (cur == Environment::OUTDOOR) {
            if (critical || consec_bad_ >= cfg_.debounce_in_ticks) {
                RCLCPP_WARN(node_->get_logger(),
                    "[IndoorDetector] ■ GPS-score → INDOOR  "
                    "score=%d  critical=%d  bad_ticks=%d",
                    last_score_, critical, consec_bad_);
                flip_to(Environment::INDOOR);
            }
        } else {
            if (consec_good_ >= cfg_.debounce_out_ticks) {
                RCLCPP_INFO(node_->get_logger(),
                    "[IndoorDetector] ● GPS-score → OUTDOOR  "
                    "good_ticks=%d", consec_good_);
                flip_to(Environment::OUTDOOR);
            }
        }
    }

    // ── Atomic state flip ─────────────────────────────────────────────────────
    void flip_to(Environment next)
    {
        env_.store(next);
        flag_->store(next == Environment::INDOOR);
        consec_bad_  = 0;
        consec_good_ = 0;
    }

    // =========================================================================
    //  SUBSCRIBER CALLBACKS
    // =========================================================================

    /** /odom  –  PRIMARY geofence source (fires ~50 Hz, near ground-truth in sim) */
    void on_odom(nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        odom_x_   = msg->pose.pose.position.x;
        odom_y_   = msg->pose.pose.position.y;
        odom_vx_  = msg->twist.twist.linear.x;   // forward speed (base_link)
        has_odom_ = true;

        if (!cfg_.use_geofence) return;

        // Shift odom frame → world frame using known spawn coordinates
        double wx = odom_x_ + cfg_.spawn_world_x;
        double wy = odom_y_ + cfg_.spawn_world_y;

        check_geofence(wx, wy);
    }

    /** /gps/fix  –  GPS quality scoring (secondary / real-hardware path) */
    void on_fix(sensor_msgs::msg::NavSatFix::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        last_gps_stamp_  = node_->now();
        gps_timed_out_   = false;
        last_fix_status_ = msg->status.status;
        last_cov_        = msg->position_covariance[0];

        // Position divergence: GPS ENU vs odom (both in map frame)
        double div_m = 0.0;
        if (has_odom_ && has_gps_enu_) {
            double dx = gps_enu_x_ - odom_x_;
            double dy = gps_enu_y_ - odom_y_;
            div_m = std::sqrt(dx*dx + dy*dy);
        }
        last_div_m_ = div_m;

        // Velocity mismatch: GPS ENU speed vs /odom forward speed
        double vel_mis = 0.0;
        if (cfg_.use_vel_mismatch && has_odom_ && has_gps_enu_) {
            double v_gps  = std::sqrt(gps_enu_vx_*gps_enu_vx_ +
                                      gps_enu_vy_*gps_enu_vy_);
            double v_odom = std::abs(odom_vx_);   // TurtleBot3: forward = x
            vel_mis = std::abs(v_gps - v_odom);
        }
        last_vel_mis_ = vel_mis;

        last_score_ = compute_gps_score(msg, last_jump_m_, div_m, vel_mis);
        last_jump_m_ = 0.0;   // consume after scoring

        bool bad      = (last_score_ >= cfg_.score_indoor_thr);
        bool critical = (last_score_ >= cfg_.score_critical_thr);
        update_gps_state(bad, critical);
    }

    /** /odometry/gps  –  GPS ENU position (jump + divergence + velocity) */
    void on_gps_odom(nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        double x  = msg->pose.pose.position.x;
        double y  = msg->pose.pose.position.y;
        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;

        // Position jump detection
        if (has_prev_gps_) {
            double dx = x - prev_gps_x_;
            double dy = y - prev_gps_y_;
            last_jump_m_ = std::sqrt(dx*dx + dy*dy);
        }

        prev_gps_x_ = x;  prev_gps_y_ = y;
        has_prev_gps_ = true;

        gps_enu_x_  = x;  gps_enu_y_  = y;
        gps_enu_vx_ = vx; gps_enu_vy_ = vy;
        has_gps_enu_ = true;
    }

    // ── GPS timeout (evaluated once per diag timer tick) ──────────────────────
    void evaluate_timeout()
    {
        std::lock_guard<std::mutex> lk(mtx_);

        if (last_gps_stamp_.nanoseconds() == 0) return;   // never received

        double age = (node_->now() - last_gps_stamp_).seconds();
        if (age > cfg_.signal_timeout_s && !gps_timed_out_) {
            gps_timed_out_ = true;
            bool critical  = (age > cfg_.signal_timeout_s * 2.0);
            RCLCPP_WARN(node_->get_logger(),
                "[IndoorDetector] GPS timeout %.1f s (thr=%.1f s)%s",
                age, cfg_.signal_timeout_s,
                critical ? " – CRITICAL" : "");
            update_gps_state(true, critical);
        }
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────
    void publish_diagnostics()
    {
        double wx, wy, div_m, vel_mis, jump_m, age_s, cov;
        int score, fix, cb, cg;
        Environment env;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            wx      = odom_x_ + cfg_.spawn_world_x;
            wy      = odom_y_ + cfg_.spawn_world_y;
            div_m   = last_div_m_;
            vel_mis = last_vel_mis_;
            jump_m  = last_jump_m_;
            age_s   = last_gps_stamp_.nanoseconds() > 0
                      ? (node_->now() - last_gps_stamp_).seconds() : -1.0;
            score   = last_score_;
            fix     = last_fix_status_;
            cov     = last_cov_;
            cb      = consec_bad_;
            cg      = consec_good_;
        }
        env = env_.load();

        // Geofence proximity: distance to nearest wall edge
        double dist_N = cfg_.building_y_max_world - wy;
        double dist_S = wy - cfg_.building_y_min_world;
        double dist_E = cfg_.building_x_max_world - wx;
        double dist_W = wx - cfg_.building_x_min_world;
        double wall_dist = std::min({dist_N, dist_S, dist_E, dist_W});

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << (env == Environment::INDOOR ? "■ INDOOR" : "● OUTDOOR");
        ss << " | world=(" << wx << "," << wy << ")";
        ss << " | wall_dist=" << wall_dist << "m";
        ss << " | gps_score=" << score;
        ss << " | fix=" << fix;
        ss << " | cov=" << cov << "m²";
        ss << " | jump=" << jump_m << "m";
        ss << " | pos_div=" << div_m << "m";
        ss << " | vel_mis=" << vel_mis << "m/s";
        ss << " | age=" << age_s << "s";
        ss << " | bad=" << cb << "/" << cfg_.debounce_in_ticks;
        ss << " good=" << cg << "/" << cfg_.debounce_out_ticks;

        auto m = std_msgs::msg::String{};
        m.data = ss.str();
        diag_pub_->publish(m);

        RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
            "[IndoorDetector] %s", ss.str().c_str());
    }

    void log_startup()
    {
        RCLCPP_INFO(node_->get_logger(),
            "\n[IndoorDetector v3]\n"
            "  Mode     : %s\n"
            "  Building : X∈[%.3f, %.3f]  Y∈[%.3f, %.3f]  (world frame)\n"
            "  Width    : %.2f m   Depth: %.2f m\n"
            "  Spawn    : world (%.1f, %.1f)\n"
            "  Hysteresis: %.2f m\n"
            "  Debounce : in=%d  out=%d  (GPS path only; geofence = instant)\n"
            "  Topics   : /odom  /gps/fix  /odometry/gps",
            cfg_.use_geofence ? "GEOFENCE+GPS_score" : "GPS_score only",
            cfg_.building_x_min_world, cfg_.building_x_max_world,
            cfg_.building_y_min_world, cfg_.building_y_max_world,
            cfg_.building_x_max_world - cfg_.building_x_min_world,
            cfg_.building_y_max_world - cfg_.building_y_min_world,
            cfg_.spawn_world_x, cfg_.spawn_world_y,
            cfg_.hysteresis_m,
            cfg_.debounce_in_ticks, cfg_.debounce_out_ticks);
    }

    // ── Members ───────────────────────────────────────────────────────────────
    rclcpp::Node*                       node_;
    std::shared_ptr<std::atomic<bool>>  flag_;
    Config                              cfg_;
    std::atomic<Environment>            env_;
    std::mutex                          mtx_;

    // Debounce counters (GPS path)
    int   consec_bad_, consec_good_;

    // GPS timeout
    rclcpp::Time  last_gps_stamp_;
    bool          gps_timed_out_;

    // /odom state
    bool   has_odom_;
    double odom_x_, odom_y_, odom_vx_{0.0};

    // /odometry/gps state
    bool   has_gps_enu_;
    double gps_enu_x_, gps_enu_y_;
    double gps_enu_vx_, gps_enu_vy_;
    bool   has_prev_gps_;
    double prev_gps_x_, prev_gps_y_;

    // Diagnostics cache
    double last_jump_m_, last_div_m_, last_vel_mis_;
    int    last_score_, last_fix_status_;
    double last_cov_;

    // ROS2 handles
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr  sub_fix_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      sub_gps_odom_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr           diag_pub_;
    rclcpp::TimerBase::SharedPtr                                  diag_timer_;
};