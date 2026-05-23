#!/usr/bin/env python3
import argparse
import os
import sys
import math

import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


def quaternion_to_yaw(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(angle):
    if isinstance(angle, np.ndarray):
        return np.arctan2(np.sin(angle), np.cos(angle))
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_odometry_topics(bag_dir, topics):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    missing = [topic for topic in topics if topic not in topic_types]
    if missing:
        raise ValueError(f"Topic(s) not found in bag: {', '.join(missing)}")

    traces = {
        topic: {
            'times': [],
            'positions': [],
            'yaws': []
        }
        for topic in topics
    }

    type_cache = {}
    while reader.has_next():
        topic_name, serialized_data, time_stamp = reader.read_next()
        if topic_name not in traces:
            continue
        if topic_types[topic_name] not in type_cache:
            type_cache[topic_types[topic_name]] = get_message(topic_types[topic_name])
        msg_type = type_cache[topic_types[topic_name]]
        msg = deserialize_message(serialized_data, msg_type)

        if not hasattr(msg, 'pose') or not hasattr(msg.pose, 'pose'):
            continue

        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation

        traces[topic_name]['times'].append(int(time_stamp))
        traces[topic_name]['positions'].append((pos.x, pos.y, pos.z))
        traces[topic_name]['yaws'].append(quaternion_to_yaw(orient))

    for topic in topics:
        traces[topic]['times'] = np.array(traces[topic]['times'], dtype=np.int64)
        traces[topic]['positions'] = np.array(traces[topic]['positions'], dtype=np.float64)
        traces[topic]['yaws'] = np.array(traces[topic]['yaws'], dtype=np.float64)

    return traces


def align_trace(reference, target, tolerance_ns):
    if len(reference['times']) == 0 or len(target['times']) == 0:
        return np.empty((0, 3)), np.empty((0,))

    reference_times = reference['times']
    target_times = target['times']
    reference_positions = reference['positions']
    target_positions = target['positions']
    reference_yaws = reference['yaws']
    target_yaws = target['yaws']

    target_indexes = np.searchsorted(reference_times, target_times)
    matched_ref = []
    matched_target = []

    for i, target_time in enumerate(target_times):
        candidates = []
        idx = target_indexes[i]
        if idx < len(reference_times):
            candidates.append((abs(reference_times[idx] - target_time), idx))
        if idx > 0:
            candidates.append((abs(reference_times[idx - 1] - target_time), idx - 1))
        if not candidates:
            continue
        best_dt, best_idx = min(candidates, key=lambda x: x[0])
        if best_dt <= tolerance_ns:
            matched_ref.append(best_idx)
            matched_target.append(i)

    if len(matched_ref) == 0:
        return np.empty((0, 3)), np.empty((0,))

    ref_positions = reference_positions[matched_ref]
    targ_positions = target_positions[matched_target]
    ref_yaws = reference_yaws[matched_ref]
    targ_yaws = target_yaws[matched_target]

    deltas = targ_positions - ref_positions
    yaw_errors = normalize_angle(targ_yaws - ref_yaws)
    return deltas, yaw_errors


def compute_rmse(errors):
    return math.sqrt(np.mean(np.square(errors))) if len(errors) > 0 else float('nan')


def build_report(reference_topic, traces, compare_topics, tolerance_ns, plot):
    reference = traces[reference_topic]
    metrics = []

    for topic in compare_topics:
        if topic == reference_topic:
            continue
        if topic not in traces:
            continue

        deltas, yaw_errors = align_trace(reference, traces[topic], tolerance_ns)
        if deltas.size == 0:
            metrics.append((topic, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0))
            continue

        rmse_total = compute_rmse(np.linalg.norm(deltas[:, :3], axis=1))
        rmse_xy = compute_rmse(np.linalg.norm(deltas[:, :2], axis=1))
        rmse_x = compute_rmse(deltas[:, 0])
        rmse_y = compute_rmse(deltas[:, 1])
        rmse_z = compute_rmse(deltas[:, 2])
        rmse_yaw = compute_rmse(yaw_errors)
        metrics.append((topic, rmse_total, rmse_xy, rmse_x, rmse_y, rmse_z, rmse_yaw, len(deltas)))

    if plot:
        plot_trajectories(reference_topic, traces, compare_topics)

    return metrics


def plot_trajectories(reference_topic, traces, compare_topics):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib non è installato: skipping plot.', file=sys.stderr)
        return

    plt.figure(figsize=(10, 8))
    ref_positions = traces[reference_topic]['positions']
    plt.plot(ref_positions[:, 0], ref_positions[:, 1], label=f'{reference_topic} (reference)', linewidth=3)

    for topic in compare_topics:
        if topic == reference_topic or topic not in traces:
            continue
        positions = traces[topic]['positions']
        plt.plot(positions[:, 0], positions[:, 1], label=topic, linewidth=1)

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Traiettorie confronto EKF')
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Valutazione traiettoria e RMSE per EKF da rosbag2')
    parser.add_argument('--bag', required=True, help='Directory del rosbag2 (es. Traiettoria7)')
    parser.add_argument('--ref', default='/odom', help='Topic di riferimento per il confronto (default /odom)')
    parser.add_argument('--topics', nargs='+', default=['/odometry/global', '/odometry/global2', '/odometry/fused'], help='Topic EKF da valutare')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Tolleranza di allineamento in secondi (default 0.1)')
    parser.add_argument('--plot', action='store_true', help='Mostra il grafico delle traiettorie')
    return parser.parse_args()


def main():
    args = parse_args()

    bag_dir = args.bag
    if not os.path.isdir(bag_dir):
        print(f"Errore: la directory del bag '{bag_dir}' non esiste.", file=sys.stderr)
        sys.exit(1)

    topics = [args.ref] + args.topics
    traces = read_odometry_topics(bag_dir, topics)
    metrics = build_report(args.ref, traces, args.topics, int(args.tolerance * 1e9), args.plot)

    print('\nRisultati RMSE:')
    print('Reference:', args.ref)
    print('Topic'.ljust(25), 'RMSE_3D', 'RMSE_XY', 'RMSE_X', 'RMSE_Y', 'RMSE_Z', 'RMSE_YAW', 'N')
    for topic, rmse_total, rmse_xy, rmse_x, rmse_y, rmse_z, rmse_yaw, n in metrics:
        print(f'{topic.ljust(25)} {rmse_total:8.4f} {rmse_xy:8.4f} {rmse_x:8.4f} {rmse_y:8.4f} {rmse_z:8.4f} {rmse_yaw:8.4f} {n:5d}')

    print('\nI valori RMSE sono riferiti alla posizione (x,y,z) e all’orientamento yaw.')


if __name__ == '__main__':
    main()
