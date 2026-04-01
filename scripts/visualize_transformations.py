#!/usr/bin/env python3
"""
Visualize the operator transformation pipeline from logged data.
Shows the complete transformation flow from hand motion to robot commands.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_log_data(log_path):
    """Load transformation pipeline data from JSON log file."""
    with open(log_path, "r") as f:
        data = json.load(f)
    return data


def has_nan(arr):
    """Check if numpy array contains NaN values."""
    if arr is None:
        return True
    return np.any(np.isnan(arr))


def filter_nan_frames(frames, max_nan_ratio=0.5):
    """Filter out frames with too many NaN values.

    Args:
        frames: List of frame dictionaries
        max_nan_ratio: Maximum ratio of NaN values allowed (0.5 = 50%)
    """
    valid_frames = []
    for frame in frames:
        nan_count = 0
        total_values = 0

        # Check all arrays in the frame
        for section in ["input", "output", "intermediate"]:
            if section in frame:
                for key, value in frame[section].items():
                    arr = np.array(value)
                    total_values += arr.size
                    nan_count += np.sum(np.isnan(arr))

        # Skip frames with too many NaN values
        if total_values > 0 and (nan_count / total_values) < max_nan_ratio:
            valid_frames.append(frame)

    return valid_frames


def safe_convert_to_list(arr):
    """Safely convert numpy array to list, ignoring NaN values."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        # Replace NaN with None for JSON compatibility
        arr_copy = arr.copy()
        mask = np.isnan(arr_copy)
        arr_copy[mask] = None
        return arr_copy.tolist()
    return arr


def extract_frames(data):
    """Extract and sort frames from log data."""
    frames = []
    for frame_id, frame_data in data["frames"].items():
        frames.append((int(frame_id), frame_data))
    frames.sort(key=lambda x: x[0])
    return [f[1] for f in frames]


def draw_coordinate_frame(ax, origin, rotation_matrix, scale=0.1, label=None, linewidth=2):
    """Draw a coordinate frame (X=red, Y=green, Z=blue)."""
    x_axis = rotation_matrix @ np.array([scale, 0, 0])
    y_axis = rotation_matrix @ np.array([0, scale, 0])
    z_axis = rotation_matrix @ np.array([0, 0, scale])

    # X axis (red)
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        x_axis[0],
        x_axis[1],
        x_axis[2],
        color="r",
        linewidth=linewidth,
        label=label if label else "",
    )
    # Y axis (green)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color="g", linewidth=linewidth)
    # Z axis (blue)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color="b", linewidth=linewidth)


def plot_transformation_frame(frame_data, ax, frame_idx, x_bounds=None, y_bounds=None, z_bounds=None):
    """Plot a single frame of the transformation pipeline."""
    ax.clear()
    ax.set_title(f"Transformation Pipeline - Frame {frame_idx}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Extract data
    input_data = frame_data.get("input", {})
    intermediate = frame_data.get("intermediate", {})
    output = frame_data.get("output", {})

    # Convert to numpy arrays
    hand_init_h = np.array(input_data.get("hand_init_h", [[np.nan] * 3] * 4))
    robot_init_h = np.array(input_data.get("robot_init_h", [[np.nan] * 3] * 4))
    hand_moving_h = np.array(input_data.get("hand_moving_h", [[np.nan] * 3] * 4))

    h_ht_hi = np.array(intermediate.get("h_ht_hi", [[np.nan] * 3] * 4))
    relative_affine = np.array(intermediate.get("relative_affine_in_robot_frame", [[np.nan] * 3] * 4))
    robot_moving_h = np.array(output.get("robot_moving_h", [[np.nan] * 3] * 4))

    cart_target_raw = np.array(output.get("cart_target_raw", [np.nan] * 7))
    cart_target_filtered = np.array(output.get("cart_target_filtered", [np.nan] * 7))

    # Skip drawing if critical data is NaN
    if (
        has_nan(hand_init_h[:3, 3])
        or has_nan(robot_init_h[:3, 3])
        or has_nan(hand_moving_h[:3, 3])
        or has_nan(robot_moving_h[:3, 3])
        or has_nan(cart_target_raw[:3])
        or has_nan(cart_target_filtered[:3])
    ):
        ax.text(0, 0, 0, "Frame contains NaN values", fontsize=12, ha="center")
        return

    # Draw trajectory
    ax.plot(
        [hand_init_h[0, 3], hand_moving_h[0, 3]],
        [hand_init_h[1, 3], hand_moving_h[1, 3]],
        [hand_init_h[2, 3], hand_moving_h[2, 3]],
        "r--",
        linewidth=2,
        alpha=0.6,
        label="Hand trajectory",
    )

    ax.plot(
        [robot_init_h[0, 3], robot_moving_h[0, 3]],
        [robot_init_h[1, 3], robot_moving_h[1, 3]],
        [robot_init_h[2, 3], robot_moving_h[2, 3]],
        "b-",
        linewidth=2,
        alpha=0.8,
        label="Robot trajectory",
    )

    # Draw coordinate frames
    # Initial hand frame (orange)
    draw_coordinate_frame(ax, hand_init_h[:3, 3], hand_init_h[:3, :3], scale=0.05, label="Initial Hand", linewidth=3)

    # Current hand frame (red, hollow)
    ax.scatter(*hand_moving_h[:3, 3], c="r", s=100, marker="o", edgecolors="black", linewidth=2, label="Current Hand")

    # Robot initial frame (green)
    draw_coordinate_frame(ax, robot_init_h[:3, 3], robot_init_h[:3, :3], scale=0.05, label="Initial Robot", linewidth=3)

    # Robot target frame (blue, hollow)
    ax.scatter(*robot_moving_h[:3, 3], c="b", s=100, marker="s", edgecolors="black", linewidth=2, label="Robot Target")

    # Filtered position (star)
    ax.scatter(
        *cart_target_filtered[:3], c="purple", s=150, marker="*", edgecolors="black", linewidth=2, label="Filtered Pose"
    )

    # Raw position (x)
    ax.scatter(*cart_target_raw[:3], c="magenta", s=80, marker="x", linewidth=2, label="Raw Pose")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set fixed axis bounds if provided
    if x_bounds and y_bounds and z_bounds:
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_zlim(z_bounds)

    # Add legend (only once)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.05, 1))

    # Add metadata
    metadata = frame_data.get("metadata", {})
    info_text = f"Resolution Scale: {metadata.get('resolution_scale', 1.0):.2f}\n"
    info_text += f"Timestamp: {metadata.get('timestamp', 0):.2f}"
    ax.text2D(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def plot_transformation_comparison(frames, output_path):
    """Create a static comparison plot of all frames with 2D diagrams."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Collect all positions for trajectory plotting
    hand_positions = []
    robot_positions = []
    raw_positions = []
    filtered_positions = []
    relative_vr_translations = []
    relative_robot_translations = []
    valid_frames = []

    for frame_idx, frame in enumerate(frames):
        input_data = frame.get("input", {})
        output = frame.get("output", {})
        intermediate = frame.get("intermediate", {})

        hand_moving = np.array(input_data.get("hand_moving_h", [[np.nan] * 3] * 4))
        robot_moving = np.array(output.get("robot_moving_h", [[np.nan] * 3] * 4))
        cart_raw = np.array(output.get("cart_target_raw", [np.nan] * 7))
        cart_filtered = np.array(output.get("cart_target_filtered", [np.nan] * 7))
        h_ht_hi = np.array(intermediate.get("h_ht_hi", [[np.nan] * 3] * 4))
        h_ht_hi_t = np.array(intermediate.get("h_ht_hi_t", [np.nan] * 3))

        # Skip if any critical data is NaN
        if (
            has_nan(hand_moving[:3, 3])
            or has_nan(robot_moving[:3, 3])
            or has_nan(cart_raw[:3])
            or has_nan(cart_filtered[:3])
            or has_nan(h_ht_hi[:3, 3])
            or has_nan(h_ht_hi_t)
        ):
            continue

        hand_positions.append(hand_moving[:3, 3])
        robot_positions.append(robot_moving[:3, 3])
        raw_positions.append(cart_raw[:3])
        filtered_positions.append(cart_filtered[:3])
        relative_vr_translations.append(h_ht_hi[:3, 3])
        relative_robot_translations.append(h_ht_hi_t)
        valid_frames.append(frame_idx)

    if len(hand_positions) == 0:
        print("Warning: No valid frames found for comparison plot (all data is NaN)")
        plt.close()
        return

    hand_positions = np.array(hand_positions)
    robot_positions = np.array(robot_positions)
    raw_positions = np.array(raw_positions)
    filtered_positions = np.array(filtered_positions)
    relative_vr_translations = np.array(relative_vr_translations)
    relative_robot_translations = np.array(relative_robot_translations)

    if len(hand_positions) == 0:
        print("Warning: No valid frames found for comparison plot (all data is NaN)")
        plt.close()
        return

    hand_positions = np.array(hand_positions)
    robot_positions = np.array(robot_positions)
    raw_positions = np.array(raw_positions)
    filtered_positions = np.array(filtered_positions)

    # Transform robot values: z->y, x->z, y->x
    # robot_positions = robot_positions[:, [2, 0, 1]]
    # raw_positions = raw_positions[:, [2, 0, 1]]
    # filtered_positions = filtered_positions[:, [2, 0, 1]]

    # Plot X vs Y (subplot 1, top-left)
    ax1 = axes[0, 0]
    ax1.plot(hand_positions[:, 0], hand_positions[:, 1], "r--", linewidth=2, alpha=0.7, label="Hand Trajectory")
    ax1.plot(robot_positions[:, 0], robot_positions[:, 1], "b-", linewidth=2, alpha=0.7, label="Robot Trajectory")
    ax1.plot(filtered_positions[:, 0], filtered_positions[:, 1], "purple", linewidth=1.5, alpha=0.8, label="Filtered XY")

    # Plot start and end positions
    ax1.scatter(
        hand_positions[0, 0],
        hand_positions[0, 1],
        c="green",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="Start",
    )
    ax1.scatter(
        robot_positions[0, 0], robot_positions[0, 1], c="green", s=100, marker="s", edgecolors="black", linewidth=2
    )

    ax1.scatter(
        hand_positions[-1, 0],
        hand_positions[-1, 1],
        c="red",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="End",
    )
    ax1.scatter(
        robot_positions[-1, 0], robot_positions[-1, 1], c="red", s=100, marker="s", edgecolors="black", linewidth=2
    )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("X vs Y Plane")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Plot Z over time (subplot 2, top-right)
    ax2 = axes[0, 1]
    ax2.plot(valid_frames, hand_positions[:, 2], "r--", linewidth=2, alpha=0.7, label="Hand Z")
    ax2.plot(valid_frames, robot_positions[:, 2], "b-", linewidth=2, alpha=0.7, label="Robot Z")
    ax2.plot(valid_frames, raw_positions[:, 2], "m:", linewidth=1.5, alpha=0.5, label="Raw Z")
    ax2.plot(valid_frames, filtered_positions[:, 2], "purple", linewidth=2, alpha=0.8, label="Filtered Z")

    # Plot start and end positions
    ax2.scatter(
        valid_frames[0],
        hand_positions[0, 2],
        c="green",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="Start",
    )
    ax2.scatter(valid_frames[0], robot_positions[0, 2], c="green", s=100, marker="s", edgecolors="black", linewidth=2)
    ax2.scatter(
        valid_frames[-1],
        hand_positions[-1, 2],
        c="red",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="End",
    )
    ax2.scatter(valid_frames[-1], robot_positions[-1, 2], c="red", s=100, marker="s", edgecolors="black", linewidth=2)

    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Z Position Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot X over time (subplot 3, bottom-left)
    ax3 = axes[1, 0]
    ax3.plot(valid_frames, hand_positions[:, 0], "r--", linewidth=2, alpha=0.7, label="Hand X")
    ax3.plot(valid_frames, robot_positions[:, 0], "b-", linewidth=2, alpha=0.7, label="Robot X")
    ax3.plot(valid_frames, raw_positions[:, 0], "m:", linewidth=1.5, alpha=0.5, label="Raw X")
    ax3.plot(valid_frames, filtered_positions[:, 0], "purple", linewidth=2, alpha=0.8, label="Filtered X")

    # Plot start and end positions
    ax3.scatter(
        valid_frames[0],
        hand_positions[0, 0],
        c="green",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="Start",
    )
    ax3.scatter(valid_frames[0], robot_positions[0, 0], c="green", s=100, marker="s", edgecolors="black", linewidth=2)
    ax3.scatter(
        valid_frames[-1],
        hand_positions[-1, 0],
        c="red",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="End",
    )
    ax3.scatter(valid_frames[-1], robot_positions[-1, 0], c="red", s=100, marker="s", edgecolors="black", linewidth=2)

    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("X (m)")
    ax3.set_title("X Position Over Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot Y over time (subplot 4, bottom-right)
    ax4 = axes[1, 1]
    ax4.plot(valid_frames, hand_positions[:, 1], "r--", linewidth=2, alpha=0.7, label="Hand Y")
    ax4.plot(valid_frames, robot_positions[:, 1], "b-", linewidth=2, alpha=0.7, label="Robot Y")
    ax4.plot(valid_frames, raw_positions[:, 1], "m:", linewidth=1.5, alpha=0.5, label="Raw Y")
    ax4.plot(valid_frames, filtered_positions[:, 1], "purple", linewidth=2, alpha=0.8, label="Filtered Y")

    # Plot start and end positions
    ax4.scatter(
        valid_frames[0],
        hand_positions[0, 1],
        c="green",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="Start",
    )
    ax4.scatter(valid_frames[0], robot_positions[0, 1], c="green", s=100, marker="s", edgecolors="black", linewidth=2)
    ax4.scatter(
        valid_frames[-1],
        hand_positions[-1, 1],
        c="red",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="End",
    )
    ax4.scatter(valid_frames[-1], robot_positions[-1, 1], c="red", s=100, marker="s", edgecolors="black", linewidth=2)

    ax4.set_xlabel("Frame Index")
    ax4.set_ylabel("Y (m)")
    ax4.set_title("Y Position Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot relative translation in VR space (subplot 5, bottom-left)
    ax5 = axes[2, 0]
    ax5.plot(valid_frames, relative_vr_translations[:, 0], "r-", linewidth=2, alpha=0.7, label="VR Relative X")
    ax5.plot(valid_frames, relative_vr_translations[:, 1], "g-", linewidth=2, alpha=0.7, label="VR Relative Y")
    ax5.plot(valid_frames, relative_vr_translations[:, 2], "b-", linewidth=2, alpha=0.7, label="VR Relative Z")
    ax5.set_xlabel("Frame Index")
    ax5.set_ylabel("Displacement (m)")
    ax5.set_title("Relative Translation in VR Space")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot relative translation in robot space (subplot 6, bottom-right)
    ax6 = axes[2, 1]
    ax6.plot(valid_frames, relative_robot_translations[:, 0], "r-", linewidth=2, alpha=0.7, label="Robot Relative X")
    ax6.plot(valid_frames, relative_robot_translations[:, 1], "g-", linewidth=2, alpha=0.7, label="Robot Relative Y")
    ax6.plot(valid_frames, relative_robot_translations[:, 2], "b-", linewidth=2, alpha=0.7, label="Robot Relative Z")
    ax6.set_xlabel("Frame Index")
    ax6.set_ylabel("Displacement (m)")
    ax6.set_title("Relative Translation in Robot Space")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Static comparison plot saved to {output_path}")


def calculate_global_bounds(frames):
    """Calculate global min/max bounds across all frames for fixed axis."""
    all_positions = []

    for frame in frames:
        input_data = frame.get("input", {})
        output = frame.get("output", {})

        hand_init = np.array(input_data.get("hand_init_h", [[np.nan] * 3] * 4))
        hand_moving = np.array(input_data.get("hand_moving_h", [[np.nan] * 3] * 4))
        robot_init = np.array(input_data.get("robot_init_h", [[np.nan] * 3] * 4))
        robot_moving = np.array(output.get("robot_moving_h", [[np.nan] * 3] * 4))
        cart_raw = np.array(output.get("cart_target_raw", [np.nan] * 7))
        cart_filtered = np.array(output.get("cart_target_filtered", [np.nan] * 7))

        # Collect all positions if they're not NaN
        if not has_nan(hand_init[:3, 3]):
            all_positions.append(hand_init[:3, 3])
        if not has_nan(hand_moving[:3, 3]):
            all_positions.append(hand_moving[:3, 3])
        if not has_nan(robot_init[:3, 3]):
            all_positions.append(robot_init[:3, 3])
        if not has_nan(robot_moving[:3, 3]):
            all_positions.append(robot_moving[:3, 3])
        if not has_nan(cart_raw[:3]):
            all_positions.append(cart_raw[:3])
        if not has_nan(cart_filtered[:3]):
            all_positions.append(cart_filtered[:3])

    if not all_positions:
        # Default bounds if no valid data
        return (-1, 1), (-1, 1), (-1, 1)

    all_positions = np.array(all_positions)

    # Calculate min/max for each axis with some padding
    padding = 0.1
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])

    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_bounds = (x_min - padding * x_range, x_max + padding * x_range)
    y_bounds = (y_min - padding * y_range, y_max + padding * y_range)
    z_bounds = (z_min - padding * z_range, z_max + padding * z_range)

    return x_bounds, y_bounds, z_bounds


def create_animation(frames, output_path):
    """Create an animation of the transformation pipeline."""
    # Calculate global bounds for fixed axis
    x_bounds, y_bounds, z_bounds = calculate_global_bounds(frames)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        plot_transformation_frame(frames[frame_idx], ax, frame_idx, x_bounds, y_bounds, z_bounds)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=True)

    # Save as GIF
    gif_path = str(output_path).replace(".mp4", ".gif")
    anim.save(gif_path, writer="pillow", fps=10)
    print(f"Animation saved to {gif_path}")

    plt.close()


def plot_transformation_metrics(frames, output_path):
    """Plot transformation metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract metrics for all frames
    timestamps = []
    hand_distances = []
    robot_distances = []
    rotation_diffs = []
    resolution_scales = []

    for frame_idx, frame in enumerate(frames):
        input_data = frame.get("input", {})
        output = frame.get("output", {})
        intermediate = frame.get("intermediate", {})
        metadata = frame.get("metadata", {})

        hand_init = np.array(input_data.get("hand_init_h", [[np.nan] * 3] * 4))
        hand_moving = np.array(input_data.get("hand_moving_h", [[np.nan] * 3] * 4))
        robot_init = np.array(input_data.get("robot_init_h", [[np.nan] * 3] * 4))
        robot_moving = np.array(output.get("robot_moving_h", [[np.nan] * 3] * 4))

        # Skip frames with NaN values
        if (
            has_nan(hand_init[:3, 3])
            or has_nan(hand_moving[:3, 3])
            or has_nan(robot_init[:3, 3])
            or has_nan(robot_moving[:3, 3])
        ):
            continue

        # Calculate distances from initial position
        hand_dist = np.linalg.norm(hand_moving[:3, 3] - hand_init[:3, 3])
        robot_dist = np.linalg.norm(robot_moving[:3, 3] - robot_init[:3, 3])

        # Calculate rotation difference (using relative transformation)
        h_ht_hi = np.array(intermediate.get("h_ht_hi", [[np.nan] * 3] * 4))
        if not has_nan(h_ht_hi[:3, :3]):
            rot_diff = np.arccos(np.clip((np.trace(h_ht_hi[:3, :3]) - 1) / 2, -1, 1))
        else:
            continue

        timestamps.append(metadata.get("timestamp", 0) - frames[0].get("metadata", {}).get("timestamp", 0))
        hand_distances.append(hand_dist)
        robot_distances.append(robot_dist)
        rotation_diffs.append(np.degrees(rot_diff))
        resolution_scales.append(metadata.get("resolution_scale", 1.0))

    timestamps = np.array(timestamps)

    # Position distances
    axes[0, 0].plot(timestamps, hand_distances, "r-", label="Hand", linewidth=2)
    axes[0, 0].plot(timestamps, robot_distances, "b-", label="Robot", linewidth=2)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Distance from Init (m)")
    axes[0, 0].set_title("Position Displacement")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Rotation angles
    axes[0, 1].plot(timestamps, rotation_diffs, "g-", linewidth=2)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Rotation Angle (deg)")
    axes[0, 1].set_title("Hand Rotation Relative to Init")
    axes[0, 1].grid(True, alpha=0.3)

    # Scaling factor
    axes[1, 0].plot(timestamps, resolution_scales, "purple", linewidth=2, marker="o", markersize=4)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Resolution Scale")
    axes[1, 0].set_title("Resolution Scale Over Time")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])

    # Velocity (derived from position)
    if len(timestamps) > 1:
        hand_vel = np.gradient(hand_distances, timestamps)
        robot_vel = np.gradient(robot_distances, timestamps)
        axes[1, 1].plot(timestamps, hand_vel, "r--", label="Hand", linewidth=2)
        axes[1, 1].plot(timestamps, robot_vel, "b-", label="Robot", linewidth=2)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Velocity (m/s)")
        axes[1, 1].set_title("Velocity")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics plot saved to {output_path}")


def plot_frame_rate(frames, output_path, skip_first_n=10):
    """Plot frame rate and processing time over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Extract frame processing times
    frame_indices = []
    timestamps = []
    processing_times = []
    frame_intervals = []
    frame_rates = []

    for frame_idx, frame in enumerate(frames):
        metadata = frame.get("metadata", {})
        timestamp = metadata.get("timestamp", 0)
        proc_time = metadata.get("frame_processing_time", None)

        if timestamp > 0:
            frame_indices.append(frame_idx)
            timestamps.append(timestamp)
            if proc_time is not None:
                processing_times.append(proc_time)

    # Skip first n frames for more stable metrics
    if len(processing_times) > skip_first_n:
        processing_times = processing_times[skip_first_n:]

    timestamps = np.array(timestamps)

    # Skip first n frames for more stable metrics
    if len(timestamps) > skip_first_n:
        timestamps = timestamps[skip_first_n:]

    # Calculate frame intervals (time between consecutive frames)
    if len(timestamps) > 1:
        frame_intervals = np.diff(timestamps)
        # Convert to frame rate (Hz)
        with np.errstate(divide='ignore', invalid='ignore'):
            frame_rates = 1.0 / frame_intervals

    # Plot frame processing time
    ax1 = axes[0]
    if processing_times:
        # Convert to milliseconds for better readability
        processing_times_ms = np.array(processing_times) * 1000
        ax1.plot(range(len(processing_times_ms)), processing_times_ms, 'b-', linewidth=2, label='Processing Time')
        ax1.set_ylabel('Processing Time (ms)')
        ax1.set_title('Frame Processing Time Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_time = np.mean(processing_times_ms)
        median_time = np.median(processing_times_ms)
        max_time = np.max(processing_times_ms)
        min_time = np.min(processing_times_ms)

        stats_text = f'Mean: {mean_time:.2f} ms\nMedian: {median_time:.2f} ms\nMin: {min_time:.2f} ms\nMax: {max_time:.2f} ms'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No processing time data available', ha='center', va='center')
        ax1.set_ylabel('Processing Time (ms)')
        ax1.set_title('Frame Processing Time Over Time')

    # Plot frame rate
    ax2 = axes[1]
    if len(frame_rates) > 0:
        time_offset = timestamps[0]
        time_axis = timestamps[:-1] - time_offset
        valid_indices = ~np.isinf(frame_rates) & ~np.isnan(frame_rates)

        if np.any(valid_indices):
            ax2.plot(time_axis[valid_indices], frame_rates[valid_indices], 'r-', linewidth=2, label='Frame Rate')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Frame Rate (Hz)')
            ax2.set_title('Actual Frame Rate Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add statistics
            valid_rates = frame_rates[valid_indices]
            mean_rate = np.mean(valid_rates)
            median_rate = np.median(valid_rates)
            min_rate = np.min(valid_rates)
            max_rate = np.max(valid_rates)

            stats_text = f'Mean: {mean_rate:.2f} Hz\nMedian: {median_rate:.2f} Hz\nMin: {min_rate:.2f} Hz\nMax: {max_rate:.2f} Hz'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'No valid frame rate data', ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for frame rate calculation', ha='center', va='center')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frame Rate (Hz)')
        ax2.set_title('Actual Frame Rate Over Time')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Frame rate plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize operator transformation pipeline")
    parser.add_argument("log_file", type=str, help="Path to transformation pipeline log file")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--static", action="store_true", help="Generate static comparison plot")
    parser.add_argument("--metrics", action="store_true", help="Generate metrics plot")
    parser.add_argument("--animate", action="store_true", help="Generate animation")
    parser.add_argument("--framerate", action="store_true", help="Generate frame rate plot")

    args = parser.parse_args()

    # Load data
    print(f"Loading log data from {args.log_file}")
    data = load_log_data(args.log_file)

    # Extract frames
    frames = extract_frames(data)
    print(f"Loaded {len(frames)} frames")

    # Filter out frames with too many NaN values
    frames = filter_nan_frames(frames)
    print(f"After filtering NaN values: {len(frames)} valid frames")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base output path
    log_name = Path(args.log_file).stem
    base_output = output_dir / f"{log_name}"
    base_output.mkdir(parents=True, exist_ok=True)

    # Generate requested visualizations
    if len(frames) == 0:
        print("No frames to visualize")
        return

    if args.static:
        print("Generating static comparison plot...")
        plot_transformation_comparison(frames, base_output / "comparison.png")

    if args.metrics:
        print("Generating metrics plot...")
        plot_transformation_metrics(frames, base_output / "metrics.png")

    if args.framerate:
        print("Generating frame rate plot...")
        plot_frame_rate(frames, base_output / "framerate.png")

    if args.animate:
        print("Generating animation...")
        create_animation(frames, base_output / "animation.gif")

    # If no specific visualization requested, show a single frame
    if not (args.static or args.metrics or args.framerate or args.animate):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Show the middle frame
        mid_frame = frames[len(frames) // 2]
        plot_transformation_frame(mid_frame, ax, len(frames) // 2)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
