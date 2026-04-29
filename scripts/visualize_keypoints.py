#!/usr/bin/env python3
"""Visualize logged VR keypoints data."""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def load_json_file(filepath: Path):
    """Load JSON data from file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        return None


def load_multiple_files(pattern: str, file_type: str = "keypoints"):
    """Load and combine multiple JSON files matching a pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching: {pattern}")
        return None

    print(f"Loading {len(files)} files...")
    combined_data = None

    for filepath in files:
        data = load_json_file(Path(filepath))
        if data is None:
            continue

        if combined_data is None:
            combined_data = data
        else:
            if file_type == "keypoints":
                combined_data["frames"].extend(data.get("frames", []))
            elif file_type == "raw":
                combined_data["records"].extend(data.get("records", []))
            combined_data["total_frames"] = len(combined_data.get("frames", combined_data.get("records", [])))

    if combined_data is None:
        print("No valid data found in any file")
        return None

    total = len(combined_data.get("frames", combined_data.get("records", [])))
    print(f"Combined {total} total frames")
    return combined_data


def load_raw_vr_data(filepath: Path):
    """Load raw VR data from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def load_keypoint_data(filepath: Path):
    """Load transformed keypoint data from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def visualize_raw_vr_data(data: dict, frame_idx: int = 0, animate: bool = False):
    """Visualize raw VR keypoints in 3D."""
    records = data.get("records", [])
    if not records:
        print("No records found")
        return

    all_keypoints = []
    all_rotated = []
    for record in records:
        kp = record.get("processed_keypoints", [])
        if kp:
            all_keypoints.append(np.array(kp).reshape(-1, 3))
        rotated = record.get("rotated_keypoints", [])
        if rotated:
            all_rotated.append(np.array(rotated).reshape(-1, 3))

    all_keypoints = np.vstack(all_keypoints) if all_keypoints else np.zeros((1, 3))
    all_rotated = np.vstack(all_rotated) if all_rotated else np.zeros((1, 3))

    all_points = np.vstack([all_keypoints, all_rotated])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    margin = 0.1
    x_range = x_max - x_min + margin
    y_range = y_max - y_min + margin
    z_range = z_max - z_min + margin
    max_range = max(x_range, y_range, z_range)

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title(f"Original Keypoints - {data.get('hand_side', 'unknown')}")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_title("Rotated 90° around X")

    ax3 = fig.add_subplot(133)
    ax3.axis("off")

    def set_axis_limits(ax):
        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    def plot_frame(idx):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        record = records[idx]
        keypoints = record.get("processed_keypoints", [])
        rotated_keypoints = record.get("rotated_keypoints", [])

        if len(keypoints) == 0:
            print(f"Frame {idx}: No keypoints")
            return

        keypoints = np.array(keypoints).reshape(-1, 3)

        ax1.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c="blue", s=50)
        ax1.plot(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], "b-", alpha=0.3)

        for i, (x, y, z) in enumerate(keypoints):
            ax1.text(x, y, z, str(i), fontsize=8)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(f"Original - Frame {idx}")
        set_axis_limits(ax1)

        if len(rotated_keypoints) > 0:
            rotated = np.array(rotated_keypoints).reshape(-1, 3)

            ax2.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c="red", s=50)
            ax2.plot(rotated[:, 0], rotated[:, 1], rotated[:, 2], "r-", alpha=0.3)

            for i, (x, y, z) in enumerate(rotated):
                ax2.text(x, y, z, str(i), fontsize=8)

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y (was -Z)")
        ax2.set_zlabel("Z (was Y)")
        ax2.set_title(f"Rotated - Frame {idx}")
        set_axis_limits(ax2)

        info_text = (
            f"Frame: {idx}/{len(records) - 1}\n"
            f"Timestamp: {record.get('timestamp', 'N/A')}\n"
            f"Shape: {record.get('keypoints_shape', 'N/A')}\n\n"
            f"Rotation: 90° around X\n"
            f"x' = x\n"
            f"y' = -z\n"
            f"z' = y"
        )
        ax3.text(0.1, 0.5, info_text, fontsize=10, family="monospace", verticalalignment="center")
        ax3.axis("off")

    if animate:
        from matplotlib.animation import FuncAnimation

        def update(frame):
            plot_frame(frame)
            return []

        ani = FuncAnimation(fig, update, frames=len(records), interval=50, blit=True)
        plt.show()
    else:
        plot_frame(frame_idx)
        plt.tight_layout()
        plt.show()


def visualize_transformed_data(data: dict, frame_idx: int = 0, animate: bool = False):
    """Visualize transformed keypoints with coordinate frame."""
    frames = data.get("frames", [])
    if not frames:
        print("No frames found")
        return

    all_keypoints = []
    all_origins = []
    for frame in frames:
        kp = frame.get("keypoints", [])
        if kp:
            kp_array = np.array(kp)
            if not np.any(np.isnan(kp_array)):
                all_keypoints.append(kp_array)
        coord_frame = frame.get("coordinate_frame", {})
        origin = coord_frame.get("origin", [0, 0, 0])
        origin_array = np.array(origin)
        if not np.any(np.isnan(origin_array)):
            all_origins.append(origin_array)

    all_keypoints = np.vstack(all_keypoints) if all_keypoints else np.zeros((1, 3))
    all_origins = np.array(all_origins) if all_origins else np.zeros((1, 3))

    all_points = np.vstack([all_keypoints, all_origins])

    # Filter out NaN and Inf values
    valid_mask = np.all(np.isfinite(all_points), axis=1)
    all_points = all_points[valid_mask] if np.any(valid_mask) else np.zeros((1, 3))

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    margin = 0.15
    x_range = x_max - x_min + margin
    y_range = y_max - y_min + margin
    z_range = z_max - z_min + margin
    max_range = max(x_range, y_range, z_range)

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title("Transformed Keypoints")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_title("Coordinate Frame")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_title("Combined View")

    def set_axis_limits(ax):
        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    def plot_frame(idx):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        frame = frames[idx]
        keypoints = np.array(frame.get("keypoints", []))
        coord_frame = frame.get("coordinate_frame", {})

        if len(keypoints) > 0:
            ax1.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c="blue", s=30)
            ax1.plot(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], "b-", alpha=0.3)

            for i, (x, y, z) in enumerate(keypoints):
                ax1.text(x, y, z, str(i), fontsize=6)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(f"Keypoints - Frame {idx}")
        set_axis_limits(ax1)

        origin = coord_frame.get("origin", [0, 0, 0])
        x_vec = coord_frame.get("x_vector", [1, 0, 0])
        y_vec = coord_frame.get("y_vector", [0, 1, 0])
        z_vec = coord_frame.get("z_vector", [0, 0, 1])

        origin = np.array(origin)
        x_vec = np.array(x_vec)
        y_vec = np.array(y_vec)
        z_vec = np.array(z_vec)

        scale = 0.1

        ax2.quiver(*origin, *x_vec * scale, color="r", label="X")
        ax2.quiver(*origin, *y_vec * scale, color="g", label="Y")
        ax2.quiver(*origin, *z_vec * scale, color="b", label="Z")
        ax2.scatter(*origin, c="black", s=100, marker="o")

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()
        ax2.set_title("Coordinate Frame")
        set_axis_limits(ax2)

        if len(keypoints) > 0:
            ax3.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c="blue", s=20, alpha=0.5)

        ax3.quiver(*origin, *x_vec * scale, color="r", label="X", linewidth=2)
        ax3.quiver(*origin, *y_vec * scale, color="g", label="Y", linewidth=2)
        ax3.quiver(*origin, *z_vec * scale, color="b", label="Z", linewidth=2)

        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.set_title(f"Combined - Frame {idx}/{len(frames) - 1}")
        set_axis_limits(ax3)

    if animate:
        from matplotlib.animation import FuncAnimation

        def update(frame):
            plot_frame(frame)
            return []

        ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
        plt.show()
    else:
        plot_frame(frame_idx)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize logged VR keypoints")
    parser.add_argument(
        "file",
        type=str,
        help="Path to JSON file or glob pattern (e.g., 'data/keypoint_logs/keypoints_left_*.json')",
    )
    parser.add_argument("--frame", "-f", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--animate", "-a", action="store_true", help="Animate through all frames")
    parser.add_argument(
        "--combine", "-c", action="store_true", help="Combine multiple files matching the pattern"
    )
    args = parser.parse_args()

    if args.combine or "*" in args.file:
        if "keypoints" in args.file:
            data = load_multiple_files(args.file, file_type="keypoints")
        elif "raw_vr_data" in args.file:
            data = load_multiple_files(args.file, file_type="raw")
        else:
            data = load_multiple_files(args.file, file_type="keypoints")

        if data is None:
            return
    else:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        data = load_json_file(filepath)

    # Detect file type based on content
    if "records" in data and len(data.get("records", [])) > 0:
        print(f"Visualizing raw VR data ({len(data['records'])} frames)...")
        visualize_raw_vr_data(data, args.frame, args.animate)
    elif "frames" in data:
        print(f"Visualizing transformed keypoints ({len(data['frames'])} frames)...")
        visualize_transformed_data(data, args.frame, args.animate)
    else:
        print("Unknown file format")


if __name__ == "__main__":
    main()
