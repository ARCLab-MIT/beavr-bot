#!/usr/bin/env python3
"""
Test script: command the XArm7 robot to move to a single Cartesian pose.

The pose is given in axis-angle representation `[x, y, z, rx, ry, rz]` where the
translation is in metres and the rotation components are in radians.

Example usage (move 30 cm forward, keep orientation pointing down):
    python tests/scripts/test_robot_motion.py \
        --ip 192.168.86.230 \
        --x 0.30 --y 0.00 --z 0.45 \
        --rx 3.1416 --ry 0 --rz 0

Notes:
* The underlying `Robot.move_arm_cartesian` call multiplies the position by
  `1000` (see `XARM_SCALE_FACTOR`). Therefore **pass positions in metres**
  when calling this script.
* Orientation is expected as axis-angle, not Euler angles or quaternions.
* If the robot is already at the requested pose the command is ignored.
"""

import argparse
import time
import numpy as np

from beavr.controllers.xarm7_control import DexArmControl


import json


def parse_args() -> argparse.Namespace:
    """Return CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Send a single Cartesian command to an XArm7 robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", type=str, default="192.168.86.230",
                        help="IP address of the XArm robot")
    # Manual-pose arguments (ignored if --num-actions > 0)
    parser.add_argument("--x", type=float,
                        help="Target X position in metres")
    parser.add_argument("--y", type=float,
                        help="Target Y position in metres")
    parser.add_argument("--z", type=float,
                        help="Target Z position in metres")
    parser.add_argument("--rx", type=float,
                        help="Axis-angle Rx component in radians")
    parser.add_argument("--ry", type=float,
                        help="Axis-angle Ry component in radians")
    parser.add_argument("--rz", type=float,
                        help="Axis-angle Rz component in radians")

    # Action-file playback
    parser.add_argument("--actions-file", type=str,
                        default="tests/scripts/actions.json",
                        help="JSON file containing an array of action arrays")
    parser.add_argument("--num-actions", type=int, default=0,
                        help="If >0, play back the first N actions from --actions-file")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration (seconds) passed to the controller")
    parser.add_argument("--control-mode", choices=["servo", "position"], default="servo",
                        help="Use fast servo streaming (mode 1) or planned position control (mode 0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Connecting to XArm controller …")
    ctrl = DexArmControl(ip=args.ip)

    print("Homing robot …")
    ctrl.home_arm()
    time.sleep(1.0)

    # ------------------------------------------------------------------
    # Action-file mode
    # ------------------------------------------------------------------
    if args.num_actions > 0:
        # Load action list
        with open(args.actions_file, "r") as f:
            actions = json.load(f)

        total = min(args.num_actions, len(actions))
        print(f"Executing {total} actions from '{args.actions_file}' …")

        for idx in range(total):
            action = actions[idx]
            if len(action) < 6:
                print(f"Action {idx} is too short, skipping")
                continue

            raw_pose = np.array(action[:6], dtype=np.float32)
            # Detect badly scaled translation: if all |x,y,z| < 0.01 m, assume values were in metres×0.001
            if np.max(np.abs(raw_pose[:3])) < 0.01:
                corrected = raw_pose.copy()
                corrected[:3] *= 1000.0
                print(f"[{idx+1}/{total}] Detected small translation {raw_pose[:3]}, scaled to {corrected[:3]}")
                pose = corrected
            else:
                pose = raw_pose

            print(f"[{idx+1}/{total}] Pose (m & rad): {pose}")

            if args.control_mode == "servo":
                status = ctrl.move_arm_cartesian(pose.tolist(), duration=args.duration)
            else:
                # Position control: ensure mode 0 (position) & send as mm
                ctrl.robot.set_mode_and_state(0, 0)
                pose_mm = pose.copy()
                pose_mm[:3] *= 1000.0  # metres -> mm
                status = ctrl.robot.set_position_aa(pose_mm.tolist(), is_radian=True, wait=False)
            if status != 0:
                print(f"  Warning: controller returned {status}")

            # Simple pause between actions
            time.sleep(1/50)

        print("All actions executed.")
    else:
        # ------------------------------------------------------------------
        # Single-pose mode
        # ------------------------------------------------------------------
        required = [args.x, args.y, args.z, args.rx, args.ry, args.rz]
        if any(v is None for v in required):
            raise ValueError("For single-pose mode you must supply --x --y --z --rx --ry --rz")

        target_pose = np.array(required, dtype=np.float32)
        print(f"Moving to pose (metres & radians): {target_pose}")

        if args.control_mode == "servo":
            status = ctrl.move_arm_cartesian(target_pose.tolist(), duration=args.duration)
        else:
            ctrl.robot.set_mode_and_state(0, 0)
            pose_mm = target_pose.copy()
            pose_mm[:3] *= 1000.0
            status = ctrl.robot.set_position_aa(pose_mm.tolist(), is_radian=True, wait=False)
        print(f"Controller returned status: {status}")

        # Wait until motion finishes
        while ctrl.robot.get_is_moving():
            time.sleep(0.05)

        print("Motion completed.")


if __name__ == "__main__":
    main()
