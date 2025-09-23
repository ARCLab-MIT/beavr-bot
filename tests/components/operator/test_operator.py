import time

import numpy as np

from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator
from beavr.teleop.configs.constants import robots


def _identity4():
    h = np.eye(4, dtype=np.float64)
    return h


def test_operator_publishes_cartesian_target(bus):
    host = "127.0.0.1"
    transformed_keypoints_port = 5555
    endeff_publish_port = 7777
    endeff_subscribe_port = 6666  # not used by operator in this test

    # Minimal transforms (identity): robot base to VR base and hand tracking to VR base
    h_r_v = _identity4()
    h_t_v = _identity4()

    # Create operator (right hand)
    op = XArmOperator(
        operator_name="xarm7_right_operator",
        host=host,
        transformed_keypoints_port=transformed_keypoints_port,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=endeff_publish_port,
        endeff_subscribe_port=endeff_subscribe_port,
        moving_average_limit=3,
        h_r_v=h_r_v,
        h_t_v=h_t_v,
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=None,
        logging_config={"enabled": False},
        hand_side=robots.RIGHT,
    )

    # Seed initial state by faking a robot pose response during reset
    # The operator publishes 'reset' to endeff_publish_port; the robot would reply on endeff_subscribe_port
    # For this focused test, we directly place a valid 4x4 identity pose on the robot feedback topic.
    from beavr.teleop.components.interface.interface_types import CartesianState

    bus.publish(
        host,
        endeff_subscribe_port,
        "endeff_homo",
        CartesianState(
            timestamp_s=time.time(),
            h_matrix=tuple(map(tuple, np.eye(4).tolist())),
        ),
    )

    # Provide a first hand frame to exit reset state. InputFrame.frame_vectors should be 4 vectors: origin + 3 axes.
    # Here we simulate the right hand at origin with canonical axes; also provide some keypoints (not used for pose when frame_vectors present).
    origin = (0.0, 0.0, 0.0)
    x = (1.0, 0.0, 0.0)
    y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    frame_vectors = (origin, x, y, z)
    keypoints = [(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS

    # Publish hand frame for the right hand topic consumed by operator
    right_frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"
    bus.publish(
        host,
        transformed_keypoints_port,
        right_frame_topic,
        InputFrame(
            timestamp_s=time.time(),
            hand_side=robots.RIGHT,
            keypoints=keypoints,
            is_relative=False,
            frame_vectors=frame_vectors,
        ),
    )

    # First apply should reset and cache frames
    op._apply_retargeted_angles()

    # Provide another frame identical to produce a target command
    bus.publish(
        host,
        transformed_keypoints_port,
        right_frame_topic,
        InputFrame(
            timestamp_s=time.time(),
            hand_side=robots.RIGHT,
            keypoints=keypoints,
            is_relative=False,
            frame_vectors=frame_vectors,
        ),
    )

    # Second apply should compute and publish a CartesianTarget when in CONT state
    op._apply_retargeted_angles()

    # Read what the operator published
    cmd = bus.recv_latest(endeff_publish_port, "endeff_coords")
    assert isinstance(cmd, CartesianTarget)
    # With identity transforms and zero motion,
    # target should match robot init pose => zero position and identity quat
    pos = np.asarray(cmd.position_m, dtype=np.float64)
    quat = np.asarray(cmd.orientation_xyzw, dtype=np.float64)
    np.testing.assert_allclose(pos, np.zeros(3), atol=1e-6)
    # Unit quaternion with positive w hemisphere
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)
    assert quat[3] >= 0.0
