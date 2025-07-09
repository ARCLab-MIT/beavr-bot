# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python beavr/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python beavr/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python beavr/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python beavr/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python beavr/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import time
import multiprocessing  # Needed for process types
from dataclasses import asdict
from pprint import pformat

# from safetensors.torch import load_file, save_file
from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from beavr.lerobot.common.policies.factory import make_policy
from beavr.lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from beavr.lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from beavr.lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from beavr.lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from beavr.lerobot.common.utils.utils import init_logging, log_say
from beavr.lerobot.configs import parser


########################################################################################
# Control modes
########################################################################################

_teleop_processes: list[multiprocessing.Process] | None = None  # Populated at runtime

def start_teleop_process(
    *,
    robot_name: str | None = None,
    operate: bool | None = None,
    wait_for_exit: bool = False,
):
    """Launch the teleoperation helper stack (detector/cameras/robot interface).

    Parameters
    ----------
    robot_name : str | None
        Identifier of the robot combo to load (e.g. ``leap,xarm7`` or ``leap``).
        If *None*, we will attempt to read `--teleop_robot_name` from the CLI
        (via ``beavr.configs.parser``).  If that is also missing the function
        falls back to the default "leap,xarm7".

    operate : bool | None
        Whether to start the *Operator* processes that listen to the VR
        detector and forward the commands to the robots.  Setting this to
        ``False`` is crucial when a *policy* publishes the commands itself
        through `robot.send_action()` – running both simultaneously causes
        PUB/SUB conflicts.  When *None*, the value will be automatically
        derived: it is set to ``False`` whenever a policy is supplied on the
        CLI (detected via the presence of `--control.policy.` arguments);
        otherwise the default is ``True``.
    """

    global _teleop_processes
    if _teleop_processes is not None:
        logging.info("Teleoperation already running – skipping launch.")
        return

    # Lazy imports to avoid the heavy dependency cost when teleop is not needed.
    from beavr.teleop.main import MainConfig  # pylint: disable=import-error
    from beavr.teleop.components import TeleOperator

    # ------------------------------------------------------------------
    # Resolve *robot_name*
    # ------------------------------------------------------------------
    if robot_name is None:
        # Try CLI override --teleop_robot_name=...
        robot_name = parser.parse_arg("teleop_robot_name")

    if robot_name is None:
        # Fallback for legacy flag --teleop.robot_name=...
        robot_name = parser.parse_arg("teleop.robot_name")

    if robot_name is None:
        # Ultimate fallback – use new multi-robot syntax
        robot_name = "leap,xarm7"

    # ------------------------------------------------------------------
    # Resolve *operate*
    # ------------------------------------------------------------------
    if operate is None:
        # CLI override (explicit)
        operate_cli = parser.parse_arg("teleop_operate")
        if operate_cli is not None:
            operate = operate_cli.lower() not in {"0", "false", "no", "off"}

    if operate is None:
        # Implicit heuristic – disable operator when a policy is provided
        has_policy_path = parser.get_path_arg("control.policy") is not None
        has_policy_type = parser.get_type_arg("control.policy") is not None
        operate = not (has_policy_path or has_policy_type)

    logging.info(
        "Starting teleoperation helper with robot_name='%s', operate=%s",
        robot_name,
        operate,
    )

    # Build the configuration for the selected robot combo using the new structured config.
    # Create MainConfig with robot_name and set operate flag in teleop.flags
    main_config = MainConfig(robot_name=robot_name)
    main_config.teleop.flags.operate = operate

    logging.info("Instantiating TeleOperator (Draccus version)…")
    teleop = TeleOperator(main_config)
    _teleop_processes = teleop.get_processes()

    # Start all sub-processes.
    for p in _teleop_processes:
        p.start()

    # Give PUB sockets a moment to come up before the main thread proceeds.
    time.sleep(1)

    if wait_for_exit:
        try:
            for p in _teleop_processes:
                p.join()
        except KeyboardInterrupt:
            logging.info("Teleoperation interrupted by user.")
        finally:
            stop_teleop_process()

def stop_teleop_process():
    """Terminate the TeleOperator's child processes if they are active."""
    global _teleop_processes
    if _teleop_processes:
        logging.info("Stopping teleoperation processes…")
        for p in _teleop_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        _teleop_processes = None

@safe_disconnect
def teleoperate():
    """Use the BeaVR teleoperation system by running teleop.py and waiting for it to complete."""
    start_teleop_process(wait_for_exit=True)


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # When recording, the teleoperation system is assumed to be already running.
    # This function is now only responsible for the recording logic itself.
    try:
        # TODO: Add option to record logs
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.repo_id,
                root=cfg.root,
            )
            if len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.num_image_writer_processes,
                    num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
        else:
            # Create empty dataset or load existing saved episodes
            sanity_check_dataset_name(cfg.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.repo_id,
                cfg.fps,
                root=cfg.root,
                robot=robot,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        # Load pretrained policy
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

        listener, events = init_keyboard_listener()

        # Execute a few seconds without recording to:
        # 1. teleoperate the robot to move it in starting position if no policy provided,
        # 2. give times to the robot devices to connect and start synchronizing,
        # 3. place the cameras windows on screen
        enable_teleoperation = policy is None
        log_say("Warmup record", cfg.play_sounds)
        warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_data, cfg.fps)

        recorded_episodes = 0
        while True:
            if recorded_episodes >= cfg.num_episodes:
                break

            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_episode(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_data=cfg.display_data,
                policy=policy,
                fps=cfg.fps,
                single_task=cfg.single_task,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # TODO: add an option to enable teleoperation during reset
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

            if events["stop_recording"]:
                break

        log_say("Stop recording", cfg.play_sounds, blocking=True)
        stop_recording(robot, listener, cfg.display_data)

        if cfg.push_to_hub:
            dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

        log_say("Exiting", cfg.play_sounds)
        return dataset
    finally:
        # Cleanup is now handled by the main control_robot function
        pass


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO: refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO: Add option to record logs

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Determine whether the teleoperation helper needs to run.
    start_teleop = isinstance(cfg.control, (TeleoperateControlConfig, RecordControlConfig))

    if start_teleop:
        start_teleop_process(
            robot_name=getattr(cfg.teleop, "robot_name", None),
            operate=getattr(cfg.teleop, "operate", None),
        )
        logging.info(
            "Waiting 5s for teleoperation system to initialize before robot connection..."
        )
        # TODO: Improve this logic shouldn't need sleep
        time.sleep(5)

    # Pass the controller to the robot adapter
    robot = make_robot_from_config(cfg.robot)

    try:
        if not robot.is_connected:
            robot.connect()

        if isinstance(cfg.control, TeleoperateControlConfig):
            # The teleop process was already started, just wait for it to complete.
            global _teleop_processes
            if _teleop_processes:
                logging.info("Waiting for teleoperation process to complete...")
                for p in _teleop_processes:
                    p.join()

        elif isinstance(cfg.control, RecordControlConfig):
            record(robot, cfg.control)
        elif isinstance(cfg.control, ReplayControlConfig):
            replay(robot, cfg.control)
    except KeyboardInterrupt:
        logging.info("Main control loop interrupted by user.")
    finally:
        # This block ensures cleanup happens on normal exit or interrupt.
        logging.info("Starting cleanup...")
        if robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()
        # Always try to stop the teleop process on exit
        stop_teleop_process()
        logging.info("Cleanup complete. Exiting.")


if __name__ == "__main__":
    control_robot()