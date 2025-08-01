python src/beavr/scripts/control_robot.py \
  --robot.type=multi_robot_adapter \
  --control.type=record \
  --control.fps=30 \
  --control.repo_id=aposadasn/eval_lx7r_pickup_model_test \
  --control.tags='["demo"]' \
  --control.warmup_time_s=10 \
  --control.episode_time_s=45 \
  --control.reset_time_s=45 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=arclabmit/lx7r_act_pickup_model \
  --control.single_task="Move the right xarm7 to the target position" \
  --teleop.robot_name=leap,xarm7 \
  --teleop.operate=false \
  --control.resume=false