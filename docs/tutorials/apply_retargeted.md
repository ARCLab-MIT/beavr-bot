    def _apply_retargeted_angles(self):
        """
        Calculates and applies the retargeted end-effector pose based on hand motion.
        Handles state changes (reset, pause/resume), applies transformations,
        filters the result, and publishes the command.
        """
        # 1. Check for state changes (Pause/Resume, Resolution)
        new_arm_teleop_state = self._get_arm_teleop_state()
        self.resolution_scale = self._get_resolution_scale_mode() # Update resolution scale

        # Determine if a reset is needed
        needs_reset = self.is_first_frame or \
                      (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT)

        # Update state *after* checking for transition
        self.arm_teleop_state = new_arm_teleop_state

        # If paused, do nothing further
        if self.arm_teleop_state == ARM_TELEOP_STOP and False: # TODO: This does not work with current app
            # Optionally send a 'hold position' command or simply return
            # print(f"{self.operator_name}: Teleoperation paused.")
            # We might want to clear the filter state or cache when paused
            self.comp_filter = None
            self.last_valid_hand_frame = None # Clear cached frame on pause
            print("early return")
            return

        # 2. Handle Reset Condition
        if needs_reset:
            moving_hand_frame = self._reset_teleop()
            if moving_hand_frame is None:
                print(f"ERROR ({self.operator_name}): Reset failed, cannot proceed.")
                return # Exit if reset failed
            # Reset is done, is_first_frame is now False
        else:
            # 3. Get Current Hand Frame (if not resetting)
            moving_hand_frame = self._get_hand_frame()

        # If no valid hand frame is available (after reset or during normal operation), exit
        if moving_hand_frame is None:
            # print(f"Warning ({self.operator_name}): No valid hand frame received, skipping cycle.")
            return

        # Ensure initial robot/hand poses are set (should be handled by reset)
        if self.robot_init_h is None or self.hand_init_h is None:
             print(f"ERROR ({self.operator_name}): Initial robot or hand poses not set. Triggering reset.")
             self.is_first_frame = True # Force reset on next cycle
             return

        # 4. Convert current hand frame to Homogeneous Matrix
        try:
            self.hand_moving_h = self._turn_frame_to_homo_mat(moving_hand_frame)
        except ValueError as e:
            print(f"Error ({self.operator_name}): Could not convert moving hand frame: {e}")
            return # Skip cycle if conversion fails

        # 5. Calculate Relative Transformation
        # H_HT_HI = H_HI_HH^-1 * H_HT_HH
        # Use solve for potentially better numerical stability than inv
        try:
            h_hi_hh_inv = np.linalg.inv(self.hand_init_h) # Inverse of initial hand pose
            h_ht_hi = h_hi_hh_inv @ self.hand_moving_h # Relative motion of hand w.r.t its start pose
            # Alternative using solve: H_HT_HI = np.linalg.solve(self.hand_init_H, self.hand_moving_H)
        except np.linalg.LinAlgError:
            print(f"Error ({self.operator_name}): Could not invert initial hand matrix. Resetting.")
            self.is_first_frame = True
            return

        # 6. Apply Coordinate Transformations (using provided H_R_V and H_T_V)
        # Transform relative hand motion from Hand Tracking frame (T) to Robot base frame (R)
        # Formula: H_RT_RI = H_R_V * H_V_T * H_HT_HI * H_T_V * H_V_R
        # Where H_V_T = inv(H_T_V), H_V_R = inv(H_R_V)
        # Simplified: Relative motion in Robot frame = inv(H_R_V) * H_T_V * H_HT_HI * inv(H_T_V) * H_R_V
        # Let's verify the original logic's intent. It seems to separate rotation and translation transforms.
        # H_HT_HI_r = inv(H_R_V)[:3,:3] @ H_HT_HI[:3,:3] @ H_R_V[:3,:3] # Rotation part transformed
        # H_HT_HI_t = inv(H_T_V)[:3,:3] @ H_HT_HI[:3,3] # Translation part transformed (assuming H_T_V only affects translation origin/scaling?)

        # Let's stick to the original separate transformation logic for now, using self.h_r_v and self.h_t_v
        try:
            h_r_v_inv = np.linalg.inv(self.h_r_v)
            h_t_v_inv = np.linalg.inv(self.h_t_v)

            # Transform rotation part: Apply rotation from H_R_V inverse, then relative hand rotation, then H_R_V
            h_ht_hi_r = h_r_v_inv[:3,:3] @ h_ht_hi[:3,:3] @ self.h_r_v[:3,:3]
            # Transform translation part: Apply H_T_V inverse to relative hand translation
            # Scale translation by resolution_scale
            h_ht_hi_t = h_t_v_inv[:3,:3] @ h_ht_hi[:3, 3] * self.resolution_scale

        except np.linalg.LinAlgError:
            print(f"Error ({self.operator_name}): Could not invert H_R_V or H_T_V matrix.")
            # Handle error appropriately, maybe reset or use identity
            return

        # Ensure rotation part is a valid rotation matrix
        h_ht_hi_r = self.project_to_rotation_matrix(h_ht_hi_r)

        # Combine into a relative affine transformation in the robot's base frame
        relative_affine_in_robot_frame = np.eye(4)
        relative_affine_in_robot_frame[:3, :3] = h_ht_hi_r
        relative_affine_in_robot_frame[:3, 3] = h_ht_hi_t

        # 7. Calculate Target Robot Pose
        # H_RT_RH = H_RI_RH * relative_affine_in_robot_frame
        h_rt_rh = self.robot_init_h @ relative_affine_in_robot_frame

        # Ensure the final target pose has a valid rotation matrix
        h_rt_rh[:3, :3] = self.project_to_rotation_matrix(h_rt_rh[:3, :3])
        self.robot_moving_h = copy(h_rt_rh) # Store the calculated target pose

        # 8. Convert Target Pose to Cartesian [pos, quat]
        cart_target_raw = self._homo2cart(self.robot_moving_h)

        # 9. Apply Filtering
        if self.use_filter:
            # Initialize filter on the first valid frame after reset/start
            if self.comp_filter is None:
                 # Use the *raw* target pose from the first frame as the initial filter state
                 self.comp_filter = CompStateFilter(
                     init_state=cart_target_raw,
                     pos_ratio=0.7,  # Default values, consider making configurable
                     ori_ratio=0.85,
                     adaptive=True
                 )
                 cart_target_filtered = cart_target_raw # Use raw value for the very first frame
            else:
                 cart_target_filtered = self.comp_filter(cart_target_raw)
        else:
            cart_target_filtered = cart_target_raw # No filtering

        # 10. Convert Filtered Pose to desired output format (axis-angle)
        position = cart_target_filtered[0:3]
        orientation_quat = cart_target_filtered[3:7]
        # Ensure quaternion is normalized before converting to axis-angle
        norm = np.linalg.norm(orientation_quat)
        if norm > 1e-6:
             orientation_quat /= norm
        else:
             orientation_quat = np.array([0.,0.,0.,1.]) # Default to identity if norm is zero

        x, y, z = self.quat_to_axis_angle(orientation_quat[0], orientation_quat[1], orientation_quat[2], orientation_quat[3])
        axis_angle_orientation = [x, -z, y]

        # 11. Publish Command
        command_data = {
            "position": position.tolist(), # Convert numpy arrays to lists for JSON compatibility
            "orientation": axis_angle_orientation,
            "timestamp": time.time()
        }
        self.unified_publisher.pub_keypoints(command_data, "endeff_coords")

        # 12. Logging (Optional)
        if self.logging_enabled and self.pose_logger:
            try:
                # Ensure all matrices are valid before logging
                if self.hand_init_h is not None and \
                   self.robot_init_h is not None and \
                   self.hand_moving_h is not None and \
                   self.robot_moving_h is not None:
                    self.pose_logger.log_frame(
                        self.hand_init_h,
                        self.robot_init_h,
                        self.hand_moving_h,
                        self.robot_moving_h # Log the target pose *before* filtering
                    )
            except Exception as e:
                print(f"Error logging frame ({self.operator_name}): {e}")

        # Optional diagnostics print
        # if time.time() % 5 < 0.02: # Print roughly every 5 seconds based on VR_FREQ
        #     print(f"\nDiagnostics ({self.operator_name}):")
        #     print(f"  Teleop State: {'CONT' if self.arm_teleop_state == ARM_TELEOP_CONT else 'STOP'}")
        #     print(f"  Resolution Scale: {self.resolution_scale:.2f}")
        #     if self.hand_moving_h is not None:
        #          print(f"  Hand Moving Det: {np.linalg.det(self.hand_moving_h[:3,:3]):.4f}")
        #     if self.robot_moving_h is not None:
        #          print(f"  Robot Target Det: {np.linalg.det(self.robot_moving_h[:3,:3]):.4f}")
        #     print(f"  Target Pos: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        #     print(f"  Target Euler (deg): R={np.degrees(euler_orientation[0]):.1f}, P={np.degrees(euler_orientation[1]):.1f}, Y={np.degrees(euler_orientation[2]):.1f}")
        #     if self.comp_filter and hasattr(self.comp_filter, 'velocity'):
        #          print(f"  Filter Vel Norm: {np.linalg.norm(self.comp_filter.velocity):.4f}")
