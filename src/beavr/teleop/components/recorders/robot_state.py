import os
import time
import h5py
import zmq
import numpy as np
from .recorder import Recorder
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.configs.constants import robots
import logging

from beavr.teleop.utils.network import ZMQKeypointSubscriber

logger = logging.getLogger(__name__)


# To record robot information by subscribing to ZMQ state dictionaries
class RobotInformationRecord(Recorder):
    def __init__(self, host: str, port: int, topic: str, record_key: str, storage_path: str, robot_name: str):
        """
        Initializes the recorder to subscribe to robot state dictionaries via ZMQ
        and extract a specific key.

        Args:
            host: The hostname or IP address of the ZMQ publisher.
            port: The port number of the ZMQ publisher.
            topic: The ZMQ topic to subscribe to (should match the robot's publishing topic, e.g., 'right_xarm7').
            record_key: The key to extract from the received state dictionary
                        (e.g., 'joint_states', 'xarm_cartesian_states').
            storage_path: The directory path to save the recorded data.
            robot_name: The name of the robot being recorded (used for filename).
        """
        self.record_key = record_key # Key to extract from dict
        self.storage_path = storage_path
        self.robot_name = robot_name # Used for filename and logging
        self._host = host
        self._port = port
        self._zmq_topic = topic # ZMQ topic to subscribe to

        logger.info(f"Initializing RobotInformationRecord for: {robot_name} - Recording Key: '{record_key}'")
        logger.info(f"  Connecting to ZMQ Publisher at: tcp://{host}:{port}, Subscribing to Topic: '{self._zmq_topic}'")

        # Instantiate the ZMQ subscriber
        self.subscriber = ZMQKeypointSubscriber(
            host=self._host,
            port=self._port,
            topic=self._zmq_topic # Subscribe to the specific robot's topic
        )

        # Initialize storage and state variables
        self._init_storage() # Uses robot_name and record_key for filename
        self.num_datapoints = 0
        self.buffer = [] # Use a simple list buffer for received data points
        self.buffer_limit = 1000 # Save every N data points (adjust as needed)

        # Timer for frequency estimation (optional)
        self.timer = FrequencyTimer(robots.RECORDER_FREQ) # Or a different default/config value

    def stream(self):
        """
        Continuously receives state dictionaries via ZMQ, extracts the relevant data,
        and buffers it for saving.
        """
        logger.info(f"[{self.robot_name} - {self.record_key}] Starting ZMQ subscription loop...")
        self.record_start_time = time.time()
        received_count = 0
        last_print_time = time.time()
        log_interval = 100 # Log type/shape every 100 messages per key
        log_counter = 0

        try:
            while True:
                # MODIFIED: Call recv_keypoints without timeout_ms
                # Use NOBLOCK flag to prevent waiting indefinitely
                try:
                    state_dict = self.subscriber.recv_keypoints(flags=zmq.NOBLOCK)
                except zmq.Again:
                    # No message received, sleep briefly and continue
                    time.sleep(0.001) # Sleep 1ms to yield CPU
                    state_dict = None
                    # Continue to the next iteration of the while loop
                    # to check for KeyboardInterrupt or print stats
                    # continue # Optional: uncomment if you only want to process when a message arrives

                if state_dict is not None:
                    received_count += 1
                    # Extract the specific data point using the record_key
                    if self.record_key in state_dict:
                        data_to_store = state_dict[self.record_key]
                        timestamp = state_dict.get('timestamp', time.time()) # Fallback timestamp

                        # --- START: Added Logging ---
                        log_counter += 1
                        if log_counter % log_interval == 1:
                            data_type = type(data_to_store)
                            shape_info = "N/A"
                            dtype_info = "N/A"
                            if isinstance(data_to_store, np.ndarray):
                                shape_info = data_to_store.shape
                                dtype_info = data_to_store.dtype
                            logger.info(f"[{self.robot_name} - {self.record_key}] Data sample type: {data_type}, Shape: {shape_info}, Dtype: {dtype_info}")
                        # --- END: Added Logging ---

                        # Add the extracted data and timestamp to the buffer
                        self.buffer.append({'data': data_to_store, 'timestamp': timestamp})

                        # Check if buffer limit is reached
                        if len(self.buffer) >= self.buffer_limit:
                            self._save_buffer() # Save the current buffer
                            self.buffer = [] # Clear the buffer

                    else:
                        # Only print this warning occasionally to avoid spam
                        if received_count % 100 == 1:
                             logger.warning(f"Warning: Key '{self.record_key}' not found in received state dict for {self.robot_name}. Keys: {list(state_dict.keys())}")

                # Optional: Print stats periodically
                current_time = time.time()
                if current_time - last_print_time > 5.0: # Print every 5 seconds
                    elapsed = current_time - self.record_start_time
                    rate = received_count / elapsed if elapsed > 0 else 0
                    logger.info(f"[{self.robot_name} - {self.record_key}] Received {received_count} messages in {elapsed:.2f}s ({rate:.1f}/s). Buffer size: {len(self.buffer)}")
                    last_print_time = current_time

                # Removed the separate sleep here as it's handled in the zmq.Again block

        except KeyboardInterrupt:
            logger.info(f"[{self.robot_name} - {self.record_key}] KeyboardInterrupt received. Saving remaining buffer...")
            self.record_end_time = time.time()
            self._save_buffer() # Save any remaining data
            logger.info(f"[{self.robot_name} - {self.record_key}] Exiting.")
        except Exception as e:
            logger.error(f"[{self.robot_name} - {self.record_key}] Error in stream loop: {e}")
            # Optionally save buffer on error too
            self._save_buffer()
        finally:
            # Ensure subscriber is stopped
            if hasattr(self, 'subscriber'):
                self.subscriber.stop()

    def _init_storage(self):
        """Initializes storage file path and name."""
        # Use robot_name and record_key for a unique filename
        self._filename = '{}_{}'.format(self.robot_name, self.record_key)
        self.notify_component_start('{}'.format(self._filename)) # Assuming this is for logging/status
        self._recorder_file_name = os.path.join(self.storage_path, self._filename + '.h5')
        logger.info(f"[{self.robot_name} - {self.record_key}] Recording data to: {self._recorder_file_name}")
        # Data containers are handled by the buffer list now

    def _save_buffer(self):
        """Saves the current contents of the buffer to the HDF5 file,
        handling dictionary payloads by saving them into groups."""
        if not self.buffer:
            # print(f"[{self.robot_name} - {self.record_key}] Buffer is empty. Nothing to save.") # Less verbose
            return

        num_to_save_initially = len(self.buffer)
        logger.info(f"[{self.robot_name} - {self.record_key}] Processing {num_to_save_initially} data points from buffer for saving...")

        # --- Data Aggregation Logic ---
        aggregated_data = {} # Dict to store lists for each sub-key (e.g., 'joint_position', 'timestamp')
        valid_items_processed = 0

        for item in self.buffer:
            data_payload = item['data']
            # Use the timestamp stored alongside the data in the buffer item
            timestamp = item['timestamp']

            if isinstance(data_payload, dict):
                # Record the main timestamp for this dictionary payload
                if 'timestamps' not in aggregated_data:
                    aggregated_data['timestamps'] = []
                aggregated_data['timestamps'].append(timestamp)

                # Iterate through keys within the data payload dictionary
                for sub_key, sub_data in data_payload.items():
                    if sub_key == 'timestamp': # Avoid duplicating timestamp if present in payload
                        continue
                    # Ensure sub_data is numpy array for consistency if possible
                    if isinstance(sub_data, (list, tuple)):
                         try:
                              sub_data = np.array(sub_data, dtype=np.float32)
                         except ValueError:
                              logger.warning(f"  Warning: Could not convert list/tuple to np.float32 for sub_key '{sub_key}'. Skipping this sub_key for this item.")
                              continue # Skip this sub_key for this item
                    elif not isinstance(sub_data, np.ndarray) and sub_data is not None:
                         # Try converting other non-None types
                         try:
                              sub_data = np.array([sub_data], dtype=np.float32) # Wrap scalar in array
                         except (TypeError, ValueError):
                              logger.warning(f"  Warning: Could not convert scalar to np.float32 for sub_key '{sub_key}'. Skipping this sub_key for this item.")
                              continue # Skip this sub_key for this item


                    if sub_key not in aggregated_data:
                        aggregated_data[sub_key] = []

                    # Only append if sub_data is a numpy array now
                    if isinstance(sub_data, np.ndarray):
                         aggregated_data[sub_key].append(sub_data)
                    elif sub_data is None:
                         # Handle None: Append a placeholder? Or ensure alignment later?
                         # For now, let's append None and handle during stacking
                         aggregated_data[sub_key].append(None)
                    # else: skip other types

                valid_items_processed += 1

            elif isinstance(data_payload, np.ndarray):
                # Handle case where data is already a numpy array (simpler data types)
                # Save it under a default 'data' key within the group
                if 'data' not in aggregated_data:
                    aggregated_data['data'] = []
                if 'timestamps' not in aggregated_data:
                    aggregated_data['timestamps'] = []
                aggregated_data['data'].append(data_payload)
                aggregated_data['timestamps'].append(timestamp)
                valid_items_processed += 1

            # Add handling for other simple scalar types if needed, saving under 'data' key
            elif isinstance(data_payload, (int, float, bool, np.number)):
                 if 'data' not in aggregated_data:
                     aggregated_data['data'] = []
                 if 'timestamps' not in aggregated_data:
                     aggregated_data['timestamps'] = []
                 aggregated_data['data'].append(np.array([data_payload], dtype=np.float32)) # Store as numpy array
                 aggregated_data['timestamps'].append(timestamp)
                 valid_items_processed += 1
            elif data_payload is None:
                # Option: Skip None payloads entirely if they provide no info
                logger.warning(f"  Skipping None payload for timestamp {timestamp}")
                continue
            else:
                logger.warning(f"  Warning: Unsupported data payload type: {type(data_payload)}. Skipping.")
                continue

        if valid_items_processed == 0:
             logger.warning(f"[{self.robot_name} - {self.record_key}] No valid data processed from buffer. Nothing to save.")
             self.buffer = [] # Clear buffer even if nothing saved
             return

        # --- NumPy Conversion ---
        final_datasets = {}
        skipped_keys = []
        num_records_to_save = 0

        if 'timestamps' in aggregated_data:
             num_records_to_save = len(aggregated_data['timestamps'])
             try:
                  final_datasets['timestamps'] = np.array(aggregated_data['timestamps'], dtype=np.float64)
             except Exception as e:
                  logger.error(f"  ERROR converting timestamps: {e}. Aborting save.")
                  self.buffer = []
                  return
             del aggregated_data['timestamps'] # Processed timestamps
        else:
             logger.error("  ERROR: No timestamps were aggregated. Aborting save.")
             self.buffer = []
             return


        for key, data_list in aggregated_data.items():
            if len(data_list) != num_records_to_save:
                 logger.warning(f"  Warning: Mismatched record count for key '{key}' ({len(data_list)} vs {num_records_to_save} timestamps). Skipping this key.")
                 skipped_keys.append(key)
                 continue

            # Handle potential None values before stacking (e.g., replace with NaN or zeros)
            processed_list = []
            has_none = False
            first_valid_item = next((item for item in data_list if item is not None), None)

            if first_valid_item is None:
                 logger.warning(f"  Warning: All data for key '{key}' is None. Skipping.")
                 skipped_keys.append(key)
                 continue

            # Determine placeholder based on the first valid item's dtype and shape
            placeholder = None
            if isinstance(first_valid_item, np.ndarray):
                 placeholder = np.full(first_valid_item.shape, np.nan, dtype=first_valid_item.dtype)
                 # Check if dtype supports NaN, otherwise use zero
                 if not np.issubdtype(placeholder.dtype, np.floating):
                      placeholder = np.zeros(first_valid_item.shape, dtype=first_valid_item.dtype)


            for item in data_list:
                if item is None:
                    if placeholder is not None:
                         processed_list.append(placeholder)
                         has_none = True
                    else:
                         # Should not happen if first_valid_item was found, but as safety:
                         logger.error(f"  Error: Cannot create placeholder for None in key '{key}'. Skipping key.")
                         processed_list = None # Mark for skipping
                         break
                elif isinstance(item, np.ndarray):
                     processed_list.append(item)
                else:
                     # This case should ideally not be reached due to earlier checks
                     logger.warning(f"  Warning: Unexpected non-array item '{item}' for key '{key}' during processing. Skipping key.")
                     processed_list = None
                     break

            if processed_list is None:
                 skipped_keys.append(key)
                 continue

            if has_none:
                 logger.info(f"  Note: Replaced None values with placeholders for key '{key}'.")

            # Now, attempt to stack the processed list
            try:
                # Check shapes before stacking
                first_shape = processed_list[0].shape
                if all(x.shape == first_shape for x in processed_list):
                    final_datasets[key] = np.stack(processed_list)
                else:
                    logger.error(f"  ERROR: Inconsistent shapes for key '{key}' after processing Nones. Cannot stack. Skipping key.")
                    skipped_keys.append(key)
            except ValueError as ve:
                 logger.error(f"  ERROR: Stacking failed for key '{key}'. Error: {ve}. Skipping key.")
                 skipped_keys.append(key)
            except Exception as e:
                logger.error(f"[{self.robot_name} - {self.record_key}] FAILED during numpy conversion for key '{key}': {e}. Skipping key.")
                skipped_keys.append(key)

        # --- HDF5 Writing ---
        save_mode = 'a' # Use append mode
        try:
            with h5py.File(self._recorder_file_name, save_mode) as file:
                logger.info(f"[{self.robot_name} - {self.record_key}] Appending {len(final_datasets)} data arrays ({num_records_to_save} records) to group '{self.record_key}' in {self._recorder_file_name}")

                # Create a group for this record_key if it doesn't exist
                group = file.require_group(self.record_key)

                # Get current total before appending
                total_datapoints_in_file = group.attrs.get('total_datapoints_recorded', 0)

                for key, array_data in final_datasets.items():
                    if array_data.size > 0:
                        dataset_name = key # Use the sub-key as the dataset name

                        # Check if dataset exists for appending
                        if dataset_name in group:
                            dset = group[dataset_name]
                            # Check compatibility (dtype, shape except first dim)
                            if dset.dtype == array_data.dtype and dset.shape[1:] == array_data.shape[1:]:
                                 logger.info(f"  - Appending {array_data.shape[0]} records to dataset: '{self.record_key}/{dataset_name}' (New shape: {(dset.shape[0] + array_data.shape[0],) + dset.shape[1:]})")
                                 dset.resize((dset.shape[0] + array_data.shape[0]), axis=0)
                                 dset[-array_data.shape[0]:] = array_data
                            else:
                                 logger.error(f"  ERROR: Cannot append to dataset '{self.record_key}/{dataset_name}'. Incompatible dtype or shape. Existing: {dset.dtype}, {dset.shape}. New: {array_data.dtype}, {array_data.shape}. Skipping.")
                                 continue # Skip incompatible dataset
                        else:
                            # Create new dataset (resizable)
                            # Object dtype check - should be prevented by earlier logic, but double-check
                            if array_data.dtype == object:
                                 logger.error(f"  ERROR: Cannot save object dtype array '{self.record_key}/{dataset_name}' to HDF5. Skipping.")
                                 continue # Skip this dataset

                            logger.info(f"  - Creating dataset: '{self.record_key}/{dataset_name}' with shape {array_data.shape} and dtype {array_data.dtype}")
                            maxshape = (None,) + array_data.shape[1:] # Make it resizable along the first axis
                            group.create_dataset(dataset_name, data=array_data, compression="gzip", compression_opts=6, chunks=True, maxshape=maxshape)
                    else:
                         logger.info(f"  - Skipping empty dataset: {self.record_key}/{key}")

                # Update metadata in the group attributes
                self.num_datapoints += num_records_to_save # Update total count for this recorder instance
                group.attrs['total_datapoints_recorded'] = total_datapoints_in_file + num_records_to_save
                group.attrs['datapoints_in_last_save'] = num_records_to_save
                group.attrs['last_update_timestamp'] = time.time()
                # Add other relevant metadata if needed
                group.attrs['original_record_key'] = self.record_key # Keep track of the top-level key
                group.attrs['robot_name'] = self.robot_name
                group.attrs['zmq_topic'] = self._zmq_topic
                if hasattr(self, 'record_start_time'): group.attrs['record_start_time'] = self.record_start_time
                if hasattr(self, 'record_end_time'): group.attrs['record_end_time'] = self.record_end_time


            logger.info(f"[{self.robot_name} - {self.record_key}] Successfully appended buffer to {self._recorder_file_name}")
        except Exception as e:
             logger.error(f"[{self.robot_name} - {self.record_key}] FAILED to write HDF5 file {self._recorder_file_name}: {e}")
             # Consider how to handle this - maybe retry? For now, data is lost.
        finally:
             # Clear buffer regardless of success or failure to prevent reprocessing
             self.buffer = []

    def __del__(self):
        # Ensure ZMQ subscriber is stopped on deletion
        if hasattr(self, 'subscriber'):
            logger.info(f"[{self.robot_name} - {self.record_key}] Cleaning up ZMQ subscriber...")
            self.subscriber.stop()