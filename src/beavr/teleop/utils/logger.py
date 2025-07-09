import json
import csv
from datetime import datetime
import os
import numpy as np
import atexit
import glob
import time
import logging

logger = logging.getLogger(__name__)


class BaseLogger:
    """Base class for all loggers."""
    def __init__(self, log_dir="logs", prefix=""):
        self.log_dir = log_dir
        self.prefix = prefix
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ensure_log_dir()
        self.data = {}  # Use dict with timestamp as key instead of list
        self.log_file = os.path.join(self.log_dir, f"{prefix}_log_{self.session_id}.json")
        self.write_batch_size = 5  # Save every 5 frames
        self.total_frames = 0  # Track total frames across saves
        
        # Clean up any existing temp files from previous runs
        self._cleanup_temp_files()
        
        # Create the initial empty file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "frames": {},
                    "total_frames": 0
                }, f, indent=2)
        logger.info(f"Logger initialized. Writing to: {self.log_file}")
        
        # Register cleanup function
        atexit.register(self._cleanup)
        
        self._closed = False  # Add closed flag
        
    def ensure_log_dir(self):
        """Create logs directory if it doesn't exist."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def _convert_numpy_to_list(self, data):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data
    
    def _cleanup_temp_files(self):
        """Clean up any temporary JSON files in the log directory."""
        # Find all temp files for this prefix
        temp_pattern = os.path.join(self.log_dir, f"{self.prefix}_log_*.json.tmp")
        temp_files = glob.glob(temp_pattern)
        
        if temp_files:
            logger.info(f"Found {len(temp_files)} temporary log files. Cleaning up...")
            for temp_file in temp_files:
                try:
                    # Get the corresponding main file
                    main_file = temp_file[:-4]  # Remove .tmp extension
                    
                    # If the main file exists, check if we can merge data
                    if os.path.exists(main_file):
                        try:
                            # Try to load and merge data from temp file
                            with open(temp_file, 'r') as f:
                                temp_data = json.load(f)
                            
                            with open(main_file, 'r') as f:
                                main_data = json.load(f)
                                
                            # Merge frames from temp into main
                            if 'frames' in temp_data and 'frames' in main_data:
                                main_data['frames'].update(temp_data['frames'])
                                main_data['total_frames'] = len(main_data['frames'])
                                
                                # Save merged data
                                with open(main_file, 'w') as f:
                                    json.dump(main_data, f, indent=2)
                                logger.info(f"Merged data from {temp_file} into {main_file}")
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Could not merge data from {temp_file}: {e}")
                    
                    # Delete the temp file
                    os.remove(temp_file)
                    logger.info(f"Deleted temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {temp_file}: {e}")

    def _save_json(self, filename):
        """Save the logged data to a JSON file."""
        temp_filename = None
        try:
            file_data = {
                "session_id": self.session_id,
                "frames": {},
                "total_frames": 0
            }
            
            # If file exists, read it first
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    try:
                        file_data = json.load(f)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Warning: Could not load existing data: {str(e)}")
                        logger.info("Starting fresh")
            
            # Update with new data
            file_data['frames'].update(self.data)
            file_data['total_frames'] = len(file_data['frames'])

            # Write to a temporary file first, then rename it
            temp_filename = filename + '.tmp'
            with open(temp_filename, 'w') as f:
                json.dump(file_data, f, indent=2)
                # Ensure all data is written to disk
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename operation
            os.replace(temp_filename, filename)
            
            return True
        except Exception as e:
            logger.error(f"Error saving log file: {e}")
            # Only try to remove if temp_filename is defined
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)  # Clean up temp file if it exists
                    logger.info(f"Cleaned up temp file {temp_filename} after error")
                except Exception as cleanup_err:
                    logger.error(f"Failed to clean up temp file: {cleanup_err}")
            return False

    def export_csv(self, filename):
        """Export the logged data to CSV format."""
        if not self.data:
            return
            
        # Flatten the data structure
        csv_data = []
        for frame in self.data.values():
            flat_data = {
                "timestamp": frame["timestamp"],
                "frame": frame["frame"]
            }
            
            # Flatten each data array
            for key, value in frame.items():
                if isinstance(value, (list, np.ndarray)):
                    for i, v in enumerate(value):
                        flat_data[f"{key}_{i}"] = v
                        
            csv_data.append(flat_data)
            
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    def _cleanup(self):
        """Ensure all data is saved when the process exits"""
        if not self._closed:
            self.close()

    def close(self):
        """Explicitly close the logger and save all remaining data."""
        if self._closed:
            return
            
        if self.data:  # Save any remaining data
            logger.info(f"\nSaving final {len(self.data)} frames for {self.prefix}...")
            try:
                success = self._save_json(self.log_file)
                if success:
                    self.data = {}
                    logger.info(f"Final log saved successfully: {self.log_file}")
                else:
                    logger.warning("Warning: Failed to save final log data")
            except Exception as e:
                logger.error(f"Error during close: {e}")
        
        # Clean up any remaining temp files
        self._cleanup_temp_files()
        
        self._closed = True

class PoseLogger(BaseLogger):
    """Logger for pose transformations."""
    def __init__(self, log_dir="logs", prefix="operator"):
        super().__init__(log_dir, prefix)
        self.log_file = os.path.join(self.log_dir, f"{prefix}_log_{self.session_id}.json")
        self.frame_count = 0
        
    def log_frame(self, init_hand_pose, init_robot_pose, current_hand_pose, transformed_pose):
        """Log a single frame of pose data."""
        if self._closed:
            raise RuntimeError("Cannot log to closed logger")
            
        frame_data = {
            "init_hand_pose": self._convert_numpy_to_list(init_hand_pose),
            "init_robot_pose": self._convert_numpy_to_list(init_robot_pose),
            "current_hand_pose": self._convert_numpy_to_list(current_hand_pose),
            "transformed_pose": self._convert_numpy_to_list(transformed_pose),
            "timestamp": time.time(),
            "frame": self.frame_count
        }
        # logger.info(f"Logging frame {self.frame_count} for {self.prefix}")
        self.data[str(self.frame_count)] = frame_data  # Use frame number as key
        self.frame_count += 1  # Increment frame counter
        
        # Save every 5 frames
        if self.frame_count % self.write_batch_size == 0:
            if self._save_json(self.log_file):
                self.data = {}  # Only clear data after successful save

class HandLogger(BaseLogger):
    """Logger for hand data."""
    def __init__(self, log_dir="logs", prefix="hand"):
        super().__init__(log_dir, prefix)
        self.log_file = os.path.join(self.log_dir, f"{prefix}_log_{self.session_id}.json")
        self.frame_count = 0
        
    def log_frame(self, finger_input_positions, finger_computed_angles, finger_states=None):
        """Log hand state data with detailed finger information."""
        if self._closed:
            raise RuntimeError("Cannot log to closed logger")
            
        frame_data = {
            "timestamp": time.time(),
            "frame": self.frame_count,
            "finger_input_positions": {
                k: self._convert_numpy_to_list(v) 
                for k, v in finger_input_positions.items()
            },
            "finger_computed_angles": {
                k: self._convert_numpy_to_list(v) 
                for k, v in finger_computed_angles.items()
            }
        }
        
        if finger_states is not None:
            frame_data["finger_states"] = self._convert_numpy_to_list(finger_states)
            
        self.data[str(self.frame_count)] = frame_data
        self.frame_count += 1

        # Save every 5 frames
        if self.frame_count % self.write_batch_size == 0:
            if self._save_json(self.log_file):
                self.data = {}

class RobotLogger(BaseLogger):
    """Logger for robot state."""
    def __init__(self, log_dir="logs", prefix="sim"):
        super().__init__(log_dir, prefix)
        self.log_file = os.path.join(self.log_dir, f"{prefix}_log_{self.session_id}.json")
        self.frame_count = 0  # Add frame counter
        
    def log_frame(self, end_effector_pose, action_pose, IK_calculated_joints):
        """Log robot state data."""
        if self._closed:
            raise RuntimeError("Cannot log to closed logger")
            
        frame_data = {
            "end_effector_pose": self._convert_numpy_to_list(end_effector_pose),
            "action_pose": self._convert_numpy_to_list(action_pose),
            "IK_calculated_joints": self._convert_numpy_to_list(IK_calculated_joints),
            "timestamp": time.time(),
            "frame": self.frame_count
        }

        # logger.info(f"Logging frame {self.frame_count} for {self.prefix}")
            
        self.data[str(self.frame_count)] = frame_data
        self.frame_count += 1

        # Save every 5 frames
        if self.frame_count % (self.write_batch_size*2) == 0:
            if self._save_json(self.log_file):
                self.data = {}

def synchronize_logs(log_files, output_file):
    """Synchronize multiple log files based on timestamps."""
    all_data = {}
    
    # Load all log files
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
            prefix = os.path.basename(log_file).split('_')[0]
            all_data[prefix] = data['frames']
    
    # Find common timestamps
    timestamps = set.intersection(*[set(d.keys()) for d in all_data.values()])
    
    # Create synchronized data
    synced_data = {}
    for ts in sorted(timestamps):
        synced_data[ts] = {
            prefix: data[ts]
            for prefix, data in all_data.items()
        }
    
    # Save synchronized data
    with open(output_file, 'w') as f:
        json.dump(synced_data, f, indent=2) 

def setup_root_logger(level: int = logging.DEBUG):
    """Configure the root logger only once (no-op if already configured)."""
    root = logging.getLogger()
    if root.handlers:
        # Configuration already exists – just raise the level if needed
        if root.level > level:
            root.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s %(processName)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )