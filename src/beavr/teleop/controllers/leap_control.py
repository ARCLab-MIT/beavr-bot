import numpy as np
from dynamixel_sdk import (
    PortHandler,
    PacketHandler, 
    GroupBulkWrite, 
    GroupBulkRead, 
    COMM_SUCCESS, 
    DXL_LOBYTE, 
    DXL_LOWORD, 
    DXL_HIBYTE, DXL_HIWORD)
import time
import threading
from queue import Queue
import logging

logger = logging.getLogger(__name__)

# Control table addresses (for XL330-M288)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126
ADDR_OPERATING_MODE = 11

# Data Lengths
LEN_PRESENT_POSITION = 4  # Position data is 4 bytes
LEN_PRESENT_VELOCITY = 4  # Velocity data is 4 bytes
LEN_PRESENT_CURRENT = 2   # Current data is 2 bytes

# Protocol version
PROTOCOL_VERSION = 2.0

# Default settings
BAUDRATE = 4000000  # 4 Mbps
DEVICENAME = '/dev/ttyUSB0'   # Change based on your setup

LEAP_HOME_VALUES = [2048] * 16  # Neutral position for all motors

MOTOR_IDS = [0, 1, 2, 3,        # Index
             4, 5, 6, 7,        # Middle
             8, 9, 10, 11,      # Ring
             12, 13, 14, 15 ]   # Thumb

class DynamixelClient:
    def __init__(self):
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.motor_ids = MOTOR_IDS

        if self.portHandler.openPort():
            logger.info("Succeeded to open the port")
        else:
            raise RuntimeError("Failed to open the port")

        if self.portHandler.setBaudRate(BAUDRATE):
            logger.info("Succeeded to set the baudrate")
        else:
            raise RuntimeError("Failed to change the baudrate")

        # Initialize bulk write for positions
        self.groupBulkWrite = GroupBulkWrite(self.portHandler, self.packetHandler)

        # Enable torque for all motors
        for motor_id in self.motor_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, ADDR_TORQUE_ENABLE, 1)

        # Initialize separate bulk read handlers for each data type
        self.groupBulkReadPosition = GroupBulkRead(self.portHandler, self.packetHandler)
        self.groupBulkReadVelocity = GroupBulkRead(self.portHandler, self.packetHandler)
        self.groupBulkReadCurrent = GroupBulkRead(self.portHandler, self.packetHandler)
        
        # Add parameters for each bulk read handler
        for motor_id in self.motor_ids:
            # Position (existing)
            if not self.groupBulkReadPosition.addParam(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                logger.error(f"[ID:{motor_id}] groupBulkRead addparam failed for position")
            # Velocity
            if not self.groupBulkReadVelocity.addParam(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                logger.error(f"[ID:{motor_id}] groupBulkRead addparam failed for velocity")
            # Current
            if not self.groupBulkReadCurrent.addParam(motor_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT):
                logger.error(f"[ID:{motor_id}] groupBulkRead addparam failed for current")

    def _read_motor_value(self, motor_id, address, byte_length):
        """ Reads a value from a motor register """
        if byte_length == 2:
            value, _, _ = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id, address)
        elif byte_length == 4:
            value, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, address)
        else:
            raise ValueError("Invalid byte length for reading.")
        return value

    def _convert_radians_to_raw(self, radians):
        """Convert radians to Dynamixel raw values (0-4095)"""
        return np.clip(((radians + np.pi) / (2 * np.pi)) * 4095, 0, 4095).astype(int)

    def bulk_write_positions(self, positions):
        """Write all positions at once using Protocol 2.0 Bulk Write"""
        try:
            # Clear any existing parameters
            self.groupBulkWrite.clearParam()
            
            # Convert radians to raw values
            raw_positions = self._convert_radians_to_raw(positions)
            
            # Add parameters for each motor
            for i, motor_id in enumerate(self.motor_ids):
                position = raw_positions[i]  # Already converted to raw int
                param = [
                    DXL_LOBYTE(DXL_LOWORD(position)),
                    DXL_HIBYTE(DXL_LOWORD(position)),
                    DXL_LOBYTE(DXL_HIWORD(position)),
                    DXL_HIBYTE(DXL_HIWORD(position))
                ]
                result = self.groupBulkWrite.addParam(motor_id, ADDR_GOAL_POSITION, 4, param)
                if result != True:
                    logger.error(f"Failed to add parameter for motor {motor_id}")
                    return False
            
            # Execute bulk write
            result = self.groupBulkWrite.txPacket()
            if result != COMM_SUCCESS:
                logger.error(f"Failed to execute bulk write: {self.packetHandler.getTxRxResult(result)}")
                return False
                
            return True

        except Exception as e:
            logger.error(f"Bulk write failed: {e}")
            return False

    def set_motor_positions(self, positions):
        """Replace individual writes with bulk write"""
        return self.bulk_write_positions(positions)

    def _bulk_read_values(self, group_handler, address, length, default_value):
        """Generic bulk read function for any data type"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                comm_result = group_handler.txRxPacket()
                if comm_result != COMM_SUCCESS:
                    logger.error(f"Bulk read failed (attempt {attempt + 1}/{max_retries}): {self.packetHandler.getTxRxResult(comm_result)}")
                    continue

                values = []
                data_valid = True
                for motor_id in self.motor_ids:
                    if group_handler.isAvailable(motor_id, address, length):
                        values.append(group_handler.getData(motor_id, address, length))
                    else:
                        data_valid = False
                        break

                if data_valid:
                    return np.array(values, dtype=np.float32)

            except Exception as e:
                logger.error(f"Bulk read error (attempt {attempt + 1}/{max_retries}): {e}")

        return np.array([default_value] * len(self.motor_ids), dtype=np.float32)

    def get_motor_positions(self):
        """Get all motor positions using bulk read"""
        return self._bulk_read_values(self.groupBulkReadPosition, ADDR_PRESENT_POSITION, 
                                    LEN_PRESENT_POSITION, 2048)  # Uses LEAP_HOME_VALUE as default

    def get_motor_velocities(self):
        """Get all motor velocities using bulk read"""
        return self._bulk_read_values(self.groupBulkReadVelocity, ADDR_PRESENT_VELOCITY, 
                                    LEN_PRESENT_VELOCITY, 0)

    def get_motor_torques(self):
        """Get all motor torques using bulk read"""
        return self._bulk_read_values(self.groupBulkReadCurrent, ADDR_PRESENT_CURRENT, 
                                    LEN_PRESENT_CURRENT, 0)

    def close(self):
        # Disable torque before closing
        for motor_id in self.motor_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, ADDR_TORQUE_ENABLE, 0)
        self.portHandler.closePort()

    def __del__(self):
        # Clean up bulk write group
        if hasattr(self, 'groupBulkWrite'):
            self.groupBulkWrite.clearParam()
        # Clean up all bulk read groups
        if hasattr(self, 'groupBulkReadPosition'):
            self.groupBulkReadPosition.clearParam()
        if hasattr(self, 'groupBulkReadVelocity'):
            self.groupBulkReadVelocity.clearParam()
        if hasattr(self, 'groupBulkReadCurrent'):
            self.groupBulkReadCurrent.clearParam()
        # ... rest of cleanup code ...

class DexArmControl:
    def __init__(self):
        self._client = DynamixelClient()
        self._last_successful_position = np.zeros(16)
        self._command_queue = Queue(maxsize=1)  # Only keep latest command
        self._movement_thread = threading.Thread(target=self._execute_movement_loop, daemon=True)
        self._movement_thread.start()

    def _convert_radians_to_raw(self, radians):
        """ Converts angles from [-pi, pi] to raw Dynamixel values [0, 4095] """
        return np.clip(((radians + np.pi) / (2 * np.pi)) * 4095, 0, 4095).astype(int)
    
    def _convert_raw_to_radians(self, raw_values):
        """ Converts raw Dynamixel values [0, 4095] to angles in [-pi, pi] """
        if raw_values is None:
            return np.zeros(len(self._client.motor_ids), dtype=np.float32)
        return ((raw_values / 4095) * (2 * np.pi)) - np.pi
    
    def _convert_rpm_to_rad_per_sec(self, rpm):
        """ Converts RPM to rad/s """
        return rpm * (2 * np.pi) / 60

    def move_hand(self, target_joints_radians):
        """Queue up movement command"""
        try:
            # Update queue with latest command, dropping old ones
            while not self._command_queue.empty():
                self._command_queue.get_nowait()
            self._command_queue.put(target_joints_radians)
            return True
        except Exception as e:
            logger.error(f"Failed to queue movement: {e}")
            return False

    def _execute_movement_loop(self):
        """Background thread that executes bulk writes"""
        while True:
            try:
                if not self._command_queue.empty():
                    positions = self._command_queue.get()
                    # Use Protocol 2.0 Bulk Write (0x93) to send all positions at once
                    success = self._client.bulk_write_positions(positions)
                    if success:
                        self._last_successful_position = positions
                time.sleep(0.02)  # 50Hz max rate
            except Exception as e:
                logger.error(f"Movement error: {e}")
                time.sleep(0.1)

    def get_hand_position(self):
        """Returns cached position instead of reading"""
        return self._last_successful_position

    def get_hand_state(self):
        """Returns cached state"""
        return {
            "position": self._last_successful_position,
            "velocity": np.zeros(16),  # We don't need velocity feedback
            "effort": np.zeros(16)     # We don't need effort feedback
        }

    def get_commanded_hand_state(self):
        """Returns last commanded position"""
        return {"position": self._last_successful_position}

    def home_hand(self):
        self.move_hand(np.zeros(len(self._client.motor_ids)))  # Move to zero radians (2048 raw value)

    def reset_hand(self):
        self.home_hand()

    def close(self):
        self._client.close()

    def get_hand_velocity(self):
        return self.get_hand_state()["velocity"]

    def get_hand_torque(self):
        return self.get_hand_state()["effort"]

    def get_commanded_hand_joint_position(self):
        return self.get_commanded_hand_state()["position"]