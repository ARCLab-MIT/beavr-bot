"""
Beavr Teleop Package

This package provides teleoperation capabilities for robotic systems.
The main entry point is now in main.py, and configuration is handled in configs/models.py.
"""

import logging

# Import the new structured configuration for backward compatibility
from beavr.teleop.configs.constants.models import TeleopConfig, NetworkConfig, PortsConfig, RobotConfig, CameraConfig

# Note: main functions are not imported here to avoid circular imports when using python -m
# Import them directly: from beavr.teleop.main import run_teleop, main

logger = logging.getLogger(__name__)


def _setup_root_logger(level: int = logging.DEBUG):
    """Configure the *root* logger only once (no-op if already configured).
    
    NOTE: This function is maintained here for backward compatibility.
    New code should use the version in main.py directly.
    """
    root = logging.getLogger()
    if root.handlers:
        # A configuration already exists – just raise the level if needed.
        if root.level > level:
            root.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s %(processName)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# Backward compatibility aliases for legacy code
# TODO: Remove these once all code is migrated to the new structured config
LegacyTeleopConfig = TeleopConfig  # Alias for gradual migration


__all__ = [
    # New structured configuration (preferred)
    "TeleopConfig",
    "NetworkConfig", 
    "PortsConfig",
    "RobotConfig",
    "CameraConfig",
    
    # Utility functions
    "_setup_root_logger",
    
    # Backward compatibility (deprecated)
    "LegacyTeleopConfig",
]