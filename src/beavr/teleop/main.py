#!/usr/bin/env python3
"""
Main entry point for Beavr Teleop system.

This module provides the CLI interface and main execution logic for the teleoperation system.
Uses the structured configuration system with automatic CLI flag generation via Draccus.
"""

import draccus
import logging
from typing import Any
from dataclasses import dataclass, field


from beavr.teleop.configs.constants.models import TeleopConfig
from beavr.teleop.utils.configs import (
    Laterality,
    load_robot_config,
    load_yaml_config,
    apply_yaml_preserving_cli,
)
from beavr.teleop.utils.logger import setup_root_logger

logger = logging.getLogger(__name__)


@dataclass
class MainConfig:
    """Main configuration combining structured teleop config with robot selection."""
    
    # Use our new structured config as the base
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    
    # Robot selection - supports comma-separated multiple robots
    robot_name: str = ""

    # Laterality setting for robot configuration
    laterality: str = "right"  # Options: "right", "left", "bimanual"

    # Optional config file override
    config_file: str = "config/dev.yaml"
    
    # Data storage configuration
    storage_path: str = "data/recordings"
    
    # Built robot structure (populated at runtime)
    robot: Any = field(init=False)
    
    def __post_init__(self):
        """Initialize robot configuration based on robot_name(s)."""
        if not self.robot_name or self.robot_name == "":
            raise ValueError(
                "robot_name must be provided. Examples:\n"
                "  Single robot: --robot_name=leap\n"
                "  Multiple robots: --robot_name=leap,xarm7\n"
                "Available robots: leap, xarm7, etc."
            )
        
        # Convert to enum for internal use
        self.laterality_enum = Laterality(self.laterality)
        
        # Load robot configuration(s) using utility
        self.robot = load_robot_config(self.robot_name, self.laterality_enum)
    
    # Convenience attribute delegation for backward compatibility
    def __getattr__(self, item):
        """Delegate unknown attributes to teleop config for backward compatibility."""
        try:
            return getattr(self.teleop, item)
        except AttributeError:
            # Try teleop.network for flattened network access
            try:
                return getattr(self.teleop.network, item)
            except AttributeError as exc:
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'") from exc


def run_teleop(config: MainConfig):
    """Run the teleoperation system with given configuration."""
    
    # Setup logging
    setup_root_logger(logging.DEBUG)
    
    logger.info("🚀 Starting Beavr Teleop System")
    logger.info(f"📡 Network host: {config.teleop.network.host_address}")
    logger.info(f"🎮 Operation mode: {'ENABLED' if config.teleop.flags.operate else 'DISABLED'}")
    logger.info(f"🎯 Simulation mode: {'ENABLED' if config.teleop.flags.sim_env else 'DISABLED'}")
    logger.info(f"🤖 Robot: {config.robot_name}")
    
    from beavr.teleop.components import TeleOperator  # pylint: disable=import-error,cyclic-import

    # Initialize the teleoperator with structured config
    teleop = TeleOperator(config)
    processes = teleop.get_processes()

    try:
        # Start all processes
        logger.info(f"🔄 Starting {len(processes)} teleop processes...")
        for process in processes:
            process.start()
            logger.debug(f"  ✅ Started process: {process.name}")

        # Wait for all processes to complete
        logger.info("✨ All processes started. Press Ctrl+C to stop.")
        while any(p.is_alive() for p in processes):
            for p in processes:
                p.join(timeout=0.1)  # Short timeout to allow checking KeyboardInterrupt

    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown requested...")
        
        # First send stop signal to all processes
        for p in processes:
            if p.is_alive():
                p.terminate()
        
        # Then wait for them to finish with timeout
        for p in processes:
            p.join(timeout=2.0)
            
        # Force kill any remaining processes
        for p in processes:
            if p.is_alive():
                logger.warning(f"Process {p.name} did not terminate gracefully - force killing")
                p.kill()
                p.join(timeout=1.0)
    
    finally:
        # Final cleanup
        for p in processes:
            if p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=0.5)
                except Exception as e:
                    logger.error(f"Error cleaning up process {p.name}: {e}")
        
        logger.info("🏁 Teleop shutdown complete")


@draccus.wrap()
def main(cfg: MainConfig):
    """
    Main entry point for Beavr Teleop system.
    
    This function is wrapped with Draccus to automatically generate CLI flags for all
    configuration parameters. Configuration precedence (highest to lowest):
    1. CLI flags (via Draccus)
    2. YAML config file overrides  
    3. Default values
    
    Examples:
        # Single robot usage
        python -m beavr.teleop.main --robot_name=leap --laterality=right
        python -m beavr.teleop.main --robot_name=xarm7 --laterality=left
        
        # Multiple robots (composite configuration)
        python -m beavr.teleop.main --robot_name=leap,xarm7 --laterality=right
        python -m beavr.teleop.main --robot_name=leap,xarm7 --laterality=bimanual
        
        # Use production config
        python -m beavr.teleop.main --robot_name=leap,xarm7 --config_file=config/prod.yaml
        
        # Override network settings via CLI (highest priority)
        python -m beavr.teleop.main --robot_name=leap --teleop.network.host_address=192.168.1.100
        
        # Enable simulation mode
        python -m beavr.teleop.main --robot_name=leap --teleop.flags.sim_env=True
        
        # Override control parameters
        python -m beavr.teleop.main --robot_name=leap,xarm7 --teleop.control.vr_freq=60
        
        # Override multiple port settings
        python -m beavr.teleop.main --robot_name=xarm7 \
            --teleop.ports.keypoint_stream_port=9000 \
            --teleop.ports.control_stream_port=9001
    """
    
    # Apply YAML configuration overrides
    # Note: CLI flags (from Draccus) already applied, YAML merges underneath
    yaml_overrides = load_yaml_config(cfg.config_file)
    if yaml_overrides:
        logger.info(f"🔧 Applying YAML overrides from {cfg.config_file}")
        
        # Apply YAML overrides while preserving CLI flag precedence
        apply_yaml_preserving_cli(cfg, yaml_overrides)
    
    run_teleop(cfg)


if __name__ == "__main__":
    main()