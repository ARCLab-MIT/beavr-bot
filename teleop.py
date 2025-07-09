from beavr.teleop import TeleopConfig
from beavr.teleop.main import main

__all__ = [
    "TeleopConfig",
    "main",
]

if __name__ == "__main__":
    """
    Convenience wrapper for Beavr Teleop system.
    
    Examples:
        # Single robot
        python teleop.py --robot_name=leap --laterality=right
        python teleop.py --robot_name=xarm7 --laterality=left
        
        # Multiple robots (composite configuration)
        python teleop.py --robot_name=leap,xarm7 --laterality=right
        python teleop.py --robot_name=leap,xarm7 --laterality=bimanual
        
        # With environment config
        python teleop.py --robot_name=leap,xarm7 --config_file=config/prod.yaml
    """
    # Delegate CLI execution.
    main()