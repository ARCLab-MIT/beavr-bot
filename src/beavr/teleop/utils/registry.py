# Simple registry for component access across processes
import logging

logger = logging.getLogger(__name__)


class GlobalRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, name, instance):
        """Register a component by name"""
        cls._registry[name] = instance
        logger.info(f"Registered {name} in global registry")
    
    @classmethod
    def get(cls, name):
        """Get a component by name"""
        return cls._registry.get(name) 