from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)


class Component(ABC):
    @abstractmethod
    def stream(self):
        raise NotImplementedError()

    def notify_component_start(self, component_name):
        if component_name:
            logger.info("***************************************************************")
            logger.info(f"     Starting {component_name} component")
            logger.info("***************************************************************")