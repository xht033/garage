"""Global logger module.
This module instantiates a global logger singleton.
"""
from garage.misc.logger.logger_inputs import TabularInput
from garage.misc.logger.singleton_logger import Logger

logger = Logger()
tabular = TabularInput()

__all__ = ["logger", "tabular"]
