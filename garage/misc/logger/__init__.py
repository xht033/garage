"""Global logger module.
This module instantiates a global logger singleton.
"""
from garage.misc.logger.singleton_logger import Logger, TabularInput

logger = Logger()
tabular = TabularInput()

__all__ = ["logger", "tabular"]
