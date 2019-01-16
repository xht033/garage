"""Global logger module.
This module instantiates a global logger singleton.
"""
from garage.misc.logger.logger_inputs import TabularInput
from garage.misc.logger.logger_outputs import CsvOutput, StdOutput, TextOutput
from garage.misc.logger.singleton_logger import Logger
from garage.misc.logger.tensorboard_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()

__all__ = [
    "tabular", "logger", "CsvOutput", "StdOutput", "TextOutput",
    "TensorBoardOutput"
]
