"""Contains the output classes for the logger.

Each class is sent logger data and handles it itself.
"""
from abc import ABC, abstractmethod
import csv
import datetime
import os
import sys

import dateutil.tz

from garage.misc.console import colorize, mkdir_p
from garage.misc.logger.singleton_tabular import TabularInput


class LoggerOutput(ABC):
    """Abstract class for Logger Outputs."""

    @abstractmethod
    def log_output(self, data, **kwargs):
        """This method is called by the logger when it needs to pass data."""
        pass

    def dump(self, step=None):
        """This method is called by the logger to dump an output to file."""
        pass


class StdOutput(LoggerOutput):
    """Standard console output for the logger."""

    def __init__(self):
        self.accept_types = (str, TabularInput)

    def log_output(self,
                   data,
                   prefix='',
                   with_timestamp=True,
                   color=None,
                   **kwargs):
        """Log data to console."""
        out = ''
        if isinstance(data, str):
            out = prefix + data
            if with_timestamp:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                out = "%s | %s" % (timestamp, out)
            if color is not None:
                out = colorize(out, color)
        elif isinstance(data, TabularInput):
            out = str(data)

        print(out)
        sys.stdout.flush()


class TextOutput(LoggerOutput):
    """Text file output for logger."""

    def __init__(self, file_name):
        self.accept_types = (str, )

        mkdir_p(os.path.dirname(file_name))
        self._text_log_file = file_name
        self._log_file = open(file_name, 'a')

    def log_output(self, data, with_timestamp=True, **kwargs):
        """Log data to text file."""
        if not isinstance(data, self.accept_types):
            return

        out = data
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)

        self._log_file.write(out + '\n')
        self._log_file.flush()


class CsvOutput(LoggerOutput):
    """CSV file output for logger."""

    def __init__(self, file_name):
        self.accept_types = (TabularInput, )

        mkdir_p(os.path.dirname(file_name))
        self._log_file = open(file_name, 'w')

        self._tabular_header_written = False

    def log_output(self, data, prefix='', **kwargs):
        """Log tabular data to CSV."""
        if not isinstance(data, self.accept_types):
            return

        writer = csv.DictWriter(
            self._log_file, fieldnames=data.get_table_key_set())

        if not self._tabular_header_written:
            writer.writeheader()
            self._tabular_header_written = True
        writer.writerow(data.get_table_dict())
        self._log_file.flush()
