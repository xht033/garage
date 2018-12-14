import csv
import datetime
import os
import sys
from abc import ABC, abstractmethod

import dateutil.tz

from garage.misc.console import colorize
from garage.misc.logger import TabularInput, mkdir_p


class LoggerOutput(ABC):
    @abstractmethod
    def log(self, data, prefix='', with_timestamp=True, color=None):
        pass


class StdOutput(LoggerOutput):
    def __init__(self):
        self.accept_types = (str, TabularInput)

    def log(self, data, prefix='', with_timestamp=True, color=None):
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
    def __init__(self, file_name):
        self.accept_types = (str, )

        mkdir_p(os.path.dirname(file_name))
        self._text_log_file = file_name
        self._log_file = open(file_name, 'a')

    def log(self, data, prefix='', with_timestamp=True, color=None):
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
    def __init__(self, file_name):
        self.accept_types = (TabularInput, )

        mkdir_p(os.path.dirname(file_name))
        self._csv_log_file = file_name
        self._log_file = open(file_name, 'w')

        self._tabular_header_written = False

    def log(self, data, prefix='', with_timestamp=True, color=None):
        if not isinstance(data, self.accept_types):
            return

        writer = csv.DictWriter(
            self._log_file, fieldnames=data.get_table_key_set())

        if not self._tabular_header_written:
            writer.writeheader()
            self._tabular_header_written = True
        writer.writerow(data.get_table_dict())
        self._log_file.flush()
