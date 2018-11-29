import csv
import datetime
import os
import sys
from abc import ABC, abstractmethod
from os import path as osp

import dateutil.tz

from garage import config
from garage.misc.console import colorize
from garage.misc.logger import TabularInput, mkdir_p


class LoggerOutput(ABC):
    @abstractmethod
    def log(self, data, prefix='', with_timestamp=True, color=None):
        pass


class StdOutput(LoggerOutput):
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
        text_log_file = osp.join(config.LOG_DIR, file_name)
        mkdir_p(os.path.dirname(text_log_file))
        self._text_log_file = text_log_file
        self._log_file = open(text_log_file, 'a')

    def log(self, data, prefix='', with_timestamp=True, color=None):
        if not isinstance(data, str):
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
        csv_log_file = osp.join(config.LOG_DIR, file_name)
        mkdir_p(os.path.dirname(csv_log_file))
        self._csv_log_file = csv_log_file
        self._log_file = open(csv_log_file, 'w')

        self._tabular_header_written = False

    def log(self, data, prefix='', with_timestamp=True, color=None):
        if not isinstance(data, TabularInput):
            return

        writer = csv.DictWriter(
            self._log_file, fieldnames=data.get_table_key_set())

        if not self._tabular_header_written:
            writer.writeheader()
            self._tabular_header_written = True
        writer.writerow(data.get_table_dict())
        self._log_file.flush()
