import json
import os
from contextlib import contextmanager
from enum import Enum

import numpy as np

from garage.misc.console import mkdir_p
from garage.misc.tabulate import tabulate


def dump_variant(log_file, variant_data):
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum':
                o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {'$function': o.__module__ + "." + o.__name__}
        return json.JSONEncoder.default(self, o)


class TabularInput(object):
    def __init__(self):
        self._tabular = []
        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

    def __str__(self):
        return tabulate(self._tabular)

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if values:
            self.record_tabular(prefix + "Average" + suffix,
                                np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        try:
            yield
        finally:
            self.pop_tabular_prefix()

    def clear(self):
        self._tabular.clear()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def get_table_dict(self):
        return dict(self._tabular)

    def get_table_key_set(self):
        return set(dict(self._tabular).keys())


class Logger(object):
    def __init__(self):
        self._outputs = []
        self._prefixes = []
        self._prefix_str = ''

    def log(self, data, with_prefix=True, with_timestamp=True, color=None):
        prefix = ''
        if with_prefix:
            prefix = self._prefix_str
        for output in self._outputs:
            output.log(
                data,
                prefix=prefix,
                with_timestamp=with_timestamp,
                color=color)

    def add_output(self, output):
        self._outputs.append(output)

    def reset_output(self):
        self._outputs.clear()

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def pop_prefix(self):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)


logger = Logger()
