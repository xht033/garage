from contextlib import contextmanager

import numpy as np

from garage.misc.tabulate import tabulate


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
            if isinstance(data, output.accept_types):
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


class TabularInput(object):
    def __init__(self):
        self._tabular = []
        self._no_prefix_dict = {}
        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

    def __str__(self):
        return tabulate(self._tabular)

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))
        self._no_prefix_dict[key] = val

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

    def get_no_prefix_dict(self):
        return self._no_prefix_dict

    def get_table_key_set(self):
        return set(dict(self._tabular).keys())
