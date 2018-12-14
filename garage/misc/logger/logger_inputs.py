"""Contains inputs for the logger.

Each class takes in input and stores it for later use by the logger.
Inputs may be passed to the logger via its log() method.
"""
from contextlib import contextmanager

import numpy as np

from garage.misc.tabulate import tabulate


class TabularInput(object):
    """This class allows the user to create tables for easy display."""

    def __init__(self):
        self._tabular = []
        self._no_prefix_dict = {}
        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

    def __str__(self):
        """Returns a string representation of the table for the logger."""
        return tabulate(self._tabular)

    def record_tabular(self, key, val):
        """Allows the user to save key/value entries for the table."""
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))
        self._no_prefix_dict[key] = val

    def record_tabular_misc_stat(self, key, values, placement='back'):
        """Allows the user to record statistics of an array."""
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
        """Handles pushing and popping of a tabular prefix.

        Can be used in the following way:

        with tabular.tabular_prefix('your_prefix_'):
            # your code
            tabular.record_tabular(key, val)
        """
        self.push_tabular_prefix(key)
        try:
            yield
        finally:
            self.pop_tabular_prefix()

    def clear(self):
        """Clears the tabular."""
        self._tabular.clear()

    def get_table(self):
        """Returns the string representation of the table and clears it."""
        self.clear()
        return self.__str__()

    def push_tabular_prefix(self, key):
        """Push prefix to be appended before printed table."""
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        """Pop prefix that was appended to the printed table."""
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def get_table_dict(self):
        """Returns a dictionary of the tabular items."""
        return dict(self._tabular)

    def get_no_prefix_dict(self):
        """Returns dictionary without prefixes."""
        return self._no_prefix_dict

    def get_table_key_set(self):
        """Returns a set of the table's keys."""
        return set(dict(self._tabular).keys())
