"""This is Garage's logger singleton.

It takes in many different types of input and directs them to the correct
output.
"""
from contextlib import contextmanager


class Logger(object):
    """This is the singleton class that handles logging."""

    def __init__(self):
        self._outputs = []
        self._prefixes = []
        self._prefix_str = ''

    def log(self, data, **kwargs):
        """Magic method that takes in all different types of input."""
        if not self._outputs:
            print('No outputs have been added to the logger.')

        kwargs['prefix'] = self._prefix_str
        if 'with_prefix' in kwargs and not kwargs.with_prefix:
            kwargs['prefix'] = ''

        for output in self._outputs:
            if isinstance(data, output.accept_types):
                output.log_output(data, **kwargs)

    def add_output(self, output):
        """Add a new output to the logger.

        All data that is compatible with this output will be sent there.
        """
        self._outputs.append(output)

    def reset_all(self):
        """Remove all outputs that have been added to this logger."""
        self._outputs.clear()

    def remove_output(self, output_type):
        """Remove all outputs of a given type."""
        self._outputs = [
            output for output in self._outputs
            if not isinstance(output, output_type)
        ]

    def reset_output(self, output):
        """Removes, then re-adds a given output to the logger."""
        self.remove_output(type(output))
        self.add_output(output)

    def has_output(self, output_type):
        """Checks to see if a given logger output is attached to the logger."""
        for output in self._outputs:
            if isinstance(output, output_type):
                return True
        return False

    def get_output(self, output_type):
        """Returns the first output of the given type found."""
        for output in self._outputs:
            if isinstance(output, output_type):
                return output

    def dump_output(self, output_type, step=None):
        """Dumps all outputs of the given type."""
        for output in self._outputs:
            if isinstance(output, output_type):
                output.dump(step=step)

    @contextmanager
    def prefix(self, key):
        """Add a prefix to the logger.

        This allows text output to be prepended with a given stack of prefixes.

        Example:
        with logger.prefix('prefix: '):
            logger.log('test_string') # this will have the prefix
        logger.log('test_string2') # this will not have the prefix
        """
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    def push_prefix(self, prefix):
        """Add prefix to prefix stack."""
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def pop_prefix(self):
        """Pop prefix from prefix stack."""
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)
