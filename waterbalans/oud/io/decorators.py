import functools
import os

def filepath_or_buffer(function):
    @functools.wraps(function)
    def _filepath_or_buffer(filepath_or_buffer, *args, **kwargs):
        if hasattr(filepath_or_buffer, 'read'):
            file = filepath_or_buffer
            return function(file, *args, **kwargs)
        elif os.path.isfile(filepath_or_buffer):
            with open(filepath_or_buffer, "r") as file:
                return function(file, *args, **kwargs)
        else:
            return NotImplementedError
    return _filepath_or_buffer
