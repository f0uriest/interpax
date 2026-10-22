"""Fixtures etc for testing."""

import inspect
import warnings

_original_warn = warnings.warn


def smart_warn(message, category=None, stacklevel=1, source=None):
    # ignore deprecation warnings from 3rd party libraries, if we aren't calling them
    # directly.
    # ie, if we call foo, which calls bar, and bar emits a deprecation warning, we
    # should ignore that since it's really foo's problem. If we call bar directly
    # we want to see the warning since that's on us to fix.
    cat = category or (
        message.__class__ if isinstance(message, Warning) else UserWarning
    )

    if issubclass(cat, DeprecationWarning):
        # get the name of all the modules in the stack at the time of the warning
        stack = inspect.stack()
        modules_in_stack = [
            frame.frame.f_globals.get("__name__", "") for frame in stack
        ]

        # Filter out warnings, conftest, and python's internal stuff
        call_chain = [
            m
            for m in modules_in_stack
            if m
            and not m.startswith("warnings")
            and not m.startswith("importlib")
            and not m.startswith("_pytest")
            and not m.startswith("pluggy")
            and "conftest" not in m
        ]
        # get just the package names, don't care about specific modules
        call_chain = [c.split(".")[0] for c in call_chain]

        if call_chain:
            emitter = call_chain[0]
            for caller in call_chain[1:]:
                if caller == emitter:
                    # internal to emitting library, assume it's on us to fix
                    continue
                elif caller != emitter and caller != "interpax":
                    # warning is caused by intermediate party, ignore it
                    return

    # otherwise, fall back to original behavior (which pytest turns into an error)
    return _original_warn(message, category, stacklevel, source)


# Need to do this here before any other imports in order to catch import time
# deprecation warnings
warnings.warn = smart_warn
