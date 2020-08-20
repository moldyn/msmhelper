# -*- coding: utf-8 -*-
"""Decorators.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import functools
import warnings


# taken from https://gitlab.com/braniii/prettypyplot
def deprecated(msg=None, since=None, remove=None):
    """Add deprecated warning.

    Parameters
    ----------
    msg : str
        Message of deprecated warning.

    since : str
        Version since deprecated, e.g. '1.0.2'

    remove : str
        Version this function will be removed, e.g. '0.14.2'

    Returns
    -------
    f : function
        Return decorated function.

    Examples
    --------
    >>> @deprecated(msg='Use lag_time instead.', remove='1.0.0')
    >>> def lagtime(args):
    ...     pass  # function goes here
    # If function is called, you will get warning
    >>> lagtime(...)
    Calling deprecated function lagtime. Use lag_time instead.
    -- Function will be removed starting from 1.0.0

    """
    def deprecated_msg(func, msg, since, remove):
        warn_msg = 'Calling deprecated function {0}.'.format(func.__name__)
        if msg:
            warn_msg += ' {0}'.format(msg)
        if since:
            warn_msg += ' -- Deprecated since version {v}'.format(v=since)
        if remove:
            warn_msg += (
                ' -- Function will be removed starting from ' +
                '{v}'.format(v=remove)
            )
        return warn_msg

    def decorator_deprecated(func):
        @functools.wraps(func)
        def wrapper_deprecated(*args, **kwargs):
            warnings.warn(
                deprecated_msg(func, msg, since, remove),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)  # pragma: no cover

        return wrapper_deprecated

    return decorator_deprecated


def shortcut(name):
    """Add alternative identity to function.

    This decorator supports only functions and no class members!

    Parameters
    ----------
    name : str
        Alternative function name.

    Returns
    -------
    f : function
        Return decorated function.

    Examples
    --------
    >>> @shortcut('tau')
    >>> def lagtime(args):
    ...     pass  # function goes here
    # Function can now be called via shortcut.
    >>> tau(...)  # noqa

    """
    def decorator_shortcut(func):
        # register function
        func.__globals__[name] = func  # noqa: WPS609
        return func

    return decorator_shortcut
