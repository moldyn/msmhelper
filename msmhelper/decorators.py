# -*- coding: utf-8 -*-
"""Decorators.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import functools
import types
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
    def copy_func(func, name):
        """Return deep copy of a function."""
        func_copy = types.FunctionType(
            func.__code__,  # noqa: WPS609
            func.__globals__,  # noqa: WPS609
            name=name,
            argdefs=func.__defaults__,  # noqa: WPS609
            closure=func.__closure__,  # noqa: WPS609
        )
        # Copy attributes of function
        func_copy.__kwdefaults__ = func.__kwdefaults__  # noqa: WPS609
        return func_copy

    def decorated_doc(func):
        return (
            'This function is the shortcut of `{0}`.'.format(func.__name__) +
            'See its docstring for further help.'
        )

    def decorator_shortcut(func):
        # register function
        func_copy = copy_func(func, name=name)
        func_copy.__globals__[name] = func_copy  # noqa: WPS609
        func_copy.__doc__ = decorated_doc(func)  # noqa: WPS609, WPS125
        return func

    return decorator_shortcut


def debug(func):
    """Print each call with arguments."""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = ['{0!r}'.format(arg) for arg in args]
        kwargs_repr = [
            '{0}={1!r}'.format(key, itm) for key, itm in kwargs.items()
        ]
        print('Calling {0}({1})'.format(  # noqa: WPS421,T001
            func.__name__, ', '.join(args_repr + kwargs_repr),
        ))

        return_val = func(*args, **kwargs)

        print('{0!r} => {1!r}'.format(  # noqa: WPS421,T001
            func.__name__, return_val,
        ))
        return return_val
    return wrapper_debug
