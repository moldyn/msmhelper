# -*- coding: utf-8 -*-
"""Tests for the decorators module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import warnings

from msmhelper import decorators


def test_deprecated():
    """Test deprecated warning."""
    # define function
    kwargs = {'msg': 'msg', 'since': '1.0.0', 'remove': '1.2.0'}

    @decorators.deprecated(**kwargs)
    def func():
        return True

    warning_msg = (
        'Calling deprecated function func. {msg}'.format(**kwargs) +
        ' -- Deprecated since version {since}'.format(**kwargs) +
        ' -- Function will be removed starting from {remove}'.format(**kwargs)
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            assert func()
        except DeprecationWarning as dw:
            assert str(dw) == warning_msg
        else:
            raise AssertionError()


def test_shortcut():
    """Test shortcut decorator."""
    # test for function
    name = 'f'

    @decorators.shortcut(name)
    def func():
        pass

    try:
        f()
    except NameError:
        raise AssertionError()

    assert f.__doc__ != func.__doc__  # noqa: F821
    assert f.__name__ == name  # noqa: F821


def test_debug():
    """Test debug decorator."""
    # test for function
    def func():
        """Test docstring."""
        return True

    func_dec = decorators.debug(func)

    assert func() == func_dec()
    assert func.__doc__ == func_dec.__doc__  # noqa: F821
    assert func.__name__ == func_dec.__name__  # noqa: F821
