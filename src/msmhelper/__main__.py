# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""CLI of msmhelper.

Containing some command line interfaces to access basic functionality directly.

"""
import click

import msmhelper as mh
from msmhelper._cli.ck_test import ck_test
from msmhelper._cli.implied_timescales import implied_timescales

HELP_STR = f"""msmhelper v{mh.__version__}

Unlock the power of protein dynamics time series with Markov state modeling, by
simplifying scientific analysis.

Copyright (c) 2019-2023, Daniel Nagel
"""


@click.group(help=HELP_STR)
def main():
    """Empty group to show on help available submodules."""
    pass


main.add_command(ck_test)
main.add_command(implied_timescales)


if __name__ == '__main__':
    main()