# -*- coding: utf-8 -*-
"""Class for handling discrete state trajectories.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from msmhelper import tools


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class StateTraj:  # noqa: WPS214
    """Class for handling discrete state trajectories."""

    # noqa: E800 # add slots magic,  self.__slots__ = ('_states', '_trajs').
    def __new__(cls, trajs):
        """Initialize new instance.

        If called with instance, return instance instead.

        """
        if isinstance(trajs, StateTraj):
            return trajs
        return super().__new__(cls)

    def __init__(self, trajs):
        """Initialize StateTraj and convert to index trajectories.

        If called with StateTraj instance, it will be retuned instead.

        Parameters
        ----------
        trajs : list or ndarray or list of ndarray
            State trajectory/trajectories. The states should start from zero
            and need to be integers.

        """
        if isinstance(trajs, StateTraj):
            return

        self.state_trajs = trajs

    @property
    def states(self):
        """Return active set of states.

        Returns
        -------
        states : ndarray
            Numpy array holding active set of states.

        """
        return self._states

    @property
    def nstates(self):
        """Return number of states.

        Returns
        -------
        nstates : int
            Number of states.

        """
        return len(self.states)

    @property
    def ntrajs(self):
        """Return number of trajectories.

        Returns
        -------
        ntrajs : int
            Number of trajectories.

        """
        return len(self)

    @property
    def nframes(self):
        """Return cummulated length of all trajectories.

        Returns
        -------
        nframes : int
            Number of frames of all trajectories.

        """
        return self._nframes

    @property
    def state_trajs(self):
        """Return state trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        if np.array_equal(self.states, np.arange(self.nstates)):
            return self.trajs
        return tools.shift_data(
            self.trajs,
            np.arange(self.nstates),
            self.states,
        )

    @state_trajs.setter
    def state_trajs(self, trajs):
        """Set the state trajectory.

        Parameters
        ----------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        self._trajs = tools.format_state_traj(trajs)

        # get number of states
        self._states = tools.unique(self._trajs)

        # shift to indices
        if not np.array_equal(self._states, np.arange(self.nstates)):
            self._trajs, self._states = tools.rename_by_index(  # noqa: WPS414
                self._trajs,
                return_permutation=True,
            )

        # set number of frames
        self._nframes = np.sum((len(traj) for traj in self.trajs))

    @property
    def state_trajs_flatten(self):
        """Return flattened state trajectory.

        Returns
        -------
        trajs : ndarray
            1D ndarrays representation of state trajectories.

        """
        return np.concatenate(self.state_trajs)

    @property
    def index_trajs(self):
        """Return index trajectory.

        Same as `self.trajs`

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        return self.trajs

    @property
    def trajs(self):
        """Return index trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        return self._trajs

    @property
    def trajs_flatten(self):
        """Return flattened index trajectory.

        Returns
        -------
        trajs : ndarray
            1D ndarrays representation of index trajectories.

        """
        return np.concatenate(self._trajs)

    def __repr__(self):
        """Return representation of class."""
        kw = {
            'clname': self.__class__.__name__,
            'trajs': self.state_trajs,
        }
        return ('{clname}({trajs})'.format(**kw))

    def __str__(self):
        """Return string representation of class."""
        return ('{trajs!s}'.format(trajs=self.state_trajs))

    def __iter__(self):
        """Iterate over trajectories."""
        return iter(self.state_trajs)

    def __len__(self):
        """Return length of list of trajectories."""
        return len(self.trajs)

    def __getitem__(self, key):
        """Get key value."""
        return self.state_trajs.__getitem__(key)  # noqa: WPS609

    def __eq__(self, other):
        """Compare two objects."""
        if not isinstance(other, StateTraj):
            return NotImplemented
        return (
            self.ntrajs == other.ntrajs and
            all(
                np.array_equal(self[idx], other[idx])
                for idx in range(self.ntrajs)
            )
        )
