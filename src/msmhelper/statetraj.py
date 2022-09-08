# -*- coding: utf-8 -*-
"""Class for handling discrete state trajectories.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import numpy as np

import msmhelper as mh


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
            State trajectory/trajectories. The states need to be integers.

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
        return mh.shift_data(
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
        self._trajs = mh.tools.format_state_traj(trajs)

        # get number of states
        self._states = mh.tools.unique(self._trajs)

        # shift to indices
        if not np.array_equal(self._states, np.arange(self.nstates)):
            self._trajs, self._states = mh.tools.rename_by_index(
                self._trajs,
                return_permutation=True,
            )

        # set number of frames
        self._nframes = np.sum([len(traj) for traj in self.trajs])

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

    def estimate_markov_model(self, lagtime):
        """Estimates Markov State Model.

        This method estimates the MSM based on the transition count matrix.

        Parameters
        ----------
        lagtime : int
            Lag time for estimating the markov model given in [frames].

        Returns
        -------
        T : ndarray
            Transition rate matrix.

        permutation : ndarray
            Array with corresponding states.

        """
        return mh.estimate_markov_model(self, lagtime)

    def state_to_idx(self, state):
        """Get idx corresponding to state.

        Parameters
        ----------
        state : int
            State to get idx of.

        Returns
        -------
        idx : int
            Idx corresponding to state.

        """
        idx = np.where(self.states == state)[0]
        if not idx.size:
            raise ValueError(
                'State "{state}" does not exists in trajectory.'.format(
                    state=state,
                ),
            )
        return idx[0]


class LumpedStateTraj(StateTraj):
    """Class for handling lumped discrete state trajectories."""

    def __new__(cls, macrotrajs, microtrajs=None):
        """Initialize new instance."""
        if isinstance(macrotrajs, LumpedStateTraj):
            return macrotrajs
        return super().__new__(cls, None)

    def __init__(self, macrotrajs, microtrajs=None):
        """Initialize LumpedStateTraj and convert to index trajectories.

        If called with LumpedStateTraj instance, it will be retuned instead.

        Parameters
        ----------
        macrotrajs : list or ndarray or list of ndarray
            Lumped state trajectory/trajectories. The states need to be
            integers and all states needs to correspond to union of
            microstates.

        microtrajs : list or ndarray or list of ndarray
            State trajectory/trajectories. EaThe states should start from zero
            and need to be integers.

        """
        if isinstance(macrotrajs, LumpedStateTraj):
            return

        if microtrajs is None:
            raise TypeError(
                'microtrajs may only be None when macrotrajs is of type ' +
                'LumpedStateTraj.',
            )
        # initialize base class
        self._parse_microtrajs(microtrajs)

        # parse the microstate to macrostate lumping
        self._parse_macrotrajs(macrotrajs)

    @property
    def states(self):
        """Return active set of macrostates.

        Returns
        -------
        states : ndarray
            Numpy array holding active set of states.

        """
        return self._macrostates

    @property
    def nstates(self):
        """Return number of macrostates.

        Returns
        -------
        nstates : int
            Number of states.

        """
        return len(self.states)

    @property
    def microstate_trajs(self):
        """Return microstate trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        if np.array_equal(self.microstates, np.arange(self.nmicrostates)):
            return self.trajs
        return mh.shift_data(
            self.trajs,
            np.arange(self.nmicrostates),
            self.microstates,
        )

    @property
    def microstate_trajs_flatten(self):
        """Return flattened state trajectory.

        Returns
        -------
        trajs : ndarray
            1D ndarrays representation of state trajectories.

        """
        return np.concatenate(self.microstate_trajs)

    @property
    def state_trajs(self):
        """Return macrostate trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input macrostate data.

        """
        return mh.shift_data(
            self.trajs,
            np.arange(self.nmicrostates),
            self.state_assignment,
        )

    @property
    def state_trajs_flatten(self):
        """Return flattened macrostate trajectory.

        Returns
        -------
        trajs : ndarray
            1D ndarrays representation of macrostate trajectories.

        """
        return np.concatenate(self.state_trajs)

    @property
    def microstates(self):
        """Return active set of states.

        Returns
        -------
        states : ndarray
            Numpy array holding active set of states.

        """
        return self._states

    @property
    def nmicrostates(self):
        """Return number of states.

        Returns
        -------
        nstates : int
            Number of states.

        """
        return len(self.microstates)

    def __repr__(self):
        """Return representation of class."""
        kw = {
            'clname': self.__class__.__name__,
            'trajs': self.state_trajs,
            'microtrajs': self.microstate_trajs,
        }
        return ('{clname}({trajs}, {microtrajs})'.format(**kw))

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
        if not isinstance(other, LumpedStateTraj):
            return NotImplemented
        return (
            self.ntrajs == other.ntrajs and
            all(
                np.array_equal(self.trajs[idx], other.trajs[idx])
                for idx in range(self.ntrajs)
            ) and
            np.array_equal(self.state_assignment, other.state_assignment)
        )

    def estimate_markov_model(self, lagtime):
        """Estimates Markov State Model.

        This method estimates the microstate MSM based on the transition count
        matrix, followed by Szabo-Hummer projection formalism to macrostates.

        Parameters
        ----------
        lagtime : int
            Lag time for estimating the markov model given in [frames].

        Returns
        -------
        T : ndarray
            Transition rate matrix.

        permutation : ndarray
            Array with corresponding states.

        """
        # in the following corresponds 'i' to micro and 'a' to macro
        msm_i, _ = mh.estimate_markov_model(self, lagtime)
        if not mh.tests.is_ergodic(msm_i):
            raise TypeError('tmat needs to be ergodic transition matrix.')
        return (self._estimate_markov_model(msm_i), self.states)

    def _estimate_markov_model(self, msm_i):
        """Estimates Markov State Model."""
        ones_i = np.ones_like(self.microstates)
        ones_a = np.ones_like(self.states)
        id_i = np.diag(ones_i)
        id_a = np.diag(ones_a)

        peq_i = mh.peq(msm_i)
        peq_a = np.array([
            np.sum(peq_i[self.state_assignment == state])
            for state in self.states
        ])
        d_i = np.diag(peq_i)
        d_a = np.diag(peq_a)
        aggret = np.zeros((self.nmicrostates, self.nstates))
        aggret[(np.arange(self.nmicrostates), self.state_assignment_idx)] = 1

        m_prime = np.linalg.inv(
            id_i +
            ones_i[:, np.newaxis] * peq_i[np.newaxis:, ] -
            msm_i,
        )
        m_twoprime = np.linalg.inv(
            np.linalg.multi_dot((aggret.T, d_i, m_prime, aggret)),
        )

        msm_a = (
            id_a +
            ones_a[:, np.newaxis] * peq_a[np.newaxis:, ] -
            m_twoprime @ d_a
        )
        return mh.row_normalize_matrix(msm_a)

    def _parse_macrotrajs(self, macrotrajs):
        """Parse the macrotrajs."""
        # TODO: improve performance by not using StateTraj
        macrotrajs = StateTraj(macrotrajs)
        self._macrostates = macrotrajs.states.copy()

        # cache flattened trajectories to speed up code for many states
        macrotrajs_flatten = macrotrajs.state_trajs_flatten
        microtrajs_flatten = self.microstate_trajs_flatten

        self.state_assignment = np.zeros(self.nmicrostates, dtype=np.int64)
        for idx, microstate in enumerate(self.microstates):
            idx_first = mh.tools.find_first(microstate, microtrajs_flatten)
            self.state_assignment[idx] = macrotrajs_flatten[idx_first]

        self.state_assignment_idx = mh.shift_data(
            self.state_assignment,
            self.states,
            np.arange(self.nstates),
        )

    def _parse_microtrajs(self, trajs):
        """Parse the microtrajs."""
        self._trajs = mh.tools.format_state_traj(trajs)

        # get number of states
        self._states = mh.tools.unique(self._trajs)

        # shift to indices
        if not np.array_equal(self._states, np.arange(self.nmicrostates)):
            self._trajs, self._states = mh.tools.rename_by_index(
                self._trajs,
                return_permutation=True,
            )

        # set number of frames
        self._nframes = np.sum([len(traj) for traj in self.trajs])
