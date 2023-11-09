# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
r"""Classes for handling discrete state trajectories.

- [**StateTraj**][msmhelper.statetraj.StateTraj] is a fast implementation
  of a state trajectory and should be used for microstate dynamics.

- [**LumpedStateTraj**][msmhelper.statetraj.LumpedStateTraj] is an
  implementation of the Hummer-Szabo projection[^1] and allows to reproduce
  the microstates dynamics on the macrostates space.

!!! note
    One should also mention that for bad coarse-graining one can get negative
    entries in the transition matrix $T_{ij} < 0$. To prevent this, one can
    explicitly force $T_{ij} \ge 0$ by setting the flag `positive=True`.

[^1]: Hummer and Szabo, **Optimal Dimensionality Reduction of
      Multistate Kinetic and Markov-State Models**, *J. Phys. Chem. B*,
      119 (29), 9029-9037 (2015),
      doi: [10.1021/jp508375q](https://doi.org/10.1021/jp508375q)

"""
import numpy as np

import msmhelper as mh


class StateTraj:  # noqa: WPS214
    """Class for handling discrete state trajectories."""
    __slots__ = ('_trajs', '_states')

    def __new__(cls, trajs):
        """Initialize new instance.

        If called with instance, return instance instead.

        """
        if isinstance(trajs, StateTraj):
            return trajs
        return super().__new__(cls)

    def __init__(self, trajs):
        """Initialize StateTraj and convert to index trajectories.

        If called with StateTraj instance, it will be returned instead.

        Parameters
        ----------
        trajs : list or ndarray or list of ndarray
            State trajectory/trajectories. The states need to be integers.

        """
        if isinstance(trajs, StateTraj):
            return

        self._trajs = mh.utils.format_state_traj(trajs)

        # get number of states
        self._states = mh.utils.unique(self._trajs)

        # enforce true copy of trajs
        if np.array_equal(self._states, np.arange(self.nstates)):
            self._trajs = [traj.copy() for traj in self._trajs]
        # shift to indices
        elif np.array_equal(self._states, np.arange(1, self.nstates + 1)):
            self._states = np.arange(1, self.nstates + 1)
            self._trajs = [traj - 1 for traj in self._trajs]
        else:  # not np.array_equal(self._states, np.arange(self.nstates)):
            self._trajs, self._states = mh.utils.rename_by_index(
                self._trajs,
                return_permutation=True,
            )

    @property
    def states(self):
        """Return active set of states.

        Returns
        -------
        states : ndarray
            Numpy array holding active set of states.

        """
        return self._states.copy()

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
        """Return cumulative length of all trajectories.

        Returns
        -------
        nframes : int
            Number of frames of all trajectories.

        """
        return np.sum([len(traj) for traj in self._trajs])

    @property
    def trajs(self):
        """Return state trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        if np.array_equal(self.states, np.arange(1, self.nstates + 1)):
            return [traj + 1 for traj in self._trajs]
        if np.array_equal(self.states, np.arange(self.nstates)):
            return self.index_trajs
        return mh.shift_data(
            self._trajs,
            np.arange(self.nstates),
            self.states,
        )

    @property
    def trajs_flatten(self):
        """Return flattened state trajectory.

        Returns
        -------
        trajs : ndarray
            1D ndarray representation of state trajectories.

        """
        return np.concatenate(self.trajs)

    @property
    def index_trajs(self):
        """Return index trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        return [traj.copy() for traj in self._trajs]

    @property
    def index_trajs_flatten(self):
        """Return flattened index trajectory.

        Returns
        -------
        trajs : ndarray
            1D ndarray representation of index trajectories.

        """
        return np.concatenate(self.index_trajs)

    def __repr__(self):
        """Return representation of class."""
        kw = {
            'classname': self.__class__.__name__,
            'trajs': self.trajs,
        }
        return ('{classname}({trajs})'.format(**kw))

    def __str__(self):
        """Return string representation of class."""
        return ('{trajs!s}'.format(trajs=self.trajs))

    def __iter__(self):
        """Iterate over trajectories."""
        return iter(self.trajs)

    def __len__(self):
        """Return length of list of trajectories."""
        return len(self._trajs)

    def __getitem__(self, key):
        """Get key value."""
        return self.trajs.__getitem__(key)  # noqa: WPS609

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
            Transition probability matrix $T_{ij}$, containing the transition
            probability transition from state $i\to j$.
        states : ndarray
            Array holding states corresponding to the columns of $T_{ij}$.

        """
        return mh.msm.msm._estimate_markov_model(
            self.index_trajs,
            lagtime,
            self.nstates,
            self.states,
        )

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
    """Class for using the Hummer-Szabo projection with state trajectories."""
    __slots__ = (
        '_trajs',
        '_states',
        '_macrostates',
        '_state_assignment',
        'positive',
    )

    def __new__(cls, macrotrajs, microtrajs=None, positive=False):
        """Initialize new instance."""
        if isinstance(macrotrajs, LumpedStateTraj):
            return macrotrajs
        return super().__new__(cls, None)

    def __init__(self, macrotrajs, microtrajs=None, positive=False):
        r"""Initialize LumpedStateTraj.

        If called with LumpedStateTraj instance, it will be returned instead.
        This class is an implementation of the Hummer-Szabo projection[^1].

        [^1]: Hummer and Szabo, **Optimal Dimensionality Reduction of
              Multistate Kinetic and Markov-State Models**, *J. Phys. Chem. B*,
              119 (29), 9029-9037 (2015),
              doi: [10.1021/jp508375q](https://doi.org/10.1021/jp508375q)

        Parameters
        ----------
        macrotrajs : list or ndarray or list of ndarray
            Lumped state trajectory/trajectories. The states need to be
            integers and all states needs to correspond to union of
            microstates.
        microtrajs : list or ndarray or list of ndarray
            State trajectory/trajectories. EaThe states should start from zero
            and need to be integers.
        positive : bool
            If `True` $T_ij\ge0$ will be enforced, else small negative values
            are possible.

        """
        if isinstance(macrotrajs, LumpedStateTraj):
            return

        if microtrajs is None:
            raise TypeError(
                'microtrajs may only be None when macrotrajs is of type ' +
                'LumpedStateTraj.',
            )

        self.positive = positive

        # parse macrotraj
        macrotrajs = mh.utils.format_state_traj(macrotrajs)
        self._macrostates = mh.utils.unique(macrotrajs)

        # init microstate trajectories
        super().__init__(microtrajs)

        # cache flattened trajectories to speed up code for many states
        macrotrajs_flatten = np.concatenate(macrotrajs)
        microtrajs_flatten = self.microstate_trajs_flatten

        self._state_assignment = np.zeros(self.nmicrostates, dtype=np.int64)
        for idx, microstate in enumerate(self.microstates):
            idx_first = mh.utils.find_first(microstate, microtrajs_flatten)
            self._state_assignment[idx] = macrotrajs_flatten[idx_first]

    @property
    def states(self):
        """Return active set of macrostates.

        Returns
        -------
        states : ndarray
            Numpy array holding active set of states.

        """
        return self._macrostates.copy()

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
        if np.array_equal(self.microstates, np.arange(1, self.nstates + 1)):
            return [traj + 1 for traj in self._trajs]
        elif np.array_equal(self.microstates, np.arange(self.nmicrostates)):
            return self.microstate_index_trajs
        return mh.shift_data(
            self._trajs,
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
    def microstate_index_trajs(self):
        """Return microstate index trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the microstate index trajectory.

        """
        return [traj.copy() for traj in self._trajs]

    @property
    def microstate_index_trajs_flatten(self):
        """Return flattened microstate index trajectory.

        Returns
        -------
        trajs : ndarray
            1D ndarrays representation of microstate index trajectories.

        """
        return np.concatenate(self.microstate_index_trajs)

    @property
    def trajs(self):
        """Return macrostate trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input macrostate data.

        """
        return mh.shift_data(
            self._trajs,
            np.arange(self.nmicrostates),
            self._state_assignment,
        )

    @property
    def index_trajs(self):
        """Return index trajectory.

        Returns
        -------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        return mh.shift_data(
            self._trajs,
            np.arange(self.nmicrostates),
            self._state_assignment_idx,
        )

    @property
    def microstates(self):
        """Return active set of microstates.

        Returns
        -------
        states : ndarray
            Numpy array holding active set of states.

        """
        return self._states.copy()

    @property
    def nmicrostates(self):
        """Return number of active set of states.

        Returns
        -------
        states : ndarray
            Numpy array holding active set of states.

        """
        return len(self.microstates)

    @property
    def state_assignment(self):
        """Return micro to macrostate assignment vector.

        Returns
        -------
        state_assignment : ndarray
            Micro to macrostate assignment vector.

        """
        return self._state_assignment.copy()

    @property
    def _state_assignment_idx(self):
        """Return micro to macrostate assignment vector.

        Returns
        -------
        state_assignment_idx : ndarray
            Micro to macrostate assignment vector.

        """
        return mh.shift_data(
            self.state_assignment,
            self.states,
            np.arange(self.nstates),
        )

    def __repr__(self):
        """Return representation of class."""
        kw = {
            'classname': self.__class__.__name__,
            'trajs': self.trajs,
            'microtrajs': self.microstate_trajs,
        }
        return ('{classname}({trajs}, {microtrajs})'.format(**kw))

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
            np.array_equal(self.state_assignment, other.state_assignment) and
            self.positive == other.positive
        )

    def estimate_markov_model(self, lagtime):
        r"""Estimates Markov State Model.

        This method estimates the microstate MSM based on the transition count
        matrix, followed by Szabo-Hummer projection[^1] formalism to
        macrostates.

        [^1]: Hummer and Szabo, **Optimal Dimensionality Reduction of
              Multistate Kinetic and Markov-State Models**, *J. Phys. Chem. B*,
              119 (29), 9029-9037 (2015),
              doi: [10.1021/jp508375q](https://doi.org/10.1021/jp508375q)

        Parameters
        ----------
        lagtime : int
            Lag time for estimating the markov model given in [frames].

        Returns
        -------
        T : ndarray
            Transition probability matrix $T_{ij}$, containing the transition
            probability transition from state $i\to j$.
        states : ndarray
            Array holding states corresponding to the columns of $T_{ij}$.

        """
        # in the following corresponds 'i' to micro and 'a' to macro
        msm_i, _ = mh.msm.msm._estimate_markov_model(
            self.microstate_index_trajs,
            lagtime,
            self.nmicrostates,
            self.microstates,
        )
        if not mh.utils.tests.is_ergodic(msm_i):
            raise TypeError('tmat needs to be ergodic transition matrix.')
        return (self._estimate_markov_model(msm_i), self.states)

    def _estimate_markov_model(self, msm_i):
        """Estimates Markov State Model."""
        ones_i = np.ones_like(self.microstates)
        ones_a = np.ones_like(self.states)
        id_i = np.diag(ones_i)
        id_a = np.diag(ones_a)

        peq_i = mh.msm.peq(msm_i)
        peq_a = np.array([
            np.sum(peq_i[self.state_assignment == state])
            for state in self.states
        ])
        d_i = np.diag(peq_i)
        d_a = np.diag(peq_a)
        aggret = np.zeros((self.nmicrostates, self.nstates))
        aggret[(np.arange(self.nmicrostates), self._state_assignment_idx)] = 1

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

        # enforce T_ij >= 0
        if self.positive:
            msm_a[msm_a < 0] = 0

        return mh.msm.row_normalize_matrix(msm_a)
