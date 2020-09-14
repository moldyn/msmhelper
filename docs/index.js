URLS=[
"index.html",
"msm.html",
"benchmark.html",
"iotext.html",
"compare.html",
"tests.html",
"decorators.html",
"statetraj.html",
"tools.html"
];
INDEX=[
{
"ref":"msmhelper",
"url":0,
"doc":"![GitHub Workflow Status](https: img.shields.io/github/workflow/status/moldyn/msmhelper/Python%20package) ![GitHub All Releases](https: img.shields.io/github/downloads/moldyn/msmhelper/total) ![GitHub last commit](https: img.shields.io/github/last-commit/moldyn/msmhelper) ![GitHub release (latest by date)](https: img.shields.io/github/v/release/moldyn/msmhelper) ![LGTM Grade](https: img.shields.io/lgtm/grade/python/github/moldyn/msmhelper?label=code%20quality&logo=lgtm) ![wemake-python-styleguide](https: img.shields.io/badge/style-wemake-000000.svg)  msmhelper This is a package with helper functions to work with state trajectories. Hence, it is mainly used for Markov State Models.  Usage This package is mainly based on numpy and numba.  Usage   import msmhelper  .    Known Bugs - not known  Requirements: - Python 3.6+ - Numba 0.49.0+ - Numpy 1.16.2+ - Pyemma 2.5.7+  Changelog: - tba: - v0.4: - Add  compare module to compare two different state discretizations - Upgrade pydoc to  0.9.1 with search option. - v0.3: - Add  benchmark module with an numba optimized version of the Chapman Kolmogorov test. - v0.2: - parts of msm module are rewritten in numba - v0.1: - initial release  Roadmap: - write roadmap  Development  Additional Requirements: - wemake-python-styleguide - flake8-spellcheck  Pytest Running pytest with numba needs an additional flag   export NUMBA_DISABLE_JIT=1  pytest    Credits: - [numpy](https: docs.scipy.org/doc/numpy) - [realpython](https: realpython.com/)"
},
{
"ref":"msmhelper.msm",
"url":1,
"doc":"Create Markov State Model. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved. Authors: Daniel Nagel Georg Diez"
},
{
"ref":"msmhelper.msm.build_MSM",
"url":1,
"doc":"Wrapps pyemma.msm.estimate_markov_model. Based on the choice of reversibility it either calls pyemma for a reversible matrix or it creates a transition count matrix. Parameters      trajs : list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtime : int Lag time for estimating the markov model given in [frames]. reversible : bool, optional If  True it will uses pyemma.msm.estimate_markov_model which does not guarantee that the matrix is of full dimension. In case of  False or if not statedm the local function based on a simple transitition count matrix will be used instead. kwargs For passing values to  pyemma.msm.estimate_markov_model . Returns    - transmat : ndarray Transition rate matrix.",
"func":1
},
{
"ref":"msmhelper.msm.estimate_markov_model",
"url":1,
"doc":"Estimates Markov State Model. This method estimates the MSM based on the transition count matrix. Parameters      trajs : statetraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtime : int Lag time for estimating the markov model given in [frames]. Returns    - T : ndarray Transition rate matrix. permutation : ndarray Array with corresponding states.",
"func":1
},
{
"ref":"msmhelper.msm.left_eigenvectors",
"url":1,
"doc":"Estimate left eigenvectors. Estimates the left eigenvectors and corresponding eigenvalues of a quadratic matrix. Parameters      matrix : ndarray Quadratic 2d matrix eigenvectors and eigenvalues or determined of. Returns    - eigenvalues : ndarray N eigenvalues sorted by their value (descending). eigenvectors : ndarray N eigenvectors sorted by descending eigenvalues.",
"func":1
},
{
"ref":"msmhelper.msm.implied_timescales",
"url":1,
"doc":"Calculate the implied timescales. Calculate the implied timescales for the given values.  todo catch if for higher lagtimes the dimensionality changes Parameters      trajs : StateTraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtimes : list or ndarray int Lagtimes for estimating the markov model given in [frames]. reversible : bool If reversibility should be enforced for the markov state model. Returns    - T : ndarray Transition rate matrix.",
"func":1
},
{
"ref":"msmhelper.benchmark",
"url":2,
"doc":"Benchmark Markov State Model. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved. Authors: Daniel Nagel"
},
{
"ref":"msmhelper.benchmark.ck_test",
"url":2,
"doc":"Calculate the Chapman Kolmogorov equation. Parameters      trajs : StateTraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtimes : list or ndarray int Lagtimes for estimating the markov model given in [frames]. tmax : int Longest time to evaluate the CK equation given in [frames]. Returns    - cktest : dict Dictionary holding for each lagtime the ckequation and with 'md' the reference.",
"func":1
},
{
"ref":"msmhelper.benchmark.chapman_kolmogorov_test",
"url":2,
"doc":"Calculate the Chapman Kolmogorov equation. Parameters      trajs : StateTraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtimes : list or ndarray int Lagtimes for estimating the markov model given in [frames]. tmax : int Longest time to evaluate the CK equation given in [frames]. Returns    - cktest : dict Dictionary holding for each lagtime the ckequation and with 'md' the reference.",
"func":1
},
{
"ref":"msmhelper.iotext",
"url":3,
"doc":"Input and output text files. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.iotext.FileError",
"url":3,
"doc":"An exception for wrongly formated input files."
},
{
"ref":"msmhelper.iotext.opentxt",
"url":3,
"doc":"Open a text file. This method can load an nxm array of floats from an ascii file. It uses either pandas read_csv for a single comment or as fallback the slower numpy laodtxt for multiple comments.  warning In contrast to pandas the order of usecols will be used. So if using \u00b4data = opentxt( ., uscols=[1, 0])\u00b4 you acces the first column by  data[:, 0] and the second one by  data[:, 1] . Parameters      file_name : string Name of file to be opened. comment : str or array of str, optional Characters with which a comment starts. nrows : int, optional The maximum number of lines to be read usecols : int-array, optional Columns to be read from the file (zero indexed). skiprows : int, optional The number of leading rows which will be skipped. dtype : data-type, optional Data-type of the resulting array. Default: float. Returns    - array : ndarray Data read from the text file.",
"func":1
},
{
"ref":"msmhelper.iotext.savetxt",
"url":3,
"doc":"Save nxm array of floats to a text file. It uses numpys savetxt method and extends the header with information of execution. Parameters      file_name : string File name to store data. array : ndarray Data to be stored. header : str, optional Comment written into the header of the output file. fmt : str or sequence of strs, optional See numpy.savetxt fmt.",
"func":1
},
{
"ref":"msmhelper.iotext.opentxt_limits",
"url":3,
"doc":"Load file and split according to limit file. If limits_file is not provided it will return [traj]. Parameters      file_name : string Name of file to be opened. limits_file : str, optional File name of limit file. Should be single column ascii file. kwargs The Parameters defined in opentxt. Returns    - traj : ndarray Return array of subtrajectories.",
"func":1
},
{
"ref":"msmhelper.iotext.openmicrostates",
"url":3,
"doc":"Load 1d file and split according to limit file. Both, the limit file and the trajectory file needs to be a single column file. If limits_file is not provided it will return [traj]. The trajectory will of dtype np.int16, so the states needs to be smaller than 32767. Parameters      file_name : string Name of file to be opened. limits_file : str, optional File name of limit file. Should be single column ascii file. kwargs The Parameters defined in opentxt. Returns    - traj : ndarray Return array of subtrajectories.",
"func":1
},
{
"ref":"msmhelper.iotext.open_limits",
"url":3,
"doc":"Load and check limit file. The limits give the length of each single trajectory. So e.g. [5, 5, 5] for 3 equally-sized subtrajectories of length 5. Parameters      data_length : int Length of data read. limits_file : str, optional File name of limit file. Should be single column ascii file. Returns    - limits : ndarray Return cumsum of limits.",
"func":1
},
{
"ref":"msmhelper.compare",
"url":4,
"doc":"Set of helpful functions for comparing markov state models. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.compare.compare_discretization",
"url":4,
"doc":"Compare similarity of two state discretizations. This method compares the similarity of two state discretizations of the same dataset. There are two different methods, 'directed' gives a measure on how high is the probable to assign a frame correclty knowing the  traj1 . Hence splitting a state into many is not penalized, while merging multiple into a single state is. Selecting 'symmetric' it is check in both directions, so it checks for each state if it is possible to assigned it forward or backward. Hence, splitting and merging states is not penalized. Parameters      traj1 : StateTraj like First state discretization. traj2 : StateTraj like Second state discretization. method : ['symmetric', 'directed'] Selecting similarity norm. 'symmetric' compares if each frame is forward or backward assignable, while 'directed' checks only if it is forard assignable. Returns    - similarity : float Similarity going from [0, 1], where 1 means identical and 0 no similarity at all.",
"func":1
},
{
"ref":"msmhelper.tests",
"url":5,
"doc":"Set of helpful test functions. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.tests.is_quadratic",
"url":5,
"doc":"Check if matrix is quadratic. Parameters      matrix : ndarray, list of lists Matrix which is checked if is 2d array. Returns    - is_quadratic : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_state_traj",
"url":5,
"doc":"Check if state trajectory is correct formatted. Parameters      trajs : list of ndarray State trajectory/trajectories need to be lists of ndarrays of integers. Returns    - is_state_traj : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_index_traj",
"url":5,
"doc":"Check if states can be used as indices. Parameters      trajs : list of ndarray State trajectory/trajectories need to be lists of ndarrays of integers. Returns    - is_index : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_tmat",
"url":5,
"doc":"Check if transition matrix. Rows and cols of zeros (non-visited states) are accepted. Parameters      matrix : ndarray Transition matrix. Returns    - is_tmat : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_transition_matrix",
"url":5,
"doc":"Check if transition matrix. Rows and cols of zeros (non-visited states) are accepted. Parameters      matrix : ndarray Transition matrix. Returns    - is_tmat : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_ergodic",
"url":5,
"doc":"Check if matrix is ergodic. Taken from: Wielandt, H. \"Unzerlegbare, Nicht Negativen Matrizen.\" Mathematische Zeitschrift. Vol. 52, 1950, pp. 642\u2013648. Parameters      matrix : ndarray Transition matrix. Returns    - is_tmat : bool",
"func":1
},
{
"ref":"msmhelper.decorators",
"url":6,
"doc":"Decorators. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.decorators.deprecated",
"url":6,
"doc":"Add deprecated warning. Parameters      msg : str Message of deprecated warning. since : str Version since deprecated, e.g. '1.0.2' remove : str Version this function will be removed, e.g. '0.14.2' Returns    - f : function Return decorated function. Examples     >>> @deprecated(msg='Use lag_time instead.', remove='1.0.0') >>> def lagtime(args):  . pass  function goes here  If function is called, you will get warning >>> lagtime( .) Calling deprecated function lagtime. Use lag_time instead.  Function will be removed starting from 1.0.0",
"func":1
},
{
"ref":"msmhelper.decorators.shortcut",
"url":6,
"doc":"Add alternative identity to function. This decorator supports only functions and no class members! Parameters      name : str Alternative function name. Returns    - f : function Return decorated function. Examples     >>> @shortcut('tau') >>> def lagtime(args):  . pass  function goes here  Function can now be called via shortcut. >>> tau( .)  noqa",
"func":1
},
{
"ref":"msmhelper.statetraj",
"url":7,
"doc":"Class for handling discrete state trajectories. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.statetraj.StateTraj",
"url":7,
"doc":"Class for handling discrete state trajectories. Initialize StateTraj and convert to index trajectories. If called with StateTraj instance, it will be retuned instead. Parameters      trajs : list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers."
},
{
"ref":"msmhelper.statetraj.StateTraj.states",
"url":7,
"doc":"Return active set of states. Returns    - states : ndarray Numpy array holding active set of states."
},
{
"ref":"msmhelper.statetraj.StateTraj.nstates",
"url":7,
"doc":"Return number of states. Returns    - nstates : int Number of states."
},
{
"ref":"msmhelper.statetraj.StateTraj.ntrajs",
"url":7,
"doc":"Return number of trajectories. Returns    - ntrajs : int Number of trajectories."
},
{
"ref":"msmhelper.statetraj.StateTraj.nframes",
"url":7,
"doc":"Return cummulated length of all trajectories. Returns    - nframes : int Number of frames of all trajectories."
},
{
"ref":"msmhelper.statetraj.StateTraj.state_trajs",
"url":7,
"doc":"Return state trajectory. Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.StateTraj.state_trajs_flatten",
"url":7,
"doc":"Return flattened state trajectory. Returns    - trajs : ndarray 1D ndarrays representation of state trajectories."
},
{
"ref":"msmhelper.statetraj.StateTraj.index_trajs",
"url":7,
"doc":"Return index trajectory. Same as  self.trajs Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.StateTraj.trajs",
"url":7,
"doc":"Return index trajectory. Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.StateTraj.trajs_flatten",
"url":7,
"doc":"Return flattened index trajectory. Returns    - trajs : ndarray 1D ndarrays representation of index trajectories."
},
{
"ref":"msmhelper.tools",
"url":8,
"doc":"Set of helpful functions. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved. TODO: - Correct border effects of running mean"
},
{
"ref":"msmhelper.tools.shift_data",
"url":8,
"doc":"Shift integer array (data) from old to new values. >  CAUTION: > The values of  val_old ,  val_new and  data needs to be integers. The basic function is based on Ashwini_Chaudhary solution: https: stackoverflow.com/a/29408060 Parameters      array : StateTraj or ndarray or list or list of ndarrays 1D data or a list of data. val_old : ndarray or list Values in data which should be replaced. All values needs to be within the range of  [data.min(), data.max()] val_new : ndarray or list Values which will be used instead of old ones. dtype : data-type, optional The desired data-type. Needs to be of type unsigned integer. Returns    - array : ndarray Shifted data in same shape as input.",
"func":1
},
{
"ref":"msmhelper.tools.rename_by_population",
"url":8,
"doc":"Rename states sorted by their population starting from 1. Parameters      trajs : list or ndarray or list of ndarrays State trajectory or list of state trajectories. return_permutation : bool Return additionaly the permutation to achieve performed renaming. Default is False. Returns    - trajs : ndarray Renamed data. permutation : ndarray Permutation going from old to new state nameing. So the  i th state of the new naming corresponds to the old state  permutation[i-1] .",
"func":1
},
{
"ref":"msmhelper.tools.rename_by_index",
"url":8,
"doc":"Rename states sorted by their numerical values starting from 0. Parameters      trajs : list or ndarray or list of ndarrays State trajectory or list of state trajectories. return_permutation : bool Return additionaly the permutation to achieve performed renaming. Default is False. Returns    - trajs : ndarray Renamed data. permutation : ndarray Permutation going from old to new state nameing. So the  i th state of the new naming corresponds to the old state  permutation[i-1] .",
"func":1
},
{
"ref":"msmhelper.tools.unique",
"url":8,
"doc":"Apply numpy.unique to traj. Parameters      trajs : list or ndarray or list of ndarrays State trajectory or list of state trajectories. kwargs : Arguments of [numpy.unique()](NP_DOC.numpy.unique.html) Returns    - unique : Number of states, depending on kwargs more.",
"func":1
},
{
"ref":"msmhelper.tools.runningmean",
"url":8,
"doc":"Compute centered running average with given window size. This function returns the centered based running average of the given data. The output of this function is of the same length as the input, by assuming that the given data is zero before and after the given series. Hence, there are border affects which are not corrected. >  CAUTION: > If the given window is even (not symmetric) it will be shifted towards > the beginning of the current value. So for  window=4 , it will consider > the current position \\(i\\), the two to the left \\(i-2\\) and \\(i-1\\) and > one to the right \\(i+1\\). Function is taken from lapis: https: stackoverflow.com/questions/13728392/moving-average-or-running-mean Parameters      array : ndarray One dimensional numpy array. window : int Integer which specifies window-width. Returns    - array_rmean : ndarray Data which is time-averaged over the specified window.",
"func":1
},
{
"ref":"msmhelper.tools.swapcols",
"url":8,
"doc":"Interchange cols of an ndarray. This method swaps the specified columns.  todo Optimize memory usage Parameters      array : ndarray 2D numpy array. indicesold : integer or ndarray 1D array of indices. indicesnew : integer or ndarray 1D array of new indices Returns    - array_swapped : ndarray 2D numpy array with swappend columns.",
"func":1
},
{
"ref":"msmhelper.tools.get_runtime_user_information",
"url":8,
"doc":"Get user runtime information. >  CAUTION: > For python 3.5 or lower the date is not formatted and contains > microscends. Returns    - RUI : dict Holding username in 'user', pc name in 'pc', date of execution 'date', path of execution 'script_dir' and name of execution main file 'script_name'. In case of interactive usage, script_name is 'console'.",
"func":1
},
{
"ref":"msmhelper.tools.format_state_traj",
"url":8,
"doc":"Convert state trajectory to list of ndarrays. Parameters      trajs : list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. Returns    - trajs : list of ndarray Return list of ndarrays of integers.",
"func":1
},
{
"ref":"msmhelper.tools.matrix_power",
"url":8,
"doc":"Calculate matrix power with np.linalg.matrix_power. Same as numpy function, except only for float matrices. See [np.linalg.matrix_power](NP_DOC/numpy.linalg.matrix_power.html). Parameters      matrix : ndarray 2d matrix of type float. power : int, float Power of matrix. Returns    - matpow : ndarray Matrix power.",
"func":1
}
]