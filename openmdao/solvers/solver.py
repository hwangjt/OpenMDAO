"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""

from __future__ import division, print_function

import os

import numpy as np


from openmdao.core.analysis_error import AnalysisError
from openmdao.jacobians.assembled_jacobian import AssembledJacobian
from openmdao.recorders.recording_iteration_stack import Recording, recording_iteration
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.record_util import create_local_meta, check_path


class SolverInfo(object):
    """
    Communal object for storing some formatting for solver iprint.

    Attributes
    ----------
    prefix : str
        Prefix to prepend during this iprint.
    stack : List
        List of strings; strings are popped and appended as needed.
    """

    def __init__(self):
        """
        Initialize.
        """
        self.prefix = ""
        self.stack = []

    def clear(self):
        """
        Clear out the iprint stack, in case something is left over from a handled exception.
        """
        self.prefix = ""
        self.stack = []

    def pop(self):
        """
        Remove one level of solver depth in the printing.
        """
        last_string = self.stack.pop()
        nchar = len(last_string)
        self.prefix = self.prefix[:-nchar]

    def append_solver(self):
        """
        Add a new level for the main solver in a group.
        """
        new_str = '+  '
        self.prefix += new_str
        self.stack.append(new_str)

    def append_subsolver(self):
        """
        Add a new level for any sub-solver for your solver.
        """
        new_str = '|  '
        self.prefix += new_str
        self.stack.append(new_str)

    def append_precon(self):
        """
        Add a new level for any preconditioner to a linear solver.
        """
        new_str = '| precon:'
        self.prefix += new_str
        self.stack.append(new_str)

    def save_cache(self):
        """
        Save prefix and stack so that they can be restored later in event of an exception recovery.

        Returns
        -------
        tuple(str, list)
            cache of current stack
        """
        return (self.prefix, self.stack)

    def restore_cache(self, cache):
        """
        Restore previously saved iprint stack names.

        Parameters
        ----------
        cache : tuple(str, list)
            cache of current stack
        """
        self.prefix, self.stack = cache


class Solver(object):
    """
    Base solver class.

    This class is subclassed by NonlinearSolver and LinearSolver,
    which are in turn subclassed by actual solver implementations.

    Attributes
    ----------
    _system : <System>
        Pointer to the owning system.
    _depth : int
        How many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        List of right-hand-side (RHS) vector names.
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _iter_count : int
        Number of iterations for the current invocation of the solver.
    _rec_mgr : <RecordingManager>
        object that manages all recorders added to this solver
    _solver_info : <SolverInfo>
        Object to store some formatting for iprint that is shared across all solvers.
    options : <OptionsDictionary>
        Options dictionary.
    recording_options : <OptionsDictionary>
        Recording options dictionary.
    metadata : dict
        Dictionary holding data about this solver.
    supports : <OptionsDictionary>
        Options dictionary describing what features are supported by this
        solver.
    _filtered_vars_to_record: Dict
        Dict of list of var names to record
    _norm0: float
        Normalization factor
    """

    SOLVER = 'base_solver'
    _solver_info = SolverInfo()

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        self._system = None
        self._depth = 0
        self._vec_names = None
        self._mode = 'fwd'
        self._iter_count = 0

        self.options = OptionsDictionary()
        self.recording_options = OptionsDictionary()
        self.options.declare('maxiter', types=int, default=10,
                             desc='maximum number of iterations')
        self.options.declare('atol', default=1e-10,
                             desc='absolute error tolerance')
        self.options.declare('rtol', default=1e-10,
                             desc='relative error tolerance')
        self.options.declare('iprint', types=int, default=1,
                             desc='whether to print output')
        self.options.declare('err_on_maxiter', types=bool, default=False,
                             desc="When True, AnlysisError will be raised if we don't convege.")
        # Case recording options
        self.recording_options.declare('record_abs_error', types=bool, default=True,
                                       desc='Set to True to record absolute error at the \
                                       solver level')
        self.recording_options.declare('record_rel_error', types=bool, default=True,
                                       desc='Set to True to record relative error at the \
                                       solver level')
        self.recording_options.declare('record_solver_residuals', types=bool, default=False,
                                       desc='Set to True to record residuals at the solver level')
        self.recording_options.declare('record_metadata', types=bool, desc='Record metadata',
                                       default=True)
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                       '(processed post-includes)')
        # Case recording related
        self._filtered_vars_to_record = {}
        self._norm0 = 0.0

        # What the solver supports.
        self.supports = OptionsDictionary()
        self.supports.declare('gradients', types=bool, default=False)

        self._declare_options()
        self.options.update(kwargs)

        self.metadata = {}
        self._rec_mgr = RecordingManager()

    def add_recorder(self, recorder):
        """
        Add a recorder to the driver's RecordingManager.

        Parameters
        ----------
        recorder : <BaseRecorder>
           A recorder instance to be added to RecManager.
        """
        if MPI:
            raise RuntimeError(
                "Recording of Solvers when running parallel code is not supported yet")
        self._rec_mgr.append(recorder)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Solver.
        """
        pass

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        self._system = system
        self._depth = depth
        self._rec_mgr.startup(self)
        self._rec_mgr.record_metadata(self)

        myoutputs = myresiduals = set()
        incl = self.recording_options['includes']
        excl = self.recording_options['excludes']

        if self.recording_options['record_solver_residuals']:
            if isinstance(self, NonlinearSolver):
                residuals = self._system._residuals
            else:  # it's a LinearSolver
                residuals = self._system._vectors['residual']['linear']

            myresiduals = {n for n in residuals._names
                           if check_path(n, incl, excl)}

        if isinstance(self, NonlinearSolver):
            outputs = self._system._outputs
        else:  # it's a LinearSolver
            outputs = self._system._vectors['output']['linear']

        myoutputs = {n for n in outputs._names
                     if check_path(n, incl, excl)}

        self._filtered_vars_to_record = {
            'out': myoutputs,
            'res': myresiduals
        }

    def _set_solver_print(self, level=2, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        self.options['iprint'] = level

    def _mpi_print(self, iteration, abs_res, rel_res):
        """
        Print residuals from an iteration.

        Parameters
        ----------
        iteration : int
            iteration counter, 0-based.
        abs_res : float
            current absolute residual norm.
        rel_res : float
            current relative residual norm.
        """
        if (self.options['iprint'] == 2 and self._system.comm.rank == 0):

            prefix = self._solver_info.prefix
            solver_name = self.SOLVER

            if prefix.endswith('precon:'):
                solver_name = solver_name[3:]

            print_str = prefix + solver_name
            print_str += ' %d ; %.9g %.9g' % (iteration, abs_res, rel_res)
            print(print_str)

    def _mpi_print_header(self):
        """
        Print header text before solving.
        """
        pass

    def _run_iterator(self):
        """
        Run the iterative solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)

        while self._iter_count < maxiter and \
                norm > atol and norm / norm0 > rtol:
            with Recording(type(self).__name__, self._iter_count, self) as rec:
                self._iter_execute()
                self._iter_count += 1
                self._run_apply()
                norm = self._iter_get_norm()
                # With solvers, we want to record the norm AFTER the call, but the call needs to
                # be wrapped in the with for stack purposes, so we locally assign  norm & norm0
                # into the class.
                rec.abs = norm
                rec.rel = norm / norm0

            if norm0 == 0:
                norm0 = 1
            self._mpi_print(self._iter_count, norm, norm / norm0)

        fail = (np.isinf(norm) or np.isnan(norm) or
                (norm > atol and norm / norm0 > rtol))

        if self._system.comm.rank == 0 or os.environ.get('USE_PROC_FILES'):
            prefix = self._solver_info.prefix + self.SOLVER
            if fail:
                if iprint > -1:
                    msg = ' Failed to Converge in {} iterations'.format(self._iter_count)
                    print(prefix + msg)

                # Raise AnalysisError if requested.
                if self.options['err_on_maxiter']:
                    msg = "Solver '{}' on system '{}' failed to converge."
                    raise AnalysisError(msg.format(self.SOLVER, self._system.pathname))

            elif iprint == 1:
                print(prefix + ' Converged in {} iterations'.format(self._iter_count))
            elif iprint == 2:
                print(prefix + ' Converged')

        return fail, norm, norm / norm0

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        pass

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        pass

    def _run_apply(self):
        """
        Run the appropriate apply method on the system.
        """
        pass

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        pass

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        pass

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.t.

        Returns
        -------
        boolean
            Flag for indicating child linerization
        """
        return True

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        pass

    def __str__(self):
        """
        Return a string representation of the solver.

        Returns
        -------
        str
            String representation of the solver.
        """
        return self.SOLVER

    def record_iteration(self, **kwargs):
        """
        Record an iteration of the current Solver.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments (used for abs and rel error).
        """
        if not self._rec_mgr._recorders:
            return

        metadata = create_local_meta(self.SOLVER)

        # Get the data
        data = {}
        # if self.options['record_abs_error'] or self.options['record_rel_error']:
        #     norm = self._iter_get_norm()

        if self.recording_options['record_abs_error']:
            # data['abs'] = norm
            data['abs'] = kwargs.get('abs')
        else:
            data['abs'] = None

        if self.recording_options['record_rel_error']:
            # data['rel'] = norm / self._norm0
            data['rel'] = kwargs.get('rel')
        else:
            data['rel'] = None

        if isinstance(self, NonlinearSolver):
            outputs = self._system._outputs
        else:  # it's a LinearSolver
            outputs = self._system._vectors['output']['linear']

        data['o'] = {}
        if 'out' in self._filtered_vars_to_record:
            for out in self._filtered_vars_to_record['out']:
                if out in outputs._names:
                    data['o'][out] = outputs._names[out]
        else:
            data['o'] = outputs

        if self.recording_options['record_solver_residuals']:

            if isinstance(self, NonlinearSolver):
                residuals = self._system._residuals
            else:  # it's a LinearSolver
                residuals = self._system._vectors['residual']['linear']

            data['r'] = {}
            if 'res' in self._filtered_vars_to_record:
                for res in self._filtered_vars_to_record['res']:
                    if res in residuals._names:
                        data['r'][res] = residuals._names[res]
            else:
                data['r'] = residuals
        else:
            data['r'] = None

        self._rec_mgr.record_iteration(self, data, metadata)


class NonlinearSolver(Solver):
    """
    Base class for nonlinear solvers.
    """

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        return self._run_iterator()

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        if self.options['maxiter'] > 0:
            self._run_apply()
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _run_apply(self):
        """
        Run the the apply_nonlinear method on the system.
        """
        recording_iteration.stack.append(('_run_apply', 0))
        self._system._apply_nonlinear()
        recording_iteration.stack.pop()

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        return self._system._residuals.get_norm()


class LinearSolver(Solver):
    """
    Base class for linear solvers.

    Attributes
    ----------
    _rel_systems : set of str
        Names of systems relevant to the current solve.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(LinearSolver, self).__init__(**kwargs)
        self._rel_systems = None

    def solve(self, vec_names, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            initial error.
        float
            error at the first iteration.
        """
        self._vec_names = vec_names
        self._rel_systems = rel_systems
        self._mode = mode
        return self._run_iterator()

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        system = self._system

        self._rhs_vecs = {}
        if self._mode == 'fwd':
            b_vecs = system._vectors['residual']
        else:  # rev
            b_vecs = system._vectors['output']

        for vec_name in self._vec_names:
            self._rhs_vecs[vec_name] = b_vecs[vec_name]._clone()

        if self.options['maxiter'] > 1:
            self._run_apply()
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _run_apply(self):
        """
        Run the the apply_linear method on the system.
        """
        recording_iteration.stack.append(('_run_apply', 0))

        system = self._system
        scope_out, scope_in = system._get_scope()
        system._apply_linear(self._vec_names, self._rel_systems, self._mode, scope_out, scope_in)

        recording_iteration.stack.pop()

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        system = self._system

        if self._mode == 'fwd':
            b_vecs = system._vectors['residual']
        else:  # rev
            b_vecs = system._vectors['output']

        norm = 0
        for vec_name in self._vec_names:
            if vec_name in system._rel_vec_names:
                b_vec = b_vecs[vec_name]
                b_vec -= self._rhs_vecs[vec_name]
                norm += b_vec.get_norm()**2

        return norm ** 0.5


class BlockLinearSolver(LinearSolver):
    """
    A base class for LinearBlockGS and LinearBlockJac.
    """

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        if isinstance(self._system._jacobian, AssembledJacobian):
            raise RuntimeError("A block linear solver '%s' is being used with "
                               "an AssembledJacobian in system '%s'" %
                               (self.SOLVER, self._system.pathname))
        return super(BlockLinearSolver, self)._iter_initialize()
