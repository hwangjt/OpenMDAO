"""Define the NonlinearBlockJac class."""
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.core.analysis_error import AnalysisError


class NonlinearBlockJac(NonlinearSolver):
    """
    Nonlinear block Jacobi solver.
    """

    SOLVER = 'NL: NLBJ'

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        self._solver_info.append_subsolver()
        self._system._transfer('nonlinear', 'fwd')

        with Recording('NonlinearBlockJac', 0, self) as rec:
            for subsys in self._system._subsystems_myproc:
                try:
                    subsys._solve_nonlinear()
                    failed = False
                except AnalysisError:
                    failed = True
                if np.any(self._system.comm.allgather(failed)):
                    raise AnalysisError('In nonlinear block Jacobi iteration for system {}, a subsystem had an analysis failure')
            self._system._check_reconf_update()
            rec.abs = 0.0
            rec.rel = 0.0

        self._solver_info.pop()

    def _mpi_print_header(self):
        """
        Print header text before solving.
        """
        if (self.options['iprint'] > 0 and self._system.comm.rank == 0):

            pathname = self._system.pathname
            if pathname:
                nchar = len(pathname)
                prefix = self._solver_info.prefix
                header = prefix + "\n"
                header += prefix + nchar * "=" + "\n"
                header += prefix + pathname + "\n"
                header += prefix + nchar * "="
                print(header)
