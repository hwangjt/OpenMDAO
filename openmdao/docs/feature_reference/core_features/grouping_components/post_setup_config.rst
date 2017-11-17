
**************************************
Changing Model Settings After Setup
**************************************

After problem :code:`setup` has been called, you the entire model hierarchy has been instantiated and
:ref`setup and configure <feature_configure>` have been called on all groups and components.
However you may still want to make some changes to your model configuration.

OpenMDAO allows you to do a limited number of things after the Problem :code:`setup` is called, but before
you have called :code:`run_model` or :code:`run_driver`.
These include the following:

 - :ref:`Set initial conditions for unconnected inputs or states <set-and-get-variables>`
 - :ref:`Assign linear and nonlinear solvers <feature_solvers>`
 - Change solver settings
 - Assign Dense or Sparse Jacobians
 - :ref:`Set execution order <feature_set_order>`
 - Assign case recorders


Here, we instantiate a hierarchy of groups, and then change the solver to one that can solve this problem.

.. embed-test::
    openmdao.core.tests.test_problem.TestProblem.test_feature_post_setup_solver_configure
    :no-split:

.. tags:: Group, System