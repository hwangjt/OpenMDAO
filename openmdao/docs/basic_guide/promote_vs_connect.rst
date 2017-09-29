.. _guide_promote_vs_connect:

************************************************
Linking Variables with Promotion vs Connection
************************************************

In the previous tutorial we built up a model of the Sellar problem using two disciplinary components and a few :code:`ExecComps`.
In order to get OpenMDAO to pass the data between all the components,
we linked everything up using promoted variables so that data passed from outputs to inputs with the same promoted name.

Promoting variables is often a convenient way to establish the data passing links from outputs to inputs.
However, you can also use calls to the :code:`connect` method in order to link outputs to inputs without having to promote anything.
Here is how you would define the same Sellar model using:

    #. Variable promotion
    #. Connect statements
    #. Both variable promotion and connect statements

All three will give the exact same answer, but the way you address the variables will be slightly different in each one.

Variable Promotion
********************

.. embed-test::
    openmdao.test_suite.test_examples.test_sellar_mda_promote_connect.TestSellarMDAPromoteConnect.test_sellar_mda_promote

There are a few important details to note:

    * The promoted name of an output has to be unique within that level of the hierarchy (i.e. you can't have two outputs with the same name)
    * You are allowed to have multiple inputs promoted to the same name, but in order for a connection to be made there must also be an output with the same name. Otherwise, no connection is made.
    * You can use glob-patterns to promote lots of variables without specifying them all, but try to limit your usage of :code:`promotes=['*']`.
      Though it may seem like a convinient way to do things, it can make it really hard for other people who are reading your code to understand what variables connect to where.
      It is ok to use it in cases where it won't cause confusion,
      such as with :code:`cycle` which just exists to allow for the nonlinear solver to converge the two components or when you have :code:`ExecComps` that make it clear what the IO of that component is anyway.


.. note::

    For a more detailed set of examples for how to promote variables, check out the :ref:`feature doc on adding sub-systems to a group <feature_adding_subsystem_to_a_group>`.
    There are some more advanced things you can do, such as variable name aliasing and connecting a sub-set of indices from the output array of one component to the input of another



Connect Statements
**************************

The exact same model results can be achieved using :code:`connect` statements instead of promotions.
However, take careful note of how the variables are addressed in those connect and print statements.

.. embed-test::
    openmdao.test_suite.test_examples.test_sellar_mda_promote_connect.TestSellarMDAPromoteConnect.test_sellar_mda_connect



Variable Promotion and Connect Statements
********************************************

It is also possible to combine promotion and connection in a single model.
Here, notice that we do not have to add "cycle" in front of anything, because we promoted all the variables up from that group.

.. embed-test::
    openmdao.test_suite.test_examples.test_sellar_mda_promote_connect.TestSellarMDAPromoteConnect.test_sellar_mda_promote_connect
