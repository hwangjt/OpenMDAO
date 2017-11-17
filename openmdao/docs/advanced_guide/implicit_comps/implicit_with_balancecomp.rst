In the :ref:`previous tutorial <defining_icomps_tutorial>`, we built up a model of an electrical circuit using
a combination of :ref:`ImplicitComponent <comp-type-3-implicitcomp>` and :ref:`ExplicitComponent <comp-type-2-explicitcomp>` instances.
In that tutorial, all of the implicit relationships in the model came directly from the physics of the model itself.

However, you often need to add implicit relationships to models by driving the values of two separate variables to be equal to each other.
In this tutorial we'll show you how to do that using the :ref:`BalanceComp <balancecomp_feature>`.

**************************************************************
Using BalanceComp to Create Implicit Relationships in Groups
**************************************************************

The electrical circuit model from the :ref:`previous tutorial <defining_icomps_tutorial>` represents a very basic
circuit with a current source of .1 Amps. Here is a reminder of what the circuit looks like:

.. figure:: images/circuit_diagram.png
   :align: center
   :width: 50%
   :alt: diagram of a simple circuit with two resistors and one diode

When we solved that circuit, the resulting voltage at *node 1* was 9.9 Volts.
Lets say you wanted to power this circuit with a 1.5-Volt battery, instead of a current source.
We can make a small modification to our original model to capture this new setup.

Given any value for :code:`source.I`, this model outputs the value for :code:`n1.V` that balances the model.
The voltage at the ground is also known via :code:`ground.V`. So the voltage across the current source is

.. math::
    V_{source} = V1 - V0

So to represent a voltage source with a specific voltage, we can add an additional state variable and residual equation to our model:

.. math::
    \mathcal{R}_{batt} = V1 - V0 - V_{source}^{*}

where :math:`V_{source}^{*}`, the desired source voltage, is given by the user as parameter to the model.

We could write a new component, inheriting from :ref:`ImplicitComponent <comp-type-3-implicitcomp>`, to include this new relationship into the model.
But OpenMDAO provides :ref:`BalanceComp <balancecomp_feature>`, a general utility component, designed specifically for this type of situation.

What we're going to do is add a :ref:`BalanceComp <balancecomp_feature>` to the top level of the model.
The :code:`BalanceComp` will define a residual that will drive the source current to force the delta-V across the battery to be what we want.
We'll also add an :ref:`ExecComp <feature_exec_comp>` to compute that delta-V from the ground voltage and the voltage at node 1 and then connect everything up.
Lastly, since we added an :ref:`ImplicitComponent <comp-type-3-implicitcomp>` at the top level of the model, we'll also move the :ref:`NewtonSolver <nlnewton>` up to the top level of the model too.

.. note::

    BalanceComp can handle more than just :math:`lhs-rhs=0`. It has a number of inputs that let you tweak that behavior.
    It can support multiple residuals, and array variables as well. Check out the :ref:`documentation <balancecomp_feature>` on it for details.

.. embed-test::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
    :no-split:


Understanding How Everything is Connected in This Model
**************************************************************

There are a number of connections in this model, and several different residuals being converged.
Trying to keep track of all the connections in your head can be a bit challenging, but OpenMDAO offers
some visualization tools to help see whats going on.

Add the following code to your run script to generate an :math:`N^2` diagram that represents the connections.

.. code::

    p.setup()

    from openmdao.api import view_model
    view_model(p)


.. raw:: html
    :file: n2.html