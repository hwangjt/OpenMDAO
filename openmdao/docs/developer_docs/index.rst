Developer Docs (if you're going to contribute code)
****************************************************


Developer install
----------------------

Use :code:`git` to clone the repository:

:code:`git clone http://github.com/OpenMDAO/OpenMDAO`

Use :code:`pip` to install openmdao locally:

:code:`cd OpenMDAO`

:code:`pip install -e .`

.. note::

    The :code:`-e` option tells pip to install directly from your repository.
    This is very useful when you're developing because when you change the code or pull new commits down from GitHub you don't necessarily need to re-run pip.


Building the Docs
-------------------

You can read the docs on line, so it is not necessary to build the them on your local machine.
But if you're going to build new features or add new examples, you'll want to build the docs locally while you write them.

:code:`cd openmdao/docs`

:code:`make all`

This will build the docs in :code:`openmdao/docs/_build/html.`

Then, just :code:`open openmdao/docs/_build/html/index.html` in a browser to begin.


Documentation Style Guide
----------------------------

This document exists to help OpenMDAO 2.x.x documentation writers to follow appropriate guidelines
in terms of formatting and embedding of code.

.. toctree::
    :maxdepth: 2

    style_guide/doc_style_guide.rst
    style_guide/sphinx_decorators.rst


Automating Documentation Build/Transfer
---------------------------------------

.. toctree::
    :maxdepth: 1

    automating_doc_build/automating_doc_build.rst
