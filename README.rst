DrugReLink |build|
==================
DrugReLink is a tool that optimizes, trains, and evaluates predictive
models for links in `Hetionet <https://het.io>`_ using different network
representation learning methods to compare learned features versus the
topological features presented in [himmelstein2017]_.

This package was developed during the master's thesis of
`Lingling Xu <https://github.com/lingling93>`_ under the supervision of
`Dr. Charles Tapley Hoyt <https://github.com/cthoyt>`_.

Installation
------------
Install from `GitHub <https://github.com/drugrelink/drugrelink>`_ with:

.. code-block:: sh

    $ git clone https://github.com/drugrelink/drugrelink.git
    $ cd drugrelink
    $ pip install -e .

CLI Usage
---------
Run on a subgraph of Hetionet with:

Download examples of configuration files from  ``/resources/config_examples/``

You can specify the output file path by adding "output_directory: path_of_output" to the configuration file.

.. code-block:: bash

    $ drugrelink path_of_the_config_file

.. [himmelstein2017] Himmelstein, D. S., *et al.* (2017). `Systematic integration of biomedical knowledge prioritizes
                     drugs for repurposing <https://doi.org/10.7554/eLife.26726>`_. ELife, 6.


.. |build| image:: https://travis-ci.com/drugrelink/drugrelink.svg?branch=master
    :target: https://travis-ci.com/drugrelink/drugrelink

