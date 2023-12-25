.. MNIST sandbox mlops CS course 2023 documentation master file, created by
   sphinx-quickstart on Thu Sep 28 23:05:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MNIST sandbox for mlops CS course 2023's documentation!
==================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   mnist_sandbox

MNIST CNN solution
=========

Simple CNN for MNIST digits dataset solution. To run simple train & evaluation:

.. code-block:: bash

   poetry install

.. code-block:: bash

   poetry run python3 mnist_sandbox/main.py
