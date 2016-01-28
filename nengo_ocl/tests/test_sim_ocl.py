"""
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_sim_ocl.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import fnmatch
import sys
import os

import pyopencl as cl
import pytest

import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import find_modules, allclose

from nengo_ocl import sim_ocl
from nengo_ocl.tests.utils import load_functions

ctx = cl.create_some_context()


def OclSimulator(*args, **kwargs):
    return sim_ocl.Simulator(*args, context=ctx, **kwargs)


def pytest_funcarg__Simulator(request):
    """The Simulator class being tested."""
    return OclSimulator


def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 1e-7)
    return allclose(*args, **kwargs)


def xfail(pattern, msg):
    for key in tests:
        if fnmatch.fnmatch(key, pattern):
            tests[key] = pytest.mark.xfail(True, reason=msg)(tests[key])


nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^Simulator$')

# learning rules
xfail('test.nengo.tests.test_learning_rules.test_unsupervised',
      "Unsupervised learning rules not implemented")
xfail('test.nengo.tests.test_learning_rules.test_dt_dependence',
      "Filtering matrices (i.e. learned transform) not implemented")

# neuron types
xfail('test.nengo.tests.test_neurons.test_alif*',
      "ALIF neurons not implemented")
xfail('test.nengo.tests.test_neurons.test_izhikevich',
      "Izhikevich neurons not implemented")
xfail('test.nengo.tests.test_neurons.test_lif_min_voltage',
      "Min voltage not implemented")

# nodes
xfail('test.nengo.tests.test_node.test_none',
      "No error if nodes output None")

# processes
xfail('test.nengo.tests.test_processes.test_brownnoise',
      "Filtered noise processes not yet implemented")
xfail('test.nengo.tests.test_ensemble.test_noise_copies_ok',
      "Filtered noise processes not yet implemented")
xfail('test.nengo.tests.test_simulator.test_noise_copies_ok',
      "Filtered noise processes not yet implemented")

# synapses
xfail('test.nengo.tests.test_synapses.test_triangle',
      "Only linear filters implemented")
nengo.tests.test_synapses.allclose = allclose_tol  # looser tolerances

# resetting
xfail('test.nengo.tests.test_learning_rules.test_reset',
      "Resetting not implemented")
xfail('test.nengo.tests.test_neurons.test_reset',
      "Resetting not implemented")
xfail('test.nengo.tests.test_processes.test_reset',
      "Resetting not implemented")

locals().update(tests)


if __name__ == '__main__':
    # To profile, run `python -m cProfile -o test_sim_ocl.log test_sim_ocl.py`.
    # Appending the argument `-k <filter>` allows you to control which tests
    # are run (e.g. `-k "test_ensemble."` runs all tests in test_ensemble.py).
    pytest.main(sys.argv)
