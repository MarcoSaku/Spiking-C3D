"""
Simulator.py

Reference simulator for nengo models.
"""

from __future__ import print_function

from collections import Mapping
import logging

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder import Model
from nengo.builder.signal import SignalDict
from nengo.cache import get_default_decoder_cache
from nengo.utils.compat import range
from nengo.utils.graphs import toposort
from nengo.utils.progress import ProgressTracker
from nengo.utils.simulator import operator_depencency_graph

logger = logging.getLogger(__name__)


class ProbeDict(Mapping):
    """Map from Probe -> ndarray

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.
    """

    def __init__(self, raw):
        self.raw = raw

    def __getitem__(self, key):
        rval = self.raw[key]
        if isinstance(rval, list):
            rval = np.asarray(rval)
            rval.setflags(write=False)
        return rval

    def __str__(self):
        return str(self.raw)

    def __repr__(self):
        return repr(self.raw)

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)


class Simulator(object):
    """Reference simulator for Nengo models."""

    def __init__(self, network, dt=0.001, seed=None, model=None):
        """Initialize the simulator with a network and (optionally) a model.

        Most of the time, you will pass in a network and sometimes a dt::

            sim1 = nengo.Simulator(my_network)  # Uses default 0.001s dt
            sim2 = nengo.Simulator(my_network, dt=0.01)  # Uses 0.01s dt

        For more advanced use cases, you can initialize the model yourself,
        and also pass in a network that will be built into the same model
        that you pass in::

            sim = nengo.Simulator(my_network, model=my_model)

        If you want full control over the build process, then you can build
        your network into the model manually. If you do this, then you must
        explicitly pass in ``None`` for the network::

            sim = nengo.Simulator(None, model=my_model)

        Parameters
        ----------
        network : nengo.Network instance or None
            A network object to the built and then simulated.
            If a fully built ``model`` is passed in, then you can skip
            building the network by passing in network=None.
        dt : float, optional
            The length of a simulator timestep, in seconds.
        seed : int, optional
            A seed for all stochastic operators used in this simulator.
        model : nengo.builder.Model instance or None, optional
            A model object that contains build artifacts to be simulated.
            Usually the simulator will build this model for you; however,
            if you want to build the network manually, or to inject some
            build artifacts in the Model before building the network,
            then you can pass in a ``nengo.builder.Model`` instance.
        """
        if model is None:
            dt = float(dt)  # make sure it's a float (for division purposes)
            self.model = Model(dt=dt,
                               label="%s, dt=%f" % (network, dt),
                               decoder_cache=get_default_decoder_cache())
        else:
            self.model = model

        if network is not None:
            # Build the network into the model
            self.model.build(network)

        self.model.decoder_cache.shrink()

        # -- map from Signal.base -> ndarray
        self.signals = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))
        for op in self.model.operators:
            op.init_signals(self.signals)

        # Order the steps (they are made in `Simulator.reset`)
        self.dg = operator_depencency_graph(self.model.operators)
        self._step_order = [op for op in toposort(self.dg)
                            if hasattr(op, 'make_step')]

        # Add built states to the probe dictionary
        self._probe_outputs = self.model.params

        # Provide a nicer interface to probe outputs
        self.data = ProbeDict(self._probe_outputs)

        seed = np.random.randint(npext.maxint) if seed is None else seed
        self.reset(seed=seed)

    @property
    def dt(self):
        """The time step of the simulator"""
        return self.model.dt

    @dt.setter
    def dt(self, dummy):
        raise AttributeError("Cannot change simulator 'dt'. Please file "
                             "an issue at http://github.com/nengo/nengo"
                             "/issues and describe your use case.")

    @property
    def time(self):
        """The current time of the simulator"""
        return self.signals['__time__'].copy()

    def trange(self, dt=None):
        """Create a range of times matching probe data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., dt).

        Parameters
        ----------
        dt : float (optional)
            The sampling period of the probe to create a range for. If empty,
            will use the default probe sampling period.
        """
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    def _probe(self):
        """Copy all probed signals to buffers"""
        for probe in self.model.probes:
            period = (1 if probe.sample_every is None else
                      probe.sample_every / self.dt)
            if self.n_steps % period < 1:
                tmp = self.signals[self.model.sig[probe]['in']].copy()
                self._probe_outputs[probe].append(tmp)

    def step(self):
        """Advance the simulator by `self.dt` seconds.
        """
        self.n_steps += 1
        self.signals['__time__'][...] = self.n_steps * self.dt

        old_err = np.seterr(invalid='raise', divide='ignore')
        try:
            for step_fn in self._steps:
                step_fn()
        finally:
            np.seterr(**old_err)

        if len(self.model.probes) > 0:
            self._probe()

    def run(self, time_in_seconds, progress_bar=True):
        """Simulate for the given length of time.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        progress_bar : bool or ``ProgressBar`` or ``ProgressUpdater``, optional
            Progress bar for displaying the progress.

            By default, ``progress_bar=True``, which uses the default progress
            bar (text in most situations, or an HTML version in recent IPython
            notebooks).

            To disable the progress bar, use ``progress_bar=False``.

            For more control over the progress bar, pass in a
            :class:`nengo.utils.progress.ProgressBar`,
            or :class:`nengo.utils.progress.ProgressUpdater` instance.
        """
        steps = int(np.round(float(time_in_seconds) / self.dt))
        logger.info("Running %s for %f seconds, or %d steps",
                    self.model.label, time_in_seconds, steps)
        self.run_steps(steps, progress_bar=progress_bar)

    def run_steps(self, steps, progress_bar=True):
        """Simulate for the given number of `dt` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        progress_bar : bool or ``ProgressBar`` or ``ProgressUpdater``, optional
            Progress bar for displaying the progress.

            By default, ``progress_bar=True``, which uses the default progress
            bar (text in most situations, or an HTML version in recent IPython
            notebooks).

            To disable the progress bar, use ``progress_bar=False``.

            For more control over the progress bar, pass in a
            :class:`nengo.utils.progress.ProgressBar`,
            or :class:`nengo.utils.progress.ProgressUpdater` instance.
        """
        with ProgressTracker(steps, progress_bar) as progress:
            for i in range(steps):
                self.step()
                progress.step()

    def reset(self, seed=None):
        """Reset the simulator state.

        Parameters
        ----------
        seed : int, optional
            A seed for all stochastic operators used in the simulator.
            This will change the random sequences generated for noise
            or inputs (e.g. from Processes), but not the built objects
            (e.g. ensembles, connections).
        """
        if seed is not None:
            self.seed = seed

        self.n_steps = 0
        self.signals['__time__'][...] = 0

        # reset signals
        for key in self.signals:
            if key != '__time__':
                self.signals.reset(key)

        # rebuild steps (resets ops with their own state, like Processes)
        self.rng = np.random.RandomState(self.seed)
        self._steps = [op.make_step(self.signals, self.dt, self.rng)
                       for op in self._step_order]

        # clear probe data
        for probe in self.model.probes:
            self._probe_outputs[probe] = []
