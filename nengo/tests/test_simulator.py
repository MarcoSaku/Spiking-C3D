import numpy as np

import nengo
import nengo.simulator


def test_steps(RefSimulator):
    m = nengo.Network(label="test_steps")
    sim = RefSimulator(m)
    assert sim.n_steps == 0
    sim.step()
    assert sim.n_steps == 1
    sim.step()
    assert sim.n_steps == 2


def test_time_steps(RefSimulator):
    m = nengo.Network(label="test_time_steps")
    sim = RefSimulator(m)
    assert np.allclose(sim.signals["__time__"], 0.00)
    sim.step()
    assert np.allclose(sim.signals["__time__"], 0.001)
    sim.step()
    assert np.allclose(sim.signals["__time__"], 0.002)


def test_time_absolute(Simulator):
    m = nengo.Network()
    sim = Simulator(m)
    sim.run(0.003)
    assert np.allclose(sim.trange(), [0.001, 0.002, 0.003])


def test_trange_with_probes(Simulator):
    dt = 1e-3
    m = nengo.Network()
    periods = dt * np.arange(1, 21)
    with m:
        u = nengo.Node(output=np.sin)
        probes = [nengo.Probe(u, sample_every=p, synapse=5*p) for p in periods]

    sim = Simulator(m, dt=dt)
    sim.run(0.333)
    for i, p in enumerate(periods):
        assert len(sim.trange(p)) == len(sim.data[probes[i]])


def test_probedict():
    """Tests simulator.ProbeDict's implementation."""
    raw = {"scalar": 5,
           "list": [2, 4, 6]}
    probedict = nengo.simulator.ProbeDict(raw)
    assert np.all(probedict["scalar"] == np.asarray(raw["scalar"]))
    assert np.all(probedict.get("list") == np.asarray(raw.get("list")))
