import numpy as np

import nengo
from nengo.utils.functions import piecewise


def test_inputgatedmemory(Simulator, plt, seed):
    to_memorize = 0.5
    start_memorizing = 0.4
    with nengo.Network(seed=seed) as net:
        test_input = nengo.Node(piecewise(
            {0.0: 0, 0.1: to_memorize, start_memorizing + 0.1: 0}))
        gate_input = nengo.Node(piecewise({0.0: 0, start_memorizing: 1}))
        reset_input = nengo.Node(0)

        mem = nengo.networks.InputGatedMemory(100, 1, difference_gain=5.0)
        nengo.Connection(test_input, mem.input)
        nengo.Connection(gate_input, mem.gate)
        nengo.Connection(reset_input, mem.reset)

        mem_p = nengo.Probe(mem.output, synapse=0.01)

    sim = Simulator(net)
    sim.run(0.5)

    data = sim.data[mem_p]
    t = sim.trange()

    plt.title("gating at %.1f s" % start_memorizing)
    plt.plot(t, data, label="value in memory")
    plt.axhline(to_memorize, c='k', lw=2, label="value to remember")
    plt.axvline(start_memorizing, c='k', ls=':', label="start gating")
    plt.legend(loc='best')

    assert abs(np.mean(data[t < 0.1])) < 0.01
    assert abs(np.mean(data[(t > 0.2) & (t <= 0.4)]) - 0.5) < 0.02
    assert abs(np.mean(data[t > 0.4]) - 0.5) < 0.02
