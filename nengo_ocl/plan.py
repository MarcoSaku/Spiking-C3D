import time

import pyopencl as cl
PROFILING_ENABLE = cl.command_queue_properties.PROFILING_ENABLE


class BasePlan(object):

    def __init__(self, name="", tag=None,
                 flops_per_call=None,
                 bw_per_call=None
                 ):
        self.name = name
        self.tag = tag
        self.atimes = []
        self.btimes = []
        self.ctimes = []
        self.n_calls = 0
        # -- floating-point ops per call
        self.flops_per_call = flops_per_call
        # -- bandwidth requirement per call
        self.bw_per_call = bw_per_call

    def __str__(self):
        return '<%s%s>' % (self.name, ": %s" % self.tag if self.tag else "")

    def __repr__(self):
        return '%s{%s%s}' % (
            self.__class__.__name__,
            self.name,
            ": %s" % self.tag if self.tag else "",
        )

    def update_profiling(self):
        pass


class PythonPlan(BasePlan):

    def __init__(self, function, **kwargs):
        super(PythonPlan, self).__init__(**kwargs)
        self.function = function

    def __call__(self, profiling=False):
        if profiling:
            t0 = time.time()
        self.function()
        if profiling:
            t1 = time.time()
            self.atimes.append(0)
            self.btimes.append(0)
            self.ctimes.append(t1 - t0)
            self.n_calls += 1


class Plan(BasePlan):

    def __init__(self, queue, kern, gsize, lsize, **kwargs):
        super(Plan, self).__init__(**kwargs)
        gsize = tuple(int(s) for s in gsize)
        assert 0 not in gsize, gsize
        if lsize is not None:
            lsize = tuple(int(s) for s in lsize)
            assert len(lsize) > 0
            assert 0 not in lsize, lsize
        self.queue = queue
        self.kern = kern
        self.gsize = gsize
        self.lsize = lsize
        self._events_to_profile = []

    def __call__(self, profiling=False):
        ev = self.enqueue(profiling=profiling)
        ev.wait()

    def update_profiling(self):
        for ev in self._events_to_profile:
            self.atimes.append(
                1e-9 * (ev.profile.submit - ev.profile.queued))
            self.btimes.append(
                1e-9 * (ev.profile.start - ev.profile.submit))
            self.ctimes.append(
                1e-9 * (ev.profile.end - ev.profile.start))
            self.n_calls += 1

        self._events_to_profile[:] = []

    def enqueue(self, wait_for=None, profiling=False):
        ev = cl.enqueue_nd_range_kernel(
            self.queue, self.kern, self.gsize, self.lsize,
            wait_for=wait_for)
        if profiling:
            self._events_to_profile.append(ev)
        return ev

    def __repr__(self):
        return '%s{%s%s %s %s}' % (
            self.__class__.__name__,
            self.name,
            ": %s" % self.tag if self.tag else "",
            self.gsize,
            self.lsize,
        )


class Plans(object):

    def __init__(self, planlist, profiling):
        self.plans = planlist
        self.profiling = profiling

    def __call__(self):
        return self.call_n_times(1)

    def call_n_times(self, n):
        last_event = self.enqueue_n_times(n)
        if last_event is not None:
            last_event.wait()

        if self.profiling:
            for p in self.plans:
                p.update_profiling()

    def enqueue_n_times(self, n):
        for _ in range(n):
            last_event = None
            for plan in self.plans:
                if hasattr(plan, 'enqueue'):
                    last_event = plan.enqueue(profiling=self.profiling)
                else:
                    # wait for last event and call
                    if last_event is not None:
                        last_event.wait()
                    plan(profiling=self.profiling)

        return last_event
