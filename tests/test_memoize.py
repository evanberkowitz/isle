"""
Test memoization.
"""

import gc
import random
import unittest

from isle.memoize import MemoizeMethod


class CallTracer:
    def __init__(self):
        self.neval = 0

    def wasEvaluated(self, reset=False):
        res = self.neval != 0
        if reset:
            self.reset()
        return res

    def reset(self):
        self.neval = 0

    def tick(self):
        self.neval += 1


class DelTracerManager:
    class DelTracer:
        def __init__(self, manager):
            self.manager = manager

        def __del__(self):
            self.manager.release()

    def __init__(self):
        self.counter = 0

    def ntraced(self):
        return self.counter

    def make(self):
        self.counter += 1
        return self.DelTracer(self)

    def release(self):
        assert self.counter > 0
        self.counter -= 1


class Class0:
    def __init__(self, name):
        self.name = name
        self.tracer = CallTracer()

    def undeco(self, a, b, c):
        self.tracer.tick()
        return self.name, a + b*100 + c*10000

    @MemoizeMethod(lambda a, b, c: (a, b, c))
    def decoAll(self, a, b, c):
        return self.undeco(a, b, c)

    @MemoizeMethod(lambda a, b: (a, b))
    def decoAB(self, a, b, c):
        return self.undeco(a, b, c)


class Class1:
    def __init__(self, name):
        self.name = name
        self.tracer = CallTracer()

    def undeco(self, l, x):
        self.tracer.tick()
        return self.name, len(l) + x*100

    @MemoizeMethod(lambda l, x: (len(l), x))
    def decoAll(self, l, x):
        return self.undeco(l, x)

    @MemoizeMethod(lambda l: len(l))
    def decol(self, l, x):
        return self.undeco(l, x)


class Class2:
    @MemoizeMethod(lambda t: t)
    def deco(self, t):
        pass


class Memoize(unittest.TestCase):
    def test_compareNonMemoizedFull(self):
        """
        Compare decorated method with non-decorated one with memoization w.r.t all parameters.
        """

        c0 = Class0("c0")
        args = [1, 2, 3]

        for _ in range(100):
            # change random arg or change none
            changeIdx = random.randint(0, len(args))
            if changeIdx != len(args):
                args[changeIdx] = random.randint(-99, 99)

            referenceVal = c0.undeco(*args)
            decoVal = c0.decoAll(*args)
            self.assertEqual(referenceVal, decoVal)

    def test_compareNonMemoizedFullMulti(self):
        """
        Compare decorated method with non-decorated one with memoization w.r.t all parameters
        on multiple instances.
        """

        cs = [Class0("c0"), Class0("c1"), Class0("c2")]
        args = [1, 2, 3]

        for _ in range(100):
            # change random arg or change none
            changeIdx = random.randint(0, len(args))
            if changeIdx != len(args):
                args[changeIdx] = random.randint(-99, 99)
            c = random.choice(cs)

            referenceVal = c.undeco(*args)
            decoVal = c.decoAll(*args)
            self.assertEqual(referenceVal, decoVal)

    def test_compareNonMemoizedPartial(self):
        """
        Compare decorated method with non-decorated one with memoization w.r.t one parameter.
        """

        c0 = Class0("c0")
        args = [1, 2, 3]
        previous = None

        for _ in range(100):
            # change random arg or change none
            changeIdx = random.randint(0, len(args))
            if changeIdx != len(args):
                args[changeIdx] = random.randint(-99, 99)

            referenceVal = c0.undeco(*args)
            decoVal = c0.decoAB(*args)
            if changeIdx >= len(args)-1:
                # Either argument c was changed or none
                #   => decorated method should return the same as before.
                if previous is not None:  # If this is the first call, just skip it.
                    self.assertEqual(previous, decoVal)
            else:
                # a or b was changed, memoization has to pick it up.
                self.assertEqual(referenceVal, decoVal)
            previous = decoVal

    def test_traceEvaluationClass0(self):
        """
        Run some manually constructed scenarios with Class0.
        """

        c0 = Class0("c0")
        # memoize w.r.t all arguments
        c0.decoAll(0, 1, 2)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAll(0, 1, 2)
        self.assertFalse(c0.tracer.wasEvaluated(True))
        c0.decoAll(0, 1, 1)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAll(1, 1, 1)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAll(1, 1, 1)
        self.assertFalse(c0.tracer.wasEvaluated(True))

        # memoize w.r.t first two arguments
        c0.decoAB(1, 1, 1)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAB(1, 1, 2)
        self.assertFalse(c0.tracer.wasEvaluated(True))
        c0.decoAB(1, 2, 2)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAB(1, 2, 2)
        self.assertFalse(c0.tracer.wasEvaluated(True))

        c1 = Class0("c1")
        # two instances to not interfere
        c0.decoAB(1, 2, 2)
        c1.decoAB(1, 2, 2)
        self.assertTrue(c1.tracer.wasEvaluated(True))
        c0.decoAll(1, 2, 2)
        c1.decoAll(1, 2, 2)
        self.assertTrue(c1.tracer.wasEvaluated(True))
        self.assertTrue(c0.tracer.wasEvaluated(True))

    def test_traceEvaluationClass1(self):
        """
        Run some manually constructed scenarios with Class1.
        """

        c0 = Class1("c0")
        # memoize w.r.t all arguments
        c0.decoAll([1, 2], 7)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAll([1, 2], 7)
        self.assertFalse(c0.tracer.wasEvaluated(True))
        c0.decoAll([1, 3], 7)
        self.assertFalse(c0.tracer.wasEvaluated(True))
        c0.decoAll([1, 2, 3], 7)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAll([1, 2, 3], 8)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decoAll([1, 2, 3], 8)
        self.assertFalse(c0.tracer.wasEvaluated(True))

        # memoize w.r.t. first argument
        c0.decol([1, 2, 3], 8)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decol([1, 2, 3], 8)
        self.assertFalse(c0.tracer.wasEvaluated(True))
        c0.decol([1, 2, 3], 7)
        self.assertFalse(c0.tracer.wasEvaluated(True))
        c0.decol([1, 2], 7)
        self.assertTrue(c0.tracer.wasEvaluated(True))
        c0.decol([1, 2], 7)
        self.assertFalse(c0.tracer.wasEvaluated(True))
        c0.decol([2, 2], 7)
        self.assertFalse(c0.tracer.wasEvaluated(True))

    def test_resourceManagement(self):
        """
        Test whether caches are freed when an object goes out of scope.
        """

        tracerManager = DelTracerManager()
        self.assertEqual(0, tracerManager.ntraced())

        c0 = Class2()
        c0.deco(tracerManager.make())
        self.assertEqual(1, tracerManager.ntraced())
        # evict previous tracer from cache
        c0.deco(tracerManager.make())
        gc.collect()
        self.assertEqual(1, tracerManager.ntraced())
        # remove object, should remove cache as well
        del c0
        gc.collect()
        self.assertEqual(0, tracerManager.ntraced())

        c0 = Class2()
        c1 = Class2()
        c0.deco(tracerManager.make())
        self.assertEqual(1, tracerManager.ntraced())
        c1.deco(tracerManager.make())
        self.assertEqual(2, tracerManager.ntraced())
        # evict in c0
        c0.deco(tracerManager.make())
        gc.collect()
        self.assertEqual(2, tracerManager.ntraced())
        # evict in c1
        c1.deco(tracerManager.make())
        gc.collect()
        self.assertEqual(2, tracerManager.ntraced())
        # remove c0
        del c0
        gc.collect()
        self.assertEqual(1, tracerManager.ntraced())
        # create new instance
        c2 = Class2()
        self.assertEqual(1, tracerManager.ntraced())
        c2.deco(tracerManager.make())
        self.assertEqual(2, tracerManager.ntraced())
        # remove c2
        del c2
        gc.collect()
        self.assertEqual(1, tracerManager.ntraced())
        # remove c1
        del c1
        gc.collect()
        self.assertEqual(0, tracerManager.ntraced())


if __name__ == '__main__':
    unittest.main()
