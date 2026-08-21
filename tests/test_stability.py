"""
Unit tests for ``stability.py`` covering :class:`StreakConfirmer` and
:class:`HysteresisLatch`.  The tests use only the standard library ``unittest``
framework as required.
"""

import unittest

# Import the module under test – it lives in the same directory as this file.
from vga.core import stability
from vga.core.stability import StreakConfirmer, HysteresisLatch


class TestStreakConfirmer(unittest.TestCase):
    def setUp(self) -> None:
        self.sc = StreakConfirmer()

    def test_initial_state(self) -> None:
        """A fresh instance reports no label and a streak of zero."""
        self.assertIsNone(self.sc.label, "label should be None before any observation")
        self.assertEqual(self.sc.streak, 0, "streak should be 0 before any observation")

    def test_observe_streak_increment_and_reset(self) -> None:
        """Repeated identical labels increase the streak; a different label resets it."""
        # First observation creates the initial streak.
        self.assertEqual(self.sc.observe("a"), 1)
        self.assertEqual(self.sc.label, "a")
        self.assertEqual(self.sc.streak, 1)

        # Same label increments the counter.
        self.assertEqual(self.sc.observe("a"), 2)
        self.assertEqual(self.sc.streak, 2)

        # A different label starts a new streak at 1 and updates ``label``.
        self.assertEqual(self.sc.observe("b"), 1)
        self.assertEqual(self.sc.label, "b")
        self.assertEqual(self.sc.streak, 1)

    def test_reset_clears_state(self) -> None:
        """Calling ``reset`` returns the object to its pristine state."""
        self.sc.observe("x")
        self.sc.reset()
        self.assertIsNone(self.sc.label, "label should be None after reset")
        self.assertEqual(self.sc.streak, 0, "streak should be 0 after reset")

    def test_multiple_hashable_label_types(self) -> None:
        """The confirmer works with any hashable label – strings, ints and tuples."""
        # Integer labels.
        self.assertEqual(self.sc.observe(10), 1)
        self.assertEqual(self.sc.observe(10), 2)
        self.assertEqual(self.sc.observe(20), 1)

        # Tuple labels (hashable).
        self.assertEqual(self.sc.observe((1, 2)), 1)
        self.assertEqual(self.sc.observe((1, 2)), 2)
        self.assertEqual(self.sc.observe((3, 4)), 1)


class TestHysteresisLatch(unittest.TestCase):
    def test_invalid_parameters_raise(self) -> None:
        """Both ``enter`` and ``leave`` must be >= 1; otherwise a ``ValueError`` is raised."""
        with self.assertRaises(ValueError):
            HysteresisLatch(0, 1)
        with self.assertRaises(ValueError):
            HysteresisLatch(1, 0)

    def test_start_on_initial_state(self) -> None:
        """When ``start_on`` is True the latch begins in the ON state."""
        latch = HysteresisLatch(enter=2, leave=3, start_on=True)
        self.assertTrue(latch.on)

        # With ``leave`` == 3 we need three consecutive falsy signals to turn OFF.
        self.assertTrue(latch.update(False))   # 1st falsy – still ON
        self.assertTrue(latch.update(False))   # 2nd falsy – still ON
        self.assertFalse(latch.update(False))  # 3rd falsy flips OFF
        self.assertFalse(latch.on)

    def test_enter_one_flips_immediately(self) -> None:
        """When ``enter`` == 1 a single truthy observation turns the latch ON."""
        latch = HysteresisLatch(enter=1, leave=2)
        self.assertFalse(latch.on)
        self.assertTrue(latch.update("yes"))   # truthy flips on
        self.assertTrue(latch.on)

    def test_leave_one_flips_immediately(self) -> None:
        """When ``leave`` == 1 a single falsy observation turns the latch OFF."""
        latch = HysteresisLatch(enter=2, leave=1, start_on=True)
        self.assertTrue(latch.on)
        self.assertFalse(latch.update(False))   # falsy flips off
        self.assertFalse(latch.on)

    def test_asymmetric_enter_and_leave(self) -> None:
        """Different ``enter``/``leave`` thresholds are honoured exactly.

        * Three consecutive truthy signals turn the latch ON.
        * Two consecutive falsy signals (while ON) turn it OFF, but an
          intervening truthy resets the falsy counter.
        """
        latch = HysteresisLatch(enter=3, leave=2)
        # Build up three truthies – latch should stay OFF until the third.
        self.assertFalse(latch.update(True))
        self.assertFalse(latch.update(1))   # still OFF
        self.assertTrue(latch.update("nonempty"))  # flips ON on the third
        self.assertTrue(latch.on)

        # First falsy – not enough to turn OFF yet.
        self.assertTrue(latch.update(False))
        # An interrupting truthy resets the falsy counter.
        self.assertTrue(latch.update(True))
        # Two more falsies now succeed in turning it OFF.
        self.assertTrue(latch.update(False))   # 1st falsy after reset
        self.assertFalse(latch.update(False))  # 2nd falsy flips OFF
        self.assertFalse(latch.on)

    def test_boundary_conditions_do_not_flip(self) -> None:
        """Observations that fall just short of the required count must not change state."""
        latch = HysteresisLatch(enter=2, leave=3)
        # One truthy is insufficient to turn ON.
        self.assertFalse(latch.update(True))
        # A falsy before reaching the threshold resets the counter.
        self.assertFalse(latch.update(False))
        # Another solitary truthy – still OFF.
        self.assertFalse(latch.update("yes"))
        # Second consecutive truthy finally flips ON.
        self.assertTrue(latch.update("yes2"))

    def test_truthiness_handling(self) -> None:
        """A range of values that are considered falsy must turn the latch OFF when ``leave`` == 1."""
        falsy_values = [0, "", None, [], {}, False]
        for val in falsy_values:
            latch = HysteresisLatch(enter=1, leave=1)
            # First a truthy signal puts it ON.
            self.assertTrue(latch.update("x"))
            # The current falsy value should immediately turn it OFF.
            self.assertFalse(latch.update(val))


if __name__ == "__main__":
    unittest.main()
