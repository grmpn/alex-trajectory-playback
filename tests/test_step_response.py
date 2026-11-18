"""Check discontinuous step timing and sustained settling diagnostics."""

import unittest

import numpy as np

from alex_playback.step_response import response_metrics, step_schedule


class StepResponseTests(unittest.TestCase):
    def test_step_switches_once_at_interval_start(self):
        time, selected, switch = step_schedule(0.103, 0.207, 0.01)
        self.assertAlmostEqual(switch, 0.11)
        self.assertAlmostEqual(time[-1], 0.32)
        self.assertEqual(np.count_nonzero(np.diff(selected.astype(int))), 1)
        self.assertFalse(selected[10])
        self.assertTrue(selected[11])
        self.assertEqual(time[np.flatnonzero(selected)[0]], switch)
        for bad in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                step_schedule(bad, 1, 0.01)

    def test_settling_requires_remaining_in_band(self):
        time = np.arange(31) / 10
        desired = np.zeros((31, 2, 3))
        actual = np.zeros_like(desired)
        actual[10:15, 0, 0] = 0.01
        actual[20, 0, 0] = 0.003  # Early in-band samples do not imply settling.
        actual[10:29, 1, 1] = 0.01  # Only 0.1 s in band at the end is insufficient.
        result = response_metrics(time, desired, actual, 1.0)
        self.assertAlmostEqual(result["settling_time_to_2mm_s"][0], 1.1)
        self.assertIsNone(result["settling_time_to_2mm_s"][1])


if __name__ == "__main__":
    unittest.main()
