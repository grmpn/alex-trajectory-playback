"""Regression checks for timing, quaternion interpolation, and importer repair."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from alex_playback.dataset import interpolate_poses, timing
from alex_playback.importing import compact_obj


class PlaybackContracts(unittest.TestCase):
    def test_speed_preserves_frame_boundaries(self):
        for speed in (0.1, 0.5, 1, 1.7, 2, 20):
            steps, dt = timing(30, speed, 1 / 240)
            self.assertGreaterEqual(steps, 1)
            self.assertLessEqual(dt, 1 / 240 + 1e-12)
            self.assertAlmostEqual(268 * steps * dt * speed, 268 / 30)
        for speed in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                timing(30, speed, 1 / 240)

    def test_slerp_crosses_quaternion_sign_boundary(self):
        first = np.zeros((2, 7))
        second = np.zeros((2, 7))
        first[:, 3:] = Rotation.from_euler("z", 170, degrees=True).as_quat()
        second[:, 3:] = Rotation.from_euler("z", -170, degrees=True).as_quat()
        second[:, :3] = [1, 2, 3]
        result = interpolate_poses(first, second, np.array([0, 0.5, 1]))
        np.testing.assert_allclose(result[1, 0, :3], [0.5, 1, 1.5])
        angle = Rotation.from_quat(result[1, 0, 3:]).magnitude()
        self.assertAlmostEqual(angle, np.pi)
        # Antipodal quaternion representations must describe constant rotation.
        second = first.copy()
        second[:, 3:] *= -1
        result = interpolate_poses(first, second, np.array([0.5]))
        np.testing.assert_allclose(abs(np.sum(result[0, :, 3:] * first[:, 3:], axis=-1)), 1)

    def test_obj_compaction_preserves_faces_and_non_position_indices(self):
        with TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.obj", Path(directory) / "clean.obj"
            source.write_text(
                "mtllib source.mtl\nv 99 99 99\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvn 0 0 1\nusemtl gray\nf -3/1/1 -2/1/1 -1/1/1\n"
            )
            self.assertEqual(compact_obj(source, target), 1)
            text = target.read_text()
            self.assertNotIn("99 99 99", text)
            self.assertIn("f 1/1/1 2/1/1 3/1/1", text)
            self.assertIn("vn 0 0 1\nusemtl gray", text)
            self.assertIn(f"mtllib {Path(directory) / 'source.mtl'}", text)


if __name__ == "__main__":
    unittest.main()
