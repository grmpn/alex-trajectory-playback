"""LeRobot episode selection, joint ordering, and absent-reference semantics."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alex_playback.dataset import Dataset
from alex_playback.embodiment import ACTION_NAMES

ASSETS = Path(__file__).resolve().parents[1] / "purdue-alex-assets"


class LeRobotLoading(unittest.TestCase):
    def write_dataset(self, root, features, rows):
        (root / "meta").mkdir()
        (root / "data/chunk-000").mkdir(parents=True)
        (root / "meta/info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "fps": 30,
                    "features": features,
                }
            )
        )
        pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")

    def test_select_nonzero_episode_and_slice_with_reordered_joints(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            action = np.r_[np.arange(16) / 100, 0.25, 0.75]
            self.write_dataset(
                root,
                {"action": {"shape": [18], "names": ACTION_NAMES[::-1]}},
                [
                    {
                        "episode_index": episode,
                        "frame_index": i,
                        "source_frame_index": 100 + i * 4,
                        "timestamp": i / 30,
                        "action": action[::-1].tolist(),
                    }
                    for episode in (7, 9)
                    for i in range(5)
                ],
            )
            dataset = Dataset(root, None, ASSETS)
            self.assertEqual(dataset.episode_ids, [7, 9])
            episode = dataset.episode(9, max_frames=2, start_frame=1, end_frame=4)
            np.testing.assert_array_equal(episode.frame, [1, 2])
            np.testing.assert_array_equal(episode.source_frame, [104, 108])
            np.testing.assert_allclose(episode.action, np.tile(action, (2, 1)))
            self.assertEqual(episode.full_length, 5)
            self.assertFalse(episode.has_collision_labels)
            self.assertFalse(episode.has_valid_mask)
            self.assertIsNone(episode.ee)

    def test_ee_only_has_no_invented_joint_reference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ee = [[0.2, 0.3, 0.1, 0, 0, 0, 1, 0.4], [0.2, -0.3, 0.1, 0, 0, 0, 1, 0.8]]
            self.write_dataset(
                root,
                {"action": {"shape": [2, 8]}},
                [
                    {"episode_index": 4, "frame_index": i, "timestamp": i / 30, "action": ee}
                    for i in range(2)
                ],
            )
            episode = Dataset(root, None, ASSETS, ee_key="action").episode(4)
            self.assertFalse(episode.has_joint_actions)
            self.assertTrue(episode.has_ee_actions)
            np.testing.assert_allclose(episode.action[:, -2:], [[0.4, 0.8], [0.4, 0.8]])
            np.testing.assert_array_equal(episode.source_episode, [4, 4])
            np.testing.assert_array_equal(episode.source_frame, [0, 1])

    def test_unknown_joint_names_fail_before_simulation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_dataset(
                root,
                {"action": {"shape": [1], "names": ["other_robot_joint"]}},
                [
                    {"episode_index": 0, "frame_index": 0, "timestamp": 0.0, "action": [0.1]},
                ],
            )
            with self.assertRaisesRegex(ValueError, "Unsupported Alex action names"):
                Dataset(root, None, ASSETS)

    def test_subset_mask_does_not_mark_unspecified_joints_valid(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_dataset(
                root,
                {"action": {"shape": [1], "names": ["NECK_Z"]}},
                [
                    {
                        "episode_index": 2,
                        "frame_index": i,
                        "timestamp": i / 30,
                        "action": [0.1],
                        "quality.action_valid": [bool(i)],
                    }
                    for i in range(2)
                ],
            )
            episode = Dataset(root, None, ASSETS).episode(2)
            self.assertFalse(episode.valid[0].any())
            self.assertEqual(episode.valid[1].sum(), 1)
            self.assertTrue(episode.valid[1, ACTION_NAMES.index("NECK_Z")])


if __name__ == "__main__":
    unittest.main()
