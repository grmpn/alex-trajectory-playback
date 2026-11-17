"""Read the robot export without importing LeRobot or loading its videos."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Episode:
    index: int
    frame: np.ndarray
    source_frame: np.ndarray
    source_episode: np.ndarray
    time: np.ndarray
    action: np.ndarray
    ee: np.ndarray
    valid: np.ndarray
    collision: np.ndarray
    collision_pairs: list

    def __len__(self):
        return len(self.frame)


class Dataset:
    def __init__(self, root: Path, reports: Path, assets: Path):
        self.root = root.resolve()
        self.reports = reports.resolve()
        self.assets = assets.resolve()
        self.info = json.loads((root / "meta/info.json").read_text())
        self.fps = float(self.info["fps"])
        self.action_names = self.info["features"]["action"]["names"]
        card = yaml.safe_load((root / "README.md").read_text().split("---", 2)[1])
        self.embodiment = card["embodiment"]
        self.urdf = assets / self.embodiment["urdf"]
        revision = subprocess.check_output(
            ["git", "-C", str(assets), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != self.embodiment["asset_revision"]:
            raise ValueError("Asset checkout does not match the dataset's pinned revision")
        subprocess.run(
            ["git", "-C", str(assets), "diff", "--exit-code", "HEAD", "--", "assets", "src"],
            check=True,
            capture_output=True,
        )
        export = json.loads((reports / "region_export.json").read_text())
        self.contract = export["contract"]
        self.regions = [region for region in export["regions"] if region["exported"]]
        self.rows = []
        columns = [
            "episode_index",
            "frame_index",
            "source_frame_index",
            "source_episode_index",
            "timestamp",
            "action",
            "action.ee",
            "quality.action_valid",
        ]
        for path in sorted((root / "data").glob("chunk-*/*.parquet")):
            self.rows.extend(pq.read_table(path, columns=columns).to_pylist())
        if not self.rows:
            raise ValueError(f"No trajectory rows found in {root}")
        self.episode_ids = sorted({row["episode_index"] for row in self.rows})
        if len(set(self.action_names)) != 18 or self.action_names[-2:] != [
            "left_gripper_opening",
            "right_gripper_opening",
        ]:
            raise ValueError("Expected the Purdue Alex 16-joint + two-opening action contract")

    def episode(self, index: int, max_frames: int | None = None) -> Episode:
        rows = sorted(
            (r for r in self.rows if r["episode_index"] == index), key=lambda r: r["frame_index"]
        )
        if not rows:
            raise ValueError(f"Unknown episode {index}; available: {self.episode_ids}")
        source = np.array([r["source_frame_index"] for r in rows], dtype=np.int64)
        regions = [
            r
            for r in self.regions
            if r["source_frame_interval"] == [int(source[0]), int(source[-1]) + 1]
        ]
        if len(regions) != 1:
            raise ValueError(
                f"Cannot unambiguously align episode {index} to the export diagnostics"
            )
        directory = self.reports / f"evaluation-region-{regions[0]['source_index'] + 1}"
        score = json.loads((directory / "score.json").read_text())
        with np.load(directory / "benchmark_frame_metrics.npz", allow_pickle=False) as metrics:
            if not metrics["evaluated"][source].all():
                raise ValueError("Diagnostic source-frame alignment includes unevaluated frames")
            collision = ~metrics["collision_free"][source]
            ee_reference = metrics["tcp_pose_base"][source]
        action = np.asarray([r["action"] for r in rows], dtype=np.float64)
        ee = np.asarray([r["action.ee"] for r in rows], dtype=np.float64)
        valid = np.asarray([r["quality.action_valid"] for r in rows], dtype=bool)
        times = np.asarray([r["timestamp"] for r in rows], dtype=np.float64)
        frame = np.asarray([r["frame_index"] for r in rows], dtype=np.int64)
        if (
            action.shape != (len(rows), 18)
            or ee.shape != (len(rows), 2, 8)
            or valid.shape != action.shape
        ):
            raise ValueError("Unexpected dataset action/pose/mask shape")
        if not np.isfinite(action).all() or not np.isfinite(ee).all():
            raise ValueError("Nonfinite actions cannot be played back")
        if not np.array_equal(frame, np.arange(len(rows))) or not np.all(np.diff(source) == 1):
            raise ValueError("Episode frames must be contiguous and start at zero")
        if not np.allclose(times, frame / self.fps, atol=2e-6):
            raise ValueError("Dataset timestamps do not follow the declared FPS")
        if np.any(action[:, -2:] < 0) or np.any(action[:, -2:] > 1):
            raise ValueError("Gripper opening must be in [0, 1]")
        if not np.allclose(ee[:, :, :7], ee_reference, atol=2e-6):
            raise ValueError("Dataset TCP poses do not match the aligned diagnostic source frames")
        if not np.allclose(ee[:, :, 7], action[:, -2:], atol=1e-6):
            raise ValueError("Joint and EE actions disagree about gripper opening")
        if not np.allclose(np.linalg.norm(ee[:, :, 3:7], axis=-1), 1, atol=1e-5):
            raise ValueError("TCP orientations must be unit xyzw quaternions")
        pairs = [score["collision_frames"].get(str(int(i)), []) for i in source]
        if not np.array_equal(collision, [bool(p) for p in pairs]):
            raise ValueError("Collision labels and diagnostic collision pairs disagree")
        take = slice(None, max_frames)
        return Episode(
            index,
            frame[take],
            source[take],
            np.array([r["source_episode_index"] for r in rows], dtype=np.int64)[take],
            times[take],
            action[take],
            ee[take],
            valid[take],
            collision[take],
            pairs[take],
        )


def timing(fps: float, speed: float, max_dt: float) -> tuple[int, float]:
    """Retain every dataset frame; choose a fixed step no larger than max_dt."""
    if not all(np.isfinite(v) and v > 0 for v in (fps, speed, max_dt)):
        raise ValueError("FPS, speed, and physics step must be positive and finite")
    steps = max(1, int(np.ceil(1 / (fps * speed * max_dt) - 1e-12)))
    return steps, 1 / (fps * speed * steps)


def interpolate_poses(first: np.ndarray, second: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Linear position and shortest-arc SLERP, with xyzw throughout."""
    poses = np.empty((len(fractions), 2, 7))
    poses[:, :, :3] = first[None, :, :3] + fractions[:, None, None] * (second - first)[None, :, :3]
    for hand in range(2):
        poses[:, hand, 3:] = Slerp(
            [0, 1], Rotation.from_quat([first[hand, 3:7], second[hand, 3:7]])
        )(fractions).as_quat()
    return poses
