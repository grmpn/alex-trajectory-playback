"""Read selected LeRobot v3 episodes without importing LeRobot or loading videos."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow_dataset
import yaml
from scipy.spatial.transform import Rotation, Slerp

from .embodiment import ACTION_NAMES, ARM_JOINTS, BASE_FRAME, TOOL_FRAMES, URDF


@dataclass
class Episode:
    index: int
    frame: np.ndarray
    source_frame: np.ndarray
    source_episode: np.ndarray
    time: np.ndarray
    action: np.ndarray
    ee: np.ndarray | None
    valid: np.ndarray
    collision: np.ndarray
    collision_pairs: list
    full_length: int
    has_joint_actions: bool
    has_ee_actions: bool
    has_collision_labels: bool
    has_valid_mask: bool

    def __len__(self):
        return len(self.frame)


class Dataset:
    def __init__(
        self, root: Path, reports: Path | None, assets: Path, joint_key="action", ee_key="action.ee"
    ):
        self.root = root.resolve()
        self.reports = reports.resolve() if reports else None
        self.assets = assets.resolve()
        self.info = json.loads((root / "meta/info.json").read_text())
        if not str(self.info.get("codebase_version", "")).startswith("v3"):
            raise ValueError("Expected a LeRobot v3 dataset (meta/info.json codebase_version v3.x)")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Dataset FPS must be positive and finite")
        self.action_names = ACTION_NAMES.copy()
        self.contract = {"robot_settings": {"arm_joints": ARM_JOINTS}}
        card = {}
        readme = root / "README.md"
        if readme.exists() and readme.read_text().startswith("---"):
            card = yaml.safe_load(readme.read_text().split("---", 2)[1]) or {}
        revision = subprocess.check_output(
            ["git", "-C", str(assets), "rev-parse", "HEAD"], text=True
        ).strip()
        self.embodiment = card.get(
            "embodiment",
            {
                "asset_revision": revision,
                "urdf": URDF,
                "base_frame": BASE_FRAME,
                "tool_frames": TOOL_FRAMES,
            },
        )
        self.urdf = assets / self.embodiment["urdf"]
        if revision != self.embodiment["asset_revision"]:
            raise ValueError("Asset checkout does not match the dataset's pinned revision")
        subprocess.run(
            ["git", "-C", str(assets), "diff", "--exit-code", "HEAD", "--", "assets", "src"],
            check=True,
            capture_output=True,
        )
        self.enforce_fk_contract = bool(self.reports) or (
            card.get("kind") == "robot" and str(card.get("contract_version")) == "3.0"
        )
        self.regions = []
        if self.reports:
            export = json.loads((self.reports / "region_export.json").read_text())
            self.contract = export["contract"]
            self.regions = [region for region in export["regions"] if region["exported"]]
        files = sorted((root / "data").rglob("*.parquet"))
        if not files:
            raise ValueError(f"No trajectory Parquet files found in {root / 'data'}")
        self.table = arrow_dataset.dataset([str(p) for p in files], format="parquet")
        self.columns = set(self.table.schema.names)
        self.episode_ids = sorted(
            self.table.to_table(columns=["episode_index"])["episode_index"].unique().to_pylist()
        )
        features = self.info["features"]
        self.joint_key = joint_key if joint_key in self.columns else None
        self.ee_key = ee_key if ee_key in self.columns else None
        if self.joint_key == self.ee_key:
            self.joint_key = None
        if self.joint_key and features[self.joint_key].get("shape") in ([2, 7], [2, 8]):
            raise ValueError(f"{joint_key} contains EE poses; select it with --ee-key {joint_key}")
        if not self.joint_key and not self.ee_key:
            raise ValueError(f"Neither {joint_key!r} nor {ee_key!r} exists in this dataset")
        self.input_names = []
        if self.joint_key:
            self.input_names = features[self.joint_key].get("names")
            if not isinstance(self.input_names, list) or not self.input_names:
                raise ValueError(
                    f"features.{self.joint_key}.names must list the Alex joint/action names"
                )
            if len(set(self.input_names)) != len(self.input_names):
                raise ValueError("Joint action names must be unique")
            unknown = set(self.input_names) - set(ACTION_NAMES)
            if unknown:
                raise ValueError(f"Unsupported Alex action names: {sorted(unknown)}")
        self.action_indices = [ACTION_NAMES.index(name) for name in self.input_names]

    def episode(
        self,
        index: int,
        max_frames: int | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Episode:
        if index not in self.episode_ids:
            raise ValueError(f"Unknown episode {index}; available: {self.episode_ids}")
        wanted = [
            "episode_index",
            "frame_index",
            "source_frame_index",
            "source_episode_index",
            "timestamp",
            self.joint_key,
            self.ee_key,
            "quality.action_valid",
        ]
        columns = list(dict.fromkeys(key for key in wanted if key in self.columns))
        rows = self.table.to_table(
            columns=columns, filter=arrow_dataset.field("episode_index") == index
        ).to_pylist()
        rows.sort(key=lambda row: row["frame_index"])
        n = len(rows)
        if not n:
            raise ValueError(f"Episode {index} contains no rows")
        frame = np.array([r["frame_index"] for r in rows], dtype=np.int64)
        source = np.array(
            [r.get("source_frame_index", r["frame_index"]) for r in rows], dtype=np.int64
        )
        times = np.array([r["timestamp"] for r in rows], dtype=np.float64)
        if not np.array_equal(frame, np.arange(n)):
            raise ValueError("Episode frame indices must be contiguous and start at zero")
        timestamp_tolerance = max(2e-6, 2 * np.finfo(np.float32).eps * np.max(np.abs(times)))
        if not np.isfinite(times).all() or not np.allclose(
            times - times[0], frame / self.fps, atol=timestamp_tolerance, rtol=0
        ):
            raise ValueError("Dataset timestamps must follow the declared FPS")
        action = np.zeros((n, 18), dtype=np.float64)
        action[:, -2:] = 1.0  # Unspecified joints hold zero; unspecified grippers stay open.
        if self.joint_key:
            values = np.asarray([r[self.joint_key] for r in rows], dtype=np.float64)
            if values.shape != (n, len(self.input_names)):
                raise ValueError("Joint actions do not match features.names")
            action[:, self.action_indices] = values
        ee = None
        if self.ee_key:
            ee = np.asarray([r[self.ee_key] for r in rows], dtype=np.float64)
            if ee.shape == (n, 2, 7):
                ee = np.concatenate((ee, action[:, -2:, None]), axis=-1)
            if ee.shape != (n, 2, 8) or not np.isfinite(ee).all():
                raise ValueError(
                    "EE actions must be finite (frames, 2, 7 or 8) poses, left then right"
                )
            if not np.allclose(np.linalg.norm(ee[:, :, 3:7], axis=-1), 1, atol=1e-5):
                raise ValueError("TCP orientations must be unit xyzw quaternions")
            for hand, name in enumerate(ACTION_NAMES[-2:]):
                if name in self.input_names and not np.allclose(
                    ee[:, hand, 7], action[:, -2 + hand], atol=1e-6
                ):
                    raise ValueError("Joint and EE actions disagree about gripper opening")
                action[:, -2 + hand] = ee[:, hand, 7]
        if not np.isfinite(action).all() or np.any((action[:, -2:] < 0) | (action[:, -2:] > 1)):
            raise ValueError(
                "Joint actions must be finite; normalized gripper openings must be in [0, 1]"
            )
        valid = np.ones((n, 18), dtype=bool)
        has_mask = "quality.action_valid" in self.columns
        if has_mask:
            # Unspecified joints are held defaults, not valid observed actions.
            valid[:] = False
            mask = np.asarray([r["quality.action_valid"] for r in rows], dtype=bool)
            if self.joint_key and mask.shape == (n, len(self.input_names)):
                valid[:, self.action_indices] = mask
            elif mask.shape == valid.shape:
                valid = mask
            elif mask.shape == (n, 2, 8):
                for hand in range(2):
                    valid[:, hand * 7 : hand * 7 + 7] = mask[:, hand, :7].all(
                        axis=-1, keepdims=True
                    )
                    valid[:, 16 + hand] = mask[:, hand, 7]
            else:
                raise ValueError("Unsupported quality.action_valid mask shape")
        collision = np.zeros(n, dtype=bool)
        pairs = [[] for _ in rows]
        if self.reports:
            regions = [
                r
                for r in self.regions
                if r["source_frame_interval"] == [int(source[0]), int(source[-1]) + 1]
            ]
            if len(regions) != 1:
                raise ValueError(f"Cannot align episode {index} to the supplied export diagnostics")
            directory = self.reports / f"evaluation-region-{regions[0]['source_index'] + 1}"
            score = json.loads((directory / "score.json").read_text())
            with np.load(directory / "benchmark_frame_metrics.npz", allow_pickle=False) as metrics:
                if not metrics["evaluated"][source].all():
                    raise ValueError(
                        "Diagnostic source-frame alignment includes unevaluated frames"
                    )
                collision = ~metrics["collision_free"][source]
                if ee is not None and not np.allclose(
                    ee[:, :, :7], metrics["tcp_pose_base"][source], atol=2e-6
                ):
                    raise ValueError("Dataset TCP poses disagree with the aligned diagnostics")
            pairs = [score["collision_frames"].get(str(int(i)), []) for i in source]
            if not np.array_equal(collision, [bool(p) for p in pairs]):
                raise ValueError("Collision labels and diagnostic pairs disagree")
        stop = n if end_frame is None else min(n, end_frame)
        if max_frames is not None:
            stop = min(stop, start_frame + max_frames)
        if start_frame < 0 or stop <= start_frame:
            raise ValueError(f"Empty frame selection for episode {index} (length {n})")
        take = slice(start_frame, stop)
        return Episode(
            index,
            frame[take],
            source[take],
            np.array([r.get("source_episode_index", index) for r in rows], dtype=np.int64)[take],
            times[take],
            action[take],
            ee[take] if ee is not None else None,
            valid[take],
            collision[take],
            pairs[take],
            n,
            bool(self.joint_key),
            bool(self.ee_key),
            bool(self.reports),
            has_mask,
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
