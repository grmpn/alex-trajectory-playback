"""Portable logs and frame-aligned collision/tracking summaries."""

import csv
import json
from pathlib import Path

import numpy as np

from .model import pose_error


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def confusion(observed, expected, known=True):
    if not known:
        return None
    observed, expected = np.asarray(observed, dtype=bool), np.asarray(expected, dtype=bool)
    return {
        "frames": len(expected),
        "detected": int(observed.sum()),
        "expected": int(expected.sum()),
        "true_positive": int(np.sum(observed & expected)),
        "false_positive": int(np.sum(observed & ~expected)),
        "false_negative": int(np.sum(~observed & expected)),
        "true_negative": int(np.sum(~observed & ~expected)),
        "matches": bool(np.array_equal(observed, expected)),
    }


def ranges(indices):
    """Inclusive runs, always in the explicitly named frame index space."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [[int(group[0]), int(group[-1])] for group in groups]


def audit_episode(dataset, model, episode, destination):
    destination.mkdir(parents=True, exist_ok=False)
    configurations = np.array(
        [model.action_configuration(dataset.action_names, a) for a in episode.action]
    )
    poses = np.array([model.tcp(q) for q in configurations])
    if episode.ee is None:
        episode.ee = np.concatenate((poses, episode.action[:, -2:, None]), axis=-1)
    position_error, angle_error = pose_error(poses, episode.ee[:, :, :7])
    if (
        episode.has_joint_actions
        and dataset.enforce_fk_contract
        and (position_error.max() > 2e-6 or angle_error.max() > 2e-5)
    ):
        raise ValueError("URDF FK disagrees with the dataset TCP contract")
    pairs = (
        [model.collisions(q) for q in configurations]
        if episode.has_joint_actions
        else [[] for _ in episode.frame]
    )
    collision = np.array([bool(p) for p in pairs])
    all_invalid = ~episode.valid.any(axis=1)
    summary = {
        "episode": episode.index,
        "frames": len(episode),
        "has_joint_reference": episode.has_joint_actions,
        "has_collision_labels": episode.has_collision_labels,
        "has_valid_mask": episode.has_valid_mask,
        "nominal_mesh_vs_dataset": confusion(
            collision, episode.collision, episode.has_collision_labels and episode.has_joint_actions
        ),
        "collision_source_frame_ranges_inclusive": ranges(episode.source_frame[collision]),
        "collision_episode_frame_ranges_inclusive": ranges(episode.frame[collision]),
        "mismatch_source_frames": episode.source_frame[collision != episode.collision].tolist()
        if episode.has_collision_labels and episode.has_joint_actions
        else None,
        "all_actions_invalid": int(all_invalid.sum()) if episode.has_valid_mask else None,
        "collisions_not_fully_masked": int(np.sum(collision & ~all_invalid))
        if episode.has_valid_mask and episode.has_joint_actions
        else None,
        "fully_masked_without_collision": int(np.sum(all_invalid & ~collision))
        if episode.has_valid_mask and episode.has_joint_actions
        else None,
        "max_fk_position_error_m": float(position_error.max())
        if episode.has_joint_actions
        else None,
        "max_fk_orientation_error_rad": float(angle_error.max())
        if episode.has_joint_actions
        else None,
        "method": "Pinocchio / Coal, discrete URDF collision meshes; rigid, adjacent, and upstream exclusions",
    }
    write_json(destination / "summary.json", summary)
    write_json(
        destination / "collision_pairs.json",
        [
            {
                "frame_index": int(episode.frame[i]),
                "source_frame_index": int(episode.source_frame[i]),
                "detected": p,
                "dataset": episode.collision_pairs[i],
            }
            for i, p in enumerate(pairs)
            if p or episode.collision_pairs[i]
        ],
    )
    np.savez_compressed(
        destination / "audit.npz",
        frame_index=episode.frame,
        source_frame_index=episode.source_frame,
        dataset_collision=episode.collision,
        nominal_collision=collision,
        action_valid=episode.valid,
        has_joint_reference=episode.has_joint_actions,
        has_collision_labels=episode.has_collision_labels,
        has_valid_mask=episode.has_valid_mask,
        joint_names=np.array(model.names),
        joint_position=np.array([model.ordered(q, model.names) for q in configurations]),
        tcp_pose_base=poses,
    )
    return summary, configurations, pairs


class Recorder:
    def __init__(self, destination, episode, joint_names, metadata):
        destination.mkdir(parents=True, exist_ok=False)
        self.destination = destination
        self.episode = episode
        self.joint_names = joint_names
        self.metadata = metadata
        self.steps = []
        self.frames = []
        self.pairs = []
        self.controls = []

    def control(self, simulation_time, phase, signals):
        self.controls.append(
            {
                "simulation_time_s": simulation_time,
                "phase": phase,
                **{name: np.asarray(value).copy() for name, value in signals.items()},
            }
        )

    def step(
        self,
        frame,
        dataset_time,
        simulation_time,
        desired_q,
        actual_q,
        desired_tcp,
        actual_tcp,
        contact=False,
        penetration=False,
        reference_q=None,
    ):
        self.steps.append(
            {
                "frame_index": frame,
                "dataset_time_s": dataset_time,
                "simulation_time_s": simulation_time,
                "desired_joint_position": np.asarray(desired_q).copy(),
                "actual_joint_position": np.asarray(actual_q).copy(),
                "reference_joint_position": np.asarray(
                    desired_q if reference_q is None else reference_q
                ).copy(),
                "desired_tcp_pose_base": np.asarray(desired_tcp).copy(),
                "actual_tcp_pose_base": np.asarray(actual_tcp).copy(),
                "physx_contact": contact,
                "physx_penetration": penetration,
            }
        )

    def frame(self, i, nominal_pairs, actual_pairs, contacts):
        step = dict(self.steps[-1])
        step.update(
            {
                "source_frame_index": int(self.episode.source_frame[i]),
                "source_episode_index": int(self.episode.source_episode[i]),
                "dataset_collision": bool(self.episode.collision[i]),
                "nominal_mesh_collision": bool(nominal_pairs),
                "actual_mesh_collision": bool(actual_pairs),
                "physx_contact": bool(contacts),
                "physx_penetration": any(v["min_separation_m"] < -1e-6 for v in contacts.values()),
                "action_valid": self.episode.valid[i],
            }
        )
        self.frames.append(step)
        self.pairs.append(
            {
                "frame_index": int(self.episode.frame[i]),
                "source_frame_index": int(self.episode.source_frame[i]),
                "nominal_mesh_pairs": nominal_pairs,
                "actual_mesh_pairs": actual_pairs,
                "dataset_pairs": self.episode.collision_pairs[i],
                "physx_contacts": [{"links": list(p), **v} for p, v in sorted(contacts.items())],
            }
        )

    def save(self, complete: bool):
        if self.controls:
            np.savez_compressed(
                self.destination / "control.npz",
                joint_names=np.array(self.joint_names),
                **{
                    key: np.asarray([row[key] for row in self.controls]) for key in self.controls[0]
                },
            )
        write_json(
            self.destination / "metadata.json",
            {
                **self.metadata,
                "complete": complete,
                "joint_names": self.joint_names,
                "pose_order": "x,y,z,qx,qy,qz,qw",
                "pose_frame": "PEDESTAL_LINK",
                "hand_order": ["left", "right"],
                "frame_contact_window": "union over substeps since the preceding dataset frame; frame zero uses final settling step",
            },
        )
        if not self.frames:
            return {"complete": False, "frames": 0}
        frame_arrays = {k: np.asarray([row[k] for row in self.frames]) for k in self.frames[0]}
        step_arrays = {k: np.asarray([row[k] for row in self.steps]) for k in self.steps[0]}
        for name, arrays in (("frames", frame_arrays), ("steps", step_arrays)):
            np.savez_compressed(
                self.destination / f"{name}.npz",
                joint_names=np.array(self.joint_names),
                has_joint_reference=self.episode.has_joint_actions,
                has_collision_labels=self.episode.has_collision_labels,
                has_valid_mask=self.episode.has_valid_mask,
                **arrays,
            )
        write_json(self.destination / "collision_pairs.json", self.pairs)
        position_error, angle_error = pose_error(
            frame_arrays["actual_tcp_pose_base"], frame_arrays["desired_tcp_pose_base"]
        )
        joint_error = frame_arrays["actual_joint_position"] - frame_arrays["desired_joint_position"]
        summary = {
            "episode": self.episode.index,
            "complete": complete,
            "full_episode": self.metadata["full_episode"],
            "frames": len(self.frames),
            "physics_samples": len(self.steps),
            "joint_rmse_rad": float(np.sqrt(np.mean(joint_error**2))),
            "joint_rmse_by_joint_rad": dict(
                zip(
                    self.joint_names, np.sqrt(np.mean(joint_error**2, axis=0)).tolist(), strict=True
                )
            ),
            "tcp_position_rmse_m": np.sqrt(np.mean(position_error**2, axis=0)).tolist(),
            "tcp_orientation_rmse_rad": np.sqrt(np.mean(angle_error**2, axis=0)).tolist(),
        }
        for kind in (
            "nominal_mesh_collision",
            "actual_mesh_collision",
            "physx_contact",
            "physx_penetration",
        ):
            flags = frame_arrays[kind]
            known = self.episode.has_collision_labels and (
                kind != "nominal_mesh_collision" or self.episode.has_joint_actions
            )
            summary[f"{kind}_vs_dataset"] = confusion(
                flags, frame_arrays["dataset_collision"], known
            )
            summary[f"{kind}_source_frame_ranges_inclusive"] = ranges(
                frame_arrays["source_frame_index"][flags]
            )
        write_json(self.destination / "summary.json", summary)
        with (self.destination / "frames.csv").open("w", newline="") as handle:
            keys = [
                "frame_index",
                "source_frame_index",
                "dataset_time_s",
                "simulation_time_s",
                "dataset_collision",
                "nominal_mesh_collision",
                "actual_mesh_collision",
                "physx_contact",
                "physx_penetration",
            ]
            writer = csv.DictWriter(
                handle,
                fieldnames=keys + ["all_actions_invalid", "left_tcp_error_m", "right_tcp_error_m"],
            )
            writer.writeheader()
            for i, row in enumerate(self.frames):
                writer.writerow(
                    {
                        **{key: row[key] for key in keys},
                        "dataset_collision": row["dataset_collision"]
                        if self.episode.has_collision_labels
                        else "",
                        "nominal_mesh_collision": row["nominal_mesh_collision"]
                        if self.episode.has_joint_actions
                        else "",
                        "all_actions_invalid": not row["action_valid"].any()
                        if self.episode.has_valid_mask
                        else "",
                        "left_tcp_error_m": position_error[i, 0],
                        "right_tcp_error_m": position_error[i, 1],
                    }
                )
        return summary
