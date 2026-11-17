#!/usr/bin/env python3
"""Plot collision labels and TCP tracking errors from a completed playback run."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from alex_playback.model import pose_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    audits = sorted((args.run / "audit").glob("episode-*/audit.npz"))
    if not audits:
        parser.error("No audit logs found")
    fig, axes = plt.subplots(
        len(audits), 1, figsize=(13, 3 * len(audits)), squeeze=False, layout="constrained"
    )
    tracking, track_axes = plt.subplots(
        len(audits), 2, figsize=(13, 3 * len(audits)), squeeze=False, layout="constrained"
    )
    for row, path in enumerate(audits):
        with np.load(path, allow_pickle=False) as audit:
            source = audit["source_frame_index"]
            # A raster needs uniformly spaced columns. Gaps in source-video indices
            # must not stretch one sample across frames that were never evaluated.
            plot_frames = source if np.all(np.diff(source) == 1) else audit["frame_index"]
            frame_label = (
                "Source / episode frame" if np.all(np.diff(source) == 1) else "Episode frame"
            )
            labels, flags = [], []
            for known, label, values in (
                ("has_collision_labels", "Dataset collision", audit["dataset_collision"]),
                ("has_joint_reference", "Nominal mesh collision", audit["nominal_collision"]),
                ("has_valid_mask", "All actions invalid", ~audit["action_valid"].any(axis=1)),
            ):
                if bool(audit.get(known, True)):
                    labels.append(label)
                    flags.append(values)
        for mode, color in (("joint", "tab:blue"), ("ik", "tab:orange")):
            frames = args.run / mode / path.parent.name / "frames.npz"
            if not frames.exists():
                continue
            with np.load(frames, allow_pickle=False) as log:
                if not np.array_equal(log["source_frame_index"], source):
                    raise ValueError(f"Partial or misaligned playback log: {frames}")
                labels.extend([f"{mode}: measured mesh", f"{mode}: PhysX contact"])
                flags.extend([log["actual_mesh_collision"], log["physx_contact"]])
                error, _ = pose_error(log["actual_tcp_pose_base"], log["desired_tcp_pose_base"])
                for hand in range(2):
                    track_axes[row, hand].plot(
                        plot_frames, error[:, hand] * 1000, color=color, label=mode
                    )
        axis = axes[row, 0]
        if flags:
            axis.imshow(
                np.array(flags),
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                cmap=ListedColormap(["#eeeeee", "#be3030"]),
                vmin=0,
                vmax=1,
                extent=[plot_frames[0] - 0.5, plot_frames[-1] + 0.5, len(flags) - 0.5, -0.5],
            )
        else:
            axis.text(
                0.5, 0.5, "No collision reference available", ha="center", transform=axis.transAxes
            )
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(path.parent.name)
        axis.set_xlabel(frame_label)
        for hand in range(2):
            axis = track_axes[row, hand]
            axis.set_title(f"{path.parent.name} — {'left' if hand == 0 else 'right'} TCP")
            axis.set_xlabel(frame_label)
            axis.set_ylabel("Position error (mm)")
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend()
    fig.suptitle("Collision comparison — red means collision/contact or invalid action mask")
    fig.savefig(args.run / "collision-comparison.png", dpi=160)
    tracking.suptitle("Measured vs desired TCP position at original dataset timestamps")
    tracking.savefig(args.run / "tracking-errors.png", dpi=160)
    plt.close("all")
    print(f"Plots saved in {args.run}")


if __name__ == "__main__":
    main()
