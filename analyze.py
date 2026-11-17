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
            labels = ["Dataset collision", "Nominal mesh collision", "All actions invalid"]
            flags = [
                audit["dataset_collision"],
                audit["nominal_collision"],
                ~audit["action_valid"].any(axis=1),
            ]
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
                        source, error[:, hand] * 1000, color=color, label=mode
                    )
        axis = axes[row, 0]
        axis.imshow(
            np.array(flags),
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=ListedColormap(["#eeeeee", "#be3030"]),
            vmin=0,
            vmax=1,
            extent=[source[0] - 0.5, source[-1] + 0.5, len(flags) - 0.5, -0.5],
        )
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(path.parent.name)
        axis.set_xlabel("Source video frame")
        for hand in range(2):
            axis = track_axes[row, hand]
            axis.set_title(f"{path.parent.name} — {'left' if hand == 0 else 'right'} TCP")
            axis.set_xlabel("Source video frame")
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
