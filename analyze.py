#!/usr/bin/env python3
"""Plot desired/actual TCP positions, tracking errors, and collision diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

from alex_playback.model import pose_error


def plot_tcp_positions(run, output):
    """Use every physics sample, in the recorded base frame and simulation clock."""
    logs = sorted(
        run.glob("*/episode-*/steps.npz"),
        key=lambda path: (
            path.parent.parent.name != "joint",
            int(path.parent.name.removeprefix("episode-")),
        ),
    )
    if not logs:
        return []
    summaries = []
    with PdfPages(output / "tcp-position-tracking.pdf") as pdf:
        for path in logs:
            metadata = json.loads((path.parent / "metadata.json").read_text())
            with np.load(path, allow_pickle=False) as log:
                time = log["simulation_time_s"]
                desired = log["desired_tcp_pose_base"][..., :3]
                actual = log["actual_tcp_pose_base"][..., :3]
                contact = log["physx_contact"]
            if desired.shape != (len(time), 2, 3) or actual.shape != desired.shape:
                raise ValueError(f"Invalid TCP pose dimensions: {path}")
            if not all(np.isfinite(a).all() for a in (time, desired, actual)) or np.any(
                np.diff(time) <= 0
            ):
                raise ValueError(f"Nonfinite or unordered TCP samples: {path}")
            if metadata["hand_order"] != ["left", "right"]:
                raise ValueError(f"Unsupported hand ordering: {path}")
            error_mm = (actual - desired) * 1000
            axis_rmse = np.sqrt(np.mean(error_mm**2, axis=0))
            step = metadata.get("experiment") == "fixed_joint_target_step"
            mode, episode = metadata["mode"], metadata["episode"]
            name = (
                f"fixed-target-step-response-episode-{episode}"
                if step
                else f"{mode}-episode-{episode}-tcp-position"
            )
            figure, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, layout="constrained")
            # Contact flags describe the preceding physics interval.
            starts = np.flatnonzero(np.diff(np.r_[False, contact].astype(int)) == 1)
            ends = np.flatnonzero(np.diff(np.r_[contact, False].astype(int)) == -1)
            for hand, side in enumerate(metadata["hand_order"]):
                for axis_index, coordinate in enumerate("XYZ"):
                    axis = axes[axis_index, hand]
                    axis.plot(
                        time,
                        actual[:, hand, axis_index],
                        color="#1764ab",
                        lw=1.4,
                        label="Actual TCP",
                    )
                    axis.plot(
                        time,
                        desired[:, hand, axis_index],
                        color="#d65f00",
                        lw=1.5,
                        ls="--",
                        label="Desired TCP",
                        drawstyle="steps-post" if step else "default",
                    )
                    for begin, end in zip(starts, ends, strict=True):
                        axis.axvspan(
                            time[max(0, begin - 1)], time[end], color="#b83d57", alpha=0.10
                        )
                    if step:
                        axis.axvline(metadata["step_time_s"], color="#555555", lw=0.9, ls=":")
                    axis.set_ylabel(f"{coordinate} position (m)")
                    axis.set_title(
                        f"{side.capitalize()} TCP · {coordinate}", loc="left", fontsize=11
                    )
                    axis.text(
                        0.99,
                        0.97,
                        f"RMSE {axis_rmse[hand, axis_index]:.1f} mm",
                        ha="right",
                        va="top",
                        transform=axis.transAxes,
                        fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
                    axis.grid(alpha=0.2)
                    axis.ticklabel_format(useOffset=False, axis="y")
                    if axis_index == 2:
                        axis.set_xlabel("Simulation time (s)")
            if step:
                title = (
                    f"Fixed joint-target step · Episode {episode} · "
                    f"frame {metadata['initial_episode_frame']} → {metadata['target_episode_frame']}\n"
                    f"Switch at {metadata['step_time_s']:g} s; zero target velocity; "
                    f"hold {metadata['hold_time_s']:g} s"
                )
            else:
                title = (
                    f"Episode {episode} · {'Joint PD' if mode == 'joint' else 'Differential IK + PD'}\n"
                    f"Desired vs actual TCP position · playback speed {metadata['speed']:g}×"
                )
            title += f" · {metadata['pose_frame']}"
            if not metadata.get("complete", False):
                title += " · INCOMPLETE RUN"
            if contact.any():
                title += "\nPink shading: PhysX contact (either arm)"
            figure.suptitle(title, fontsize=13)
            axes[0, 0].legend(loc="best", fontsize=9)
            figure.savefig(output / f"{name}.png", dpi=180)
            pdf.savefig(figure)
            plt.close(figure)
            summaries.append(
                {
                    "mode": mode,
                    "episode": episode,
                    "source": str(path.resolve()),
                    "physics_samples": len(time),
                    "plot": f"{name}.png",
                    "hand_order": metadata["hand_order"],
                    "axis_order": ["x", "y", "z"],
                    "axis_rmse_mm": axis_rmse.tolist(),
                    "position_rmse_mm": np.sqrt(
                        np.mean(np.sum(error_mm**2, axis=-1), axis=0)
                    ).tolist(),
                }
            )
    (output / "tcp-position-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def plot_collisions_and_errors(run, output):
    audits = sorted((run / "audit").glob("episode-*/audit.npz"))
    if not audits:
        return
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
            frames = run / mode / path.parent.name / "frames.npz"
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
    fig.savefig(output / "collision-comparison.png", dpi=160)
    tracking.suptitle("Measured vs desired TCP position at original dataset timestamps")
    tracking.savefig(output / "tracking-errors.png", dpi=160)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Plot directory (defaults to the recording directory)"
    )
    args = parser.parse_args()
    output = args.output or args.run
    output.mkdir(parents=True, exist_ok=True)
    summaries = plot_tcp_positions(args.run, output)
    # Step experiments use a synthetic clock, so source-episode audit plots are unrelated.
    if list(args.run.glob("*/episode-*/frames.npz")) or not summaries:
        plot_collisions_and_errors(args.run, output)
    if not summaries and not list((args.run / "audit").glob("episode-*/audit.npz")):
        parser.error("No TCP recordings or audit logs found")
    print(f"Plots saved in {output}")


if __name__ == "__main__":
    main()
