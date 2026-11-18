#!/usr/bin/env python3
"""Play selected LeRobot v3 trajectories on the fixed-base Purdue Alex."""

import argparse
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from alex_playback.dataset import Dataset, timing
from alex_playback.model import RobotModel
from alex_playback.reporting import audit_episode, write_json


def positive(value):
    value = float(value)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("joint", "ik", "ee", "both", "audit"), default="joint")
    parser.add_argument("--dataset", type=Path, default=root / "datasets/iphone-robot")
    parser.add_argument(
        "--reports",
        type=Path,
        help="Optional collision diagnostics; auto-selected for the bundled dataset",
    )
    parser.add_argument(
        "--joint-key", default="action", help="LeRobot feature containing named Alex joint actions"
    )
    parser.add_argument(
        "--ee-key", default="action.ee", help="LeRobot feature containing base-frame TCP poses"
    )
    parser.add_argument("--assets", type=Path, default=root / "purdue-alex-assets")
    parser.add_argument(
        "--output", type=Path, help="New output directory (existing paths are never overwritten)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Episode indices; defaults to all available episodes",
    )
    parser.add_argument(
        "--speed",
        type=positive,
        default=1.0,
        help="Dataset time / simulation time; 0.5 is half speed",
    )
    parser.add_argument(
        "--physics-dt", type=positive, default=1 / 240, help="Maximum physics step in seconds"
    )
    parser.add_argument(
        "--settle-time", type=float, default=1.0, help="Initial PD settling time before recording"
    )
    parser.add_argument(
        "--step-target-frame",
        type=int,
        help="Run a fixed joint-target step from --start-frame to this episode frame (headless joint mode)",
    )
    parser.add_argument("--step-delay", type=positive, default=1.0, help="Seconds before the step")
    parser.add_argument(
        "--step-hold", type=positive, default=8.0, help="Seconds holding the step target"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--renderer", choices=("minimal", "raytraced", "pathtraced"), default="raytraced"
    )
    parser.add_argument("--ghost-opacity", type=float, default=0.25)
    parser.add_argument("--render-fps", type=positive, default=30.0)
    parser.add_argument(
        "--exit-after",
        action="store_true",
        help="Close the GUI when the playlist finishes instead of leaving it open",
    )
    parser.add_argument(
        "--capture-preview", action="store_true", help="Save a viewport image after each trajectory"
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First episode frame to play")
    parser.add_argument("--end-frame", type=int, help="Exclusive last episode frame to play")
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pace against wall time (default: GUI yes, headless no)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gravity-compensation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--kd-scale",
        type=positive,
        default=1.0,
        help="Multiply all upstream PD damping gains; position gains remain unchanged",
    )
    parser.add_argument(
        "--record-control",
        action="store_true",
        help="Record pre-step joint commands, measured states, and actuator torques during trajectory playback",
    )
    parser.add_argument("--ik-damping", type=positive, default=0.05)
    parser.add_argument(
        "--ik-max-joint-step",
        type=positive,
        default=0.25,
        help="Maximum IK target offset from current joints, radians",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit each episode for a smoke run; outputs are marked as partial episodes",
    )
    args = parser.parse_args(argv)
    if args.mode == "ee":
        args.mode = "ik"
    if not 0 < args.ghost_opacity < 1:
        parser.error("--ghost-opacity must be between 0 and 1")
    if args.start_frame < 0 or (args.end_frame is not None and args.end_frame <= args.start_frame):
        parser.error("Frame selection requires 0 <= --start-frame < --end-frame")
    if not 0 <= args.settle_time < float("inf"):
        parser.error("--settle-time must be nonnegative and finite")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.record_control and (args.mode == "audit" or args.step_target_frame is not None):
        parser.error("--record-control requires trajectory playback")
    if args.step_target_frame is not None:
        if args.mode != "joint" or not args.headless or args.speed != 1:
            parser.error("Step response requires --mode joint --headless and --speed 1")
        if args.step_target_frame <= args.start_frame:
            parser.error("--step-target-frame must follow --start-frame")
    if args.realtime is None:
        args.realtime = not args.headless
    output = (
        args.output or root / "outputs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    ).resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    if (
        args.reports is None
        and args.dataset.resolve() == (root / "datasets/iphone-robot").resolve()
    ):
        args.reports = root / "datasets/iphone-export-report"
    dataset = Dataset(
        args.dataset.resolve(), args.reports, args.assets.resolve(), args.joint_key, args.ee_key
    )
    ids = args.episodes if args.episodes is not None else dataset.episode_ids
    if len(ids) != len(set(ids)):
        parser.error("Episode indices must be unique")
    episodes = [dataset.episode(i, args.max_frames, args.start_frame, args.end_frame) for i in ids]
    if args.step_target_frame is not None and any(
        args.step_target_frame not in episode.frame for episode in episodes
    ):
        parser.error("--step-target-frame must be inside every selected episode/frame interval")
    if args.mode in ("joint", "both") and not dataset.joint_key:
        parser.error("This dataset has no joint actions; use --mode ee")
    model = RobotModel(
        dataset.urdf,
        dataset.assets,
        dataset.embodiment["base_frame"],
        dataset.embodiment["tool_frames"],
    )
    steps, dt = timing(dataset.fps, args.speed, args.physics_dt)
    output.mkdir(parents=True)
    write_json(
        output / "run.json",
        {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "dataset": str(dataset.root),
            "embodiment": dataset.embodiment,
            "versions": {
                name: version(name)
                for name in ("isaaclab", "isaacsim", "torch", "pin", "coal", "pyarrow")
            },
            "steps_per_frame": steps,
            "physics_dt_s": dt,
        },
    )
    audits = {}
    for episode in episodes:
        audits[episode.index] = audit_episode(
            dataset, model, episode, output / "audit" / f"episode-{episode.index}"
        )
        print(
            f"episode {episode.index} audit: {audits[episode.index][0]['nominal_mesh_vs_dataset']}",
            flush=True,
        )
    write_json(
        output / "summary.json",
        {"audit": [entry[0] for entry in audits.values()], "playback": []},
    )
    if args.mode != "audit":
        from alex_playback.simulation import run

        run(dataset, model, episodes, audits, args, output, steps, dt)
    else:
        print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
