#!/usr/bin/env python3
"""Play all three dataset trajectories on the fixed-base Purdue Alex."""

import argparse
from datetime import datetime, timezone
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


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("joint", "ik", "both", "audit"), default="joint")
    parser.add_argument("--dataset", type=Path, default=root / "datasets/iphone-robot")
    parser.add_argument("--reports", type=Path, default=root / "datasets/iphone-export-report")
    parser.add_argument("--assets", type=Path, default=root / "purdue-alex-assets")
    parser.add_argument(
        "--output", type=Path, help="New output directory (existing paths are never overwritten)"
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", help="Episode indices; defaults to all three"
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
    parser.add_argument("--headless", action="store_true")
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
    args = parser.parse_args()
    if not 0 <= args.settle_time < float("inf"):
        parser.error("--settle-time must be nonnegative and finite")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.realtime is None:
        args.realtime = not args.headless
    output = (
        args.output or root / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    ).resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    dataset = Dataset(args.dataset.resolve(), args.reports.resolve(), args.assets.resolve())
    ids = args.episodes if args.episodes is not None else dataset.episode_ids
    if len(ids) != len(set(ids)):
        parser.error("Episode indices must be unique")
    episodes = [dataset.episode(i, args.max_frames) for i in ids]
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
