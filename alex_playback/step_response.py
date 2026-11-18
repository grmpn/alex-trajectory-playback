"""Fixed joint-target experiments using the same PhysX robot and PD controller."""

import math

import numpy as np
import torch

from .reporting import write_json


def step_schedule(delay, hold, dt):
    """Quantize durations upward; commands switch at the start of a physics interval."""
    if not all(np.isfinite(value) and value > 0 for value in (delay, hold, dt)):
        raise ValueError("Step durations and physics dt must be positive and finite")
    delay_steps = math.ceil(delay / dt)
    hold_steps = math.ceil(hold / dt)
    indices = np.arange(delay_steps + hold_steps + 1)
    return indices * dt, indices >= delay_steps, delay_steps * dt


def response_metrics(time, desired, actual, step_time):
    """Position errors in metres; settling requires a continuous final 0.5 s or longer."""
    error = actual[..., :3] - desired[..., :3]
    norm = np.linalg.norm(error, axis=-1)
    after = time >= step_time
    tail = time >= max(step_time, time[-1] - 0.5)
    settling = []
    for hand in range(2):
        outside = norm[after, hand] > 0.002
        last_outside = np.flatnonzero(outside)
        first = last_outside[-1] + 1 if len(last_outside) else 0
        times = time[after]
        settled = first < len(times) and times[-1] - times[first] >= 0.5 - 1e-9
        settling.append(float(times[first] - step_time) if settled else None)
    return {
        "hand_order": ["left", "right"],
        "final_mean_axis_error_m": error[tail].mean(axis=0).tolist(),
        "final_mean_position_error_m": norm[tail].mean(axis=0).tolist(),
        "peak_position_error_after_step_m": norm[after].max(axis=0).tolist(),
        "settling_time_to_2mm_s": settling,
        "settling_definition": "3D TCP error <= 2 mm through the end, for at least 0.5 s",
        "final_error_window_s": float(time[-1] - time[tail][0]),
    }


def run_step_response(simulation, controller, contacts, episode, reference_q, args, output, dt):
    model, robot = controller.model, controller.robot
    target_index = int(np.flatnonzero(episode.frame == args.step_target_frame)[0])
    configurations = np.array([reference_q[0], reference_q[target_index]])
    if np.any(configurations < model.model.lowerPositionLimit - 1e-6) or np.any(
        configurations > model.model.upperPositionLimit + 1e-6
    ):
        raise ValueError("Step endpoints must respect the URDF joint limits")
    targets = [controller.tensor(model.ordered(q, controller.names)) for q in configurations]
    desired_poses = np.array([model.tcp(q) for q in configurations])
    zero_velocity = torch.zeros_like(targets[0])
    time, at_target, step_time = step_schedule(args.step_delay, args.step_hold, dt)
    destination = output / "joint" / f"episode-{episode.index}"
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment": "fixed_joint_target_step",
        "mode": "joint",
        "episode": episode.index,
        "complete": False,
        "initial_episode_frame": int(episode.frame[0]),
        "target_episode_frame": int(episode.frame[target_index]),
        "initial_source_frame": int(episode.source_frame[0]),
        "target_source_frame": int(episode.source_frame[target_index]),
        "pose_frame": "PEDESTAL_LINK",
        "pose_order": "x,y,z,qx,qy,qz,qw",
        "hand_order": ["left", "right"],
        "joint_names": list(controller.names),
        "physics_dt_s": dt,
        "step_time_s": step_time,
        "hold_time_s": float(time[-1] - step_time),
        "settle_time_s": max(1, math.ceil(args.settle_time / dt)) * dt,
        "gravity_compensation": args.gravity_compensation,
        "kd_scale": args.kd_scale,
        "pd_gains_by_joint": controller.gains(),
        "actuators": {name: type(a).__name__ for name, a in robot.actuators.items()},
        "target_velocity": "zero for every joint throughout the experiment",
        "sample_convention": "state and command at interval start; command is applied over [t, t+dt)",
        "initial_nominal_collision_pairs": model.collisions(configurations[0]),
        "target_nominal_collision_pairs": model.collisions(configurations[1]),
    }
    write_json(destination / "metadata.json", metadata)
    robot.write_joint_position_to_sim_index(position=targets[0])
    robot.write_joint_velocity_to_sim_index(velocity=zero_velocity)
    robot.reset()
    for _ in range(max(1, math.ceil(args.settle_time / dt))):
        controller.write(targets[0], zero_velocity)
        simulation.step(render=False)
        robot.update(dt)
        contacts.update()
    controller.validate_frames()

    rows = []
    complete = False
    try:
        for i, target_selected in enumerate(at_target):
            if not simulation.is_running():
                raise KeyboardInterrupt("Simulation closed during step response")
            target = targets[int(target_selected)]
            actual_q = robot.data.joint_pos.torch[0].cpu().numpy().copy()
            if not np.isfinite(actual_q).all():
                raise RuntimeError("Nonfinite joint state in step response")
            rows.append(
                {
                    "simulation_time_s": time[i],
                    "desired_joint_position": target[0].cpu().numpy().copy(),
                    "desired_joint_velocity": np.zeros(len(controller.names)),
                    "actual_joint_position": actual_q,
                    "actual_joint_velocity": robot.data.joint_vel.torch[0].cpu().numpy().copy(),
                    "desired_tcp_pose_base": desired_poses[int(target_selected)],
                    "actual_tcp_pose_base": controller.poses()[0].cpu().numpy().copy(),
                    "physx_contact": bool(contacts.current),
                    "physx_penetration": any(
                        v["min_separation_m"] < -1e-6 for v in contacts.current.values()
                    ),
                }
            )
            if i == len(time) - 1:
                break
            controller.write(target, zero_velocity)
            simulation.step(render=False)
            robot.update(dt)
            contacts.update()
        complete = True
    finally:
        metadata["complete"] = complete
        write_json(destination / "metadata.json", metadata)
        if rows:
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            np.savez_compressed(
                destination / "steps.npz", joint_names=np.array(controller.names), **arrays
            )
    summary = {
        "episode": episode.index,
        "complete": complete,
        "physics_samples": len(rows),
        "step_time_s": step_time,
        "tcp_step_m": (desired_poses[1, :, :3] - desired_poses[0, :, :3]).tolist(),
        "physx_contact_samples": int(arrays["physx_contact"].sum()),
        "physx_penetration_samples": int(arrays["physx_penetration"].sum()),
        **response_metrics(
            time, arrays["desired_tcp_pose_base"], arrays["actual_tcp_pose_base"], step_time
        ),
    }
    write_json(destination / "summary.json", summary)
    print(
        f"Step episode {episode.index}: final TCP error {summary['final_mean_position_error_m']} m",
        flush=True,
    )
    return summary
