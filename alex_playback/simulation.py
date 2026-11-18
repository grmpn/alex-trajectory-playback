"""PhysX playback with the asset's explicit PD actuators and differential IK."""

import argparse
import math
import time

import numpy as np
import torch

from .dataset import interpolate_poses
from .importing import fix_world_joint, prepare_urdf
from .model import pose_error
from .reporting import Recorder, write_json


class Contacts:
    """Read pairwise GPU PhysX contacts; Kit callbacks omit them in this setup."""

    def __init__(self, stage, root, model):
        from isaaclab.app.settings_manager import get_settings_manager
        from pxr import PhysxSchema, Usd, UsdPhysics

        self.model = model
        self.current = {}
        self.body_paths = []
        get_settings_manager().set("/physics/disableContactProcessing", False)
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
                self.body_paths.append(str(prim.GetPath()))

    def initialize(self):
        from isaaclab_physx.physics import PhysxManager

        self.view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            self.body_paths,
            filter_patterns=[self.body_paths] * len(self.body_paths),
            max_contact_data_count=8192,
        )
        if (self.view.sensor_count, self.view.filter_count) != (len(self.body_paths),) * 2:
            raise ValueError("PhysX contact view does not cover every robot link pair")
        if list(self.view.sensor_paths) != self.body_paths:
            raise ValueError("Unexpected PhysX contact sensor ordering")
        if any(list(paths) != self.body_paths for paths in self.view.filter_paths):
            raise ValueError("Unexpected PhysX contact filter ordering")
        self.names = [path.rsplit("/", 1)[-1] for path in self.body_paths]

    def update(self):
        # dt=1 returns normal impulses in N s, without dividing by the physics step.
        impulses, _, _, separation, counts, starts = self.view.get_normal_contact_data(1.0)
        counts, starts = counts.numpy(), starts.numpy()
        if int(counts.sum()) >= self.view.max_contact_data_count:
            raise RuntimeError("PhysX contact buffer capacity exhausted")
        self.current = {}
        pairs = np.argwhere(np.triu(counts, k=1) > 0)
        if not len(pairs):
            return
        impulses, separation = impulses.numpy().ravel(), separation.numpy().ravel()
        for first, second in pairs:
            pair = tuple(sorted((self.names[first], self.names[second])))
            if self.model.excluded(*pair):
                continue
            start, count = int(starts[first, second]), int(counts[first, second])
            if start < 0 or start + count > len(impulses):
                raise RuntimeError("PhysX contact buffer returned an invalid range")
            self.current[pair] = {
                "min_separation_m": float(separation[start : start + count].min()),
                "normal_impulse_ns": float(impulses[start : start + count].sum()),
            }

    @staticmethod
    def merge(destination, source):
        for pair, values in source.items():
            if pair not in destination:
                destination[pair] = dict(values)
            else:
                destination[pair]["min_separation_m"] = min(
                    destination[pair]["min_separation_m"], values["min_separation_m"]
                )
                destination[pair]["normal_impulse_ns"] += values["normal_impulse_ns"]


class Controller:
    def __init__(self, robot, model, dataset, args):
        from isaaclab.controllers import (
            DifferentialIKController,
            DifferentialIKControllerCfg,
        )

        self.robot, self.model, self.args = robot, model, args
        self.names = robot.data.joint_names
        self.ids = {name: i for i, name in enumerate(self.names)}
        if set(self.names) != set(model.names):
            raise ValueError(f"PhysX/URDF joints disagree: {set(self.names) ^ set(model.names)}")
        if robot.num_base_dofs != 0:
            raise ValueError("Playback requires the dataset's fixed-base embodiment")
        if not all(
            type(actuator).__name__ == "IdealPDActuator" for actuator in robot.actuators.values()
        ):
            raise ValueError(
                "The supplied explicit IdealPDActuator configuration was not preserved"
            )
        self.controlled = [i for name, i in self.ids.items() if name not in model.mimics]
        self.arm_ids = [
            [self.ids[n] for n in names]
            for names in dataset.contract["robot_settings"]["arm_joints"]
        ]
        self.body_ids = []
        self.offsets = []
        self.ik = []
        for tool in model.tool_names:
            anchor, offset = model.tcp_anchor(tool)
            self.body_ids.append(robot.data.body_names.index(anchor))
            self.offsets.append(
                torch.tensor(offset, dtype=torch.float32, device=robot.device).unsqueeze(0)
            )
            self.ik.append(
                DifferentialIKController(
                    DifferentialIKControllerCfg(
                        command_type="pose",
                        use_relative_mode=False,
                        ik_method="dls",
                        ik_params={"lambda_val": args.ik_damping},
                    ),
                    num_envs=1,
                    device=robot.device,
                )
            )

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.robot.device).reshape(1, -1)

    def poses(self):
        from isaaclab.utils.math import (
            combine_frame_transforms,
            subtract_frame_transforms,
        )

        root = self.robot.data.root_link_pose_w.torch
        result = []
        for body_id, offset in zip(self.body_ids, self.offsets, strict=True):
            body = self.robot.data.body_link_pose_w.torch[:, body_id]
            pos, quat = combine_frame_transforms(
                body[:, :3], body[:, 3:7], offset[:, :3], offset[:, 3:7]
            )
            pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], pos, quat)
            result.append(torch.cat((pos, quat), dim=-1))
        return torch.stack(result, dim=1)

    def jacobian(self, hand):
        from isaaclab.utils.math import (
            matrix_from_quat,
            quat_apply,
            quat_inv,
            skew_symmetric_matrix,
        )

        body_id = self.body_ids[hand]
        # PhysX omits the fixed root row; body_link_jacobian_w already removes the COM offset.
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, body_id - 1, :, :][
            :, :, self.arm_ids[hand]
        ].clone()
        body_quat = self.robot.data.body_link_pose_w.torch[:, body_id, 3:7]
        offset_world = quat_apply(body_quat, self.offsets[hand][:, :3])
        jacobian[:, :3] -= torch.bmm(skew_symmetric_matrix(offset_world), jacobian[:, 3:])
        base_from_world = matrix_from_quat(quat_inv(self.robot.data.root_link_pose_w.torch[:, 3:7]))
        jacobian[:, :3] = torch.bmm(base_from_world, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(base_from_world, jacobian[:, 3:])
        return jacobian

    def validate_frames(self):
        import pinocchio as pin

        q = self.model.configuration(self.names, self.robot.data.joint_pos.torch[0].cpu().numpy())
        error = pose_error(self.poses()[0].cpu().numpy(), self.model.tcp(q))
        if np.max(error[0]) > 1e-4 or np.max(error[1]) > 1e-3:
            raise ValueError(f"Simulated TCP transforms do not match URDF FK: {error}")
        for hand, frame in enumerate(self.model.tools):
            expected = pin.computeFrameJacobian(
                self.model.model, self.model.data, q, frame, pin.LOCAL_WORLD_ALIGNED
            )
            columns = [self.model.indices[self.names[i]] for i in self.arm_ids[hand]]
            measured = self.jacobian(hand)[0].cpu().numpy()
            if not np.allclose(measured, expected[:, columns], atol=2e-4, rtol=2e-4):
                raise ValueError(
                    f"TCP Jacobian frame/offset mismatch for hand {hand}: {np.max(abs(measured - expected[:, columns]))}"
                )

    def target(self, reference_q, desired_tcp, mode):
        target = self.tensor(self.model.ordered(reference_q, self.names))
        if mode == "ik":
            poses = self.poses()
            limits = self.robot.data.joint_pos_limits.torch
            for hand, controller in enumerate(self.ik):
                ids = self.arm_ids[hand]
                current = self.robot.data.joint_pos.torch[:, ids]
                controller.set_command(self.tensor(desired_tcp[hand]))
                solution = controller.compute(
                    poses[:, hand, :3], poses[:, hand, 3:7], self.jacobian(hand), current
                )
                delta = torch.clamp(
                    solution - current, -self.args.ik_max_joint_step, self.args.ik_max_joint_step
                )
                target[:, ids] = torch.clamp(current + delta, limits[:, ids, 0], limits[:, ids, 1])
        return target

    def write(self, target, velocity):
        self.robot.set_joint_position_target_index(target=target)
        self.robot.set_joint_velocity_target_index(target=velocity)
        if self.args.gravity_compensation:
            gravity = self.robot.data.gravity_compensation_forces.torch[
                :, self.robot.num_base_dofs :
            ]
            self.robot.set_joint_effort_target_index(
                target=gravity[:, self.controlled], joint_ids=self.controlled
            )
        self.robot.write_data_to_sim()

    def gains(self):
        """Resolved gains of active joints; passive mimic followers have no PD actuator."""
        return {
            name: {"kp_nm_per_rad": kp, "kd_nm_s_per_rad": kd}
            for actuator in self.robot.actuators.values()
            for name, kp, kd in zip(
                actuator.joint_names,
                actuator.stiffness[0].cpu().tolist(),
                actuator.damping[0].cpu().tolist(),
                strict=True,
            )
        }

    def control_signals(self):
        """Snapshot after write_data_to_sim, before stepping: PD inputs and outputs align."""
        actuators = self.robot.actuators
        signals = {
            "desired_joint_position": actuators.target_command.position.torch,
            "actual_joint_position": self.robot.data.joint_pos.torch,
            "desired_joint_velocity": actuators.target_command.velocity.torch,
            "actual_joint_velocity": self.robot.data.joint_vel.torch,
            "feedforward_effort": actuators.target_command.effort.torch,
            "computed_effort": actuators.computed_effort.torch,
            "applied_effort": actuators.applied_effort.torch,
            "submitted_effort": actuators.output_command.effort.torch,
        }
        values = torch.stack([value[0] for value in signals.values()]).cpu().numpy()
        return dict(zip(signals, values, strict=True))


def run(dataset, model, episodes, audits, args, output, steps_per_frame, dt):
    import isaaclab.sim as sim_utils
    from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg
    from ihmc_alex_isaaclab.robots.purdue_frames import author_alex_purdue_frames
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.assets import Articulation
    from isaaclab_physx.physics import PhysxCfg

    from .viewer import Viewer, configure_renderer

    if not args.headless:
        configure_renderer(args.renderer)
    launcher_parser = argparse.ArgumentParser()
    add_launcher_args(launcher_parser)
    launcher_args = launcher_parser.parse_args(
        ["--device", args.device, "--viz", "none" if args.headless else "kit"]
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt,
        render_interval=max(1, math.ceil(1 / (args.render_fps * dt))),
        device=args.device,
        gravity=(0, 0, -9.81),
        use_newton_actuators=False,
        physics=PhysxCfg(
            solver_type=1,
            solve_articulation_contact_last=True,
            enable_enhanced_determinism=True,
            enable_external_forces_every_iteration=True,
            min_position_iteration_count=8,
            min_velocity_iteration_count=4,
        ),
    )
    results = []
    with launch_simulation(sim_cfg, launcher_args):
        simulation = sim_utils.SimulationContext(sim_cfg)
        simulation_urdf, visual_repairs = prepare_urdf(dataset.urdf, output / "import")
        cfg = make_alex_purdue_cfg(asset_path=simulation_urdf, fix_base=True).replace(
            prim_path="/World/Robot"
        )
        for actuator in cfg.actuators.values():
            actuator.damping = (
                {pattern: gain * args.kd_scale for pattern, gain in actuator.damping.items()}
                if isinstance(actuator.damping, dict)
                else actuator.damping * args.kd_scale
            )
        cfg.spawn.usd_dir = str((output / "usd").resolve())
        # Select the backend layer explicitly instead of the neutral default.
        cfg.spawn.variants = {"Physics": "physx"}
        robot = Articulation(cfg)
        stage = sim_utils.get_current_stage()
        fix_world_joint(stage, "/World/Robot", dataset.embodiment["base_frame"])
        author_alex_purdue_frames(stage, "/World/Robot", asset_path=dataset.urdf)
        contacts = Contacts(stage, "/World/Robot", model)
        viewer = None
        if not args.headless:
            light = sim_utils.DomeLightCfg(intensity=2500)
            light.func("/World/Light", light)
            floor = sim_utils.CuboidCfg(
                size=(8, 8, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            )
            floor.func("/World/Floor", floor, translation=(0, 0, -1.0))
            viewer = Viewer(stage, model, args.ghost_opacity, args.speed, args.renderer)
            viewer.configure_viewport()
        simulation.reset()
        contacts.initialize()
        controller = Controller(robot, model, dataset, args)
        if viewer:
            viewer.configure_viewport()
            simulation.set_camera_view((1.6, 1.6, 1.0), (0.15, 0, 0.05))
            viewer.set_base_pose(robot.data.root_link_pose_w.torch[0].cpu().numpy())
        if args.step_target_frame is not None:
            from .step_response import run_step_response

            for episode in episodes:
                results.append(
                    run_step_response(
                        simulation,
                        controller,
                        contacts,
                        episode,
                        audits[episode.index][1],
                        args,
                        output,
                        dt,
                    )
                )
                write_json(output / "summary.json", {"step_response": results})
            print(f"Results: {output}", flush=True)
            return
        modes = ("joint", "ik") if args.mode == "both" else (args.mode,)
        for mode in modes:
            for episode in episodes:
                _, reference_q, nominal_pairs = audits[episode.index]
                metadata = {
                    "mode": mode,
                    "episode": episode.index,
                    "speed": args.speed,
                    "fps": dataset.fps,
                    "physics_dt_s": dt,
                    "steps_per_frame": steps_per_frame,
                    "settle_time_s": args.settle_time,
                    "gravity_compensation": args.gravity_compensation,
                    "kd_scale": args.kd_scale,
                    "pd_gains_by_joint": controller.gains(),
                    "asset_revision": dataset.embodiment["asset_revision"],
                    "urdf": dataset.embodiment["urdf"],
                    "simulation_urdf": str(simulation_urdf),
                    "visual_repairs": visual_repairs,
                    "actuators": {name: type(a).__name__ for name, a in robot.actuators.items()},
                    "ik_damping": args.ik_damping,
                    "ik_max_joint_step_rad": args.ik_max_joint_step,
                    "full_episode": len(episode) == episode.full_length,
                    "first_episode_frame": int(episode.frame[0]),
                    "reference_joint_position_kind": "dataset actions"
                    if episode.has_joint_actions
                    else "IK joint targets",
                    "ee_target_kind": "dataset actions"
                    if episode.has_ee_actions
                    else "FK of dataset joints",
                    "has_collision_labels": episode.has_collision_labels,
                    "has_valid_mask": episode.has_valid_mask,
                    "renderer": args.renderer if viewer else None,
                }
                recorder = Recorder(
                    output / mode / f"episode-{episode.index}", episode, controller.names, metadata
                )
                if args.record_control:
                    metadata["control_trace"] = {
                        "file": "control.npz",
                        "sample_convention": "after actuator computation, before physics; commands apply over [t, t+dt)",
                        "time_origin": "trajectory start; negative times are initial settling",
                        "position_target_timing": "target calculated for trajectory time t+dt; settling holds the initial reference",
                        "units": "positions rad, velocities rad/s, efforts N m, time s",
                        "applied_effort_kind": "clipped actuator command, not measured net joint torque",
                        "active_joint_names": [controller.names[j] for j in controller.controlled],
                        "effort_limit_nm_by_joint": {
                            name: limit
                            for actuator in robot.actuators.values()
                            for name, limit in zip(
                                actuator.joint_names,
                                actuator.actuator_effort_limit[0].cpu().tolist(),
                                strict=True,
                            )
                        },
                    }
                complete = False
                try:
                    if viewer:
                        viewer.set_episode(mode, episode)
                    initial = controller.tensor(model.ordered(reference_q[0], controller.names))
                    zero_velocity = torch.zeros_like(initial)
                    robot.write_joint_position_to_sim_index(position=initial)
                    robot.write_joint_velocity_to_sim_index(velocity=zero_velocity)
                    robot.reset()
                    for ik in controller.ik:
                        ik.reset()
                    target = initial
                    settle_steps = max(1, math.ceil(args.settle_time / dt))
                    for settle_step in range(settle_steps):
                        if not episode.has_joint_actions:
                            target = controller.target(reference_q[0], episode.ee[0, :, :7], mode)
                        controller.write(target, zero_velocity)
                        if args.record_control:
                            recorder.control(
                                (settle_step - settle_steps) * dt,
                                "settling",
                                controller.control_signals(),
                            )
                        simulation.step(render=False)
                        robot.update(dt)
                        contacts.update()
                        if viewer and settle_step % sim_cfg.render_interval == 0:
                            ghost_q = (
                                reference_q[0]
                                if episode.has_joint_actions
                                else model.configuration(controller.names, target[0].cpu().numpy())
                            )
                            viewer.set_reference(ghost_q)
                            simulation.render()
                    controller.validate_frames()
                    wall_start = time.monotonic()
                    physics_steps = 0
                    for i in range(len(episode)):
                        if not simulation.is_running():
                            raise KeyboardInterrupt("Simulation window closed")
                        frame_contacts = {}
                        previous = max(i - 1, 0)
                        fractions = np.arange(1, steps_per_frame + 1) / steps_per_frame
                        tcp_targets = interpolate_poses(
                            episode.ee[previous, :, :7], episode.ee[i, :, :7], fractions
                        )
                        if i == 0:
                            Contacts.merge(frame_contacts, contacts.current)
                            recorder.step(
                                int(episode.frame[0]),
                                float(episode.time[0]),
                                0,
                                target[0].cpu().numpy(),
                                robot.data.joint_pos.torch[0].cpu().numpy(),
                                model.tcp(reference_q[0])
                                if mode == "joint"
                                else episode.ee[0, :, :7],
                                controller.poses()[0].cpu().numpy(),
                                bool(contacts.current),
                                any(
                                    v["min_separation_m"] < -1e-6 for v in contacts.current.values()
                                ),
                                reference_q=model.ordered(reference_q[0], controller.names)
                                if episode.has_joint_actions
                                else target[0].cpu().numpy(),
                            )
                        else:
                            velocity = controller.tensor(
                                model.ordered(
                                    (reference_q[i] - reference_q[previous])
                                    * dataset.fps
                                    * args.speed,
                                    controller.names,
                                )
                            )
                            if mode == "ik":
                                velocity[:, controller.arm_ids[0] + controller.arm_ids[1]] = 0
                            for substep, alpha in enumerate(fractions):
                                if viewer:
                                    wall_start += viewer.wait_if_paused(simulation)
                                if not simulation.is_running():
                                    raise KeyboardInterrupt("Simulation window closed")
                                reference = (1 - alpha) * reference_q[
                                    previous
                                ] + alpha * reference_q[i]
                                desired_tcp = (
                                    model.tcp(reference)
                                    if mode == "joint"
                                    else tcp_targets[substep]
                                )
                                target = controller.target(reference, desired_tcp, mode)
                                controller.write(target, velocity)
                                if args.record_control:
                                    recorder.control(
                                        physics_steps * dt,
                                        "trajectory",
                                        controller.control_signals(),
                                    )
                                simulation.step(render=False)
                                robot.update(dt)
                                contacts.update()
                                physics_steps += 1
                                Contacts.merge(frame_contacts, contacts.current)
                                actual = robot.data.joint_pos.torch[0].cpu().numpy()
                                if not np.isfinite(actual).all():
                                    raise RuntimeError(
                                        "Simulation produced a nonfinite joint state"
                                    )
                                recorder.step(
                                    int(episode.frame[i]),
                                    float(
                                        (1 - alpha) * episode.time[previous]
                                        + alpha * episode.time[i]
                                    ),
                                    physics_steps * dt,
                                    target[0].cpu().numpy(),
                                    actual,
                                    desired_tcp,
                                    controller.poses()[0].cpu().numpy(),
                                    bool(contacts.current),
                                    any(
                                        v["min_separation_m"] < -1e-6
                                        for v in contacts.current.values()
                                    ),
                                    reference_q=model.ordered(reference, controller.names)
                                    if episode.has_joint_actions
                                    else target[0].cpu().numpy(),
                                )
                                if (
                                    viewer
                                    and physics_steps % sim_cfg.render_interval == 0
                                    and substep < steps_per_frame - 1
                                ):
                                    ghost_q = (
                                        reference
                                        if episode.has_joint_actions
                                        else model.configuration(
                                            controller.names, target[0].cpu().numpy()
                                        )
                                    )
                                    viewer.set_reference(ghost_q)
                                    simulation.render()
                                if args.realtime:
                                    time.sleep(
                                        max(0, wall_start + physics_steps * dt - time.monotonic())
                                    )
                        q_actual = model.configuration(
                            controller.names, robot.data.joint_pos.torch[0].cpu().numpy()
                        )
                        actual_pairs = model.collisions(q_actual)
                        recorder.frame(i, nominal_pairs[i], actual_pairs, frame_contacts)
                        if viewer:
                            ghost_q = (
                                reference_q[i]
                                if episode.has_joint_actions
                                else model.configuration(controller.names, target[0].cpu().numpy())
                            )
                            viewer.set_reference(ghost_q)
                            viewer.highlight(nominal_pairs[i], actual_pairs, frame_contacts)
                            viewer.set_frame(episode, i)
                            simulation.render()
                        if i % 100 == 0 or i == len(episode) - 1:
                            print(
                                f"{mode} episode {episode.index}: {i + 1}/{len(episode)} frames",
                                flush=True,
                            )
                    complete = True
                    if viewer and args.capture_preview:
                        viewer.capture(simulation, recorder.destination / "preview.png")
                finally:
                    summary = recorder.save(complete)
                    results.append({"mode": mode, **summary})
                    # Kit may terminate the process when its launcher closes.
                    write_json(
                        output / "summary.json",
                        {"audit": [entry[0] for entry in audits.values()], "playback": results},
                    )
                print(
                    f"{mode} episode {episode.index}: TCP RMSE {summary['tcp_position_rmse_m']} m",
                    flush=True,
                )
        print(f"Results: {output}", flush=True)
        if viewer:
            viewer.finish(simulation, keep_open=not args.exit_after)
