# Alex trajectory playback

Playback of the three retargeted iPhone trajectories on fixed-base Purdue Alex
with dual SAKE EZGrippers, using Isaac Lab 3.0.0 and Isaac Sim 6.2.0.
The Lab checkout is release `v3.0.0`; its core Python distribution identifies
itself as `28.0.0`, which is the value recorded in `run.json`.

## Setup and commands

Use the existing Isaac Lab environment. From this repository:

```bash
source ~/IsaacLab/.venv/bin/activate
git submodule update --init --recursive
uv pip install --python "$VIRTUAL_ENV/bin/python" -r requirements.txt
export OMNI_KIT_ACCEPT_EULA=YES

# Joint trajectory playback, all three episodes, GUI at half speed.
python playback.py --mode joint --speed 0.5

# Closed-loop DifferentialIKController playback of action.ee.
python playback.py --mode ik --speed 0.5

# Execute both modes and record a full comparison without a GUI.
python playback.py --mode both --headless --output outputs/comparison
python analyze.py outputs/comparison

# Recheck nominal URDF collisions without launching Isaac Sim.
python playback.py --mode audit
```

Use `--headless` on this machine for the verified simulation path. The GUI smoke
check remained in renderer startup for eight minutes and was stopped; GUI
rendering has not been validated.

The local dataset lives in `datasets/iphone-robot`; its source-frame collision
diagnostics live in `datasets/iphone-export-report`. These inputs and generated
outputs are Git-ignored. `--dataset`, `--reports`, and `--assets` override their
locations. `--episodes 0 2` selects episodes; `--max-frames 10` runs a short check.
Output directories must be new so previous recordings cannot be overwritten.

`--speed` scales dataset time relative to simulated time. It does not skip frames
or merely change GUI delays. The integrator uses enough substeps to retain every
original 30 Hz sample while respecting `--physics-dt` (default maximum 1/240 s).
GUI playback is paced to wall time by default; use `--no-realtime` to run as fast
as possible. Headless playback runs without wall-time pacing by default. Actual
wall-time playback can be slower than requested if the machine cannot keep up.

## Robot and control

- The dataset pins asset revision `9ac4bd45c779d3f73d61f5b144cd1bb76dc8607b` and
  `assets/robots/alex_purdue/urdf/baseline/alex_purdue_full_convex.urdf`. The loader
  checks that revision and rejects modified source assets/configurations.
- Both modes use `make_alex_purdue_cfg` and its three explicit `IdealPDActuator`
  groups, with the supplied gains, armatures, and torque/velocity limits.
  Isaac Lab's automatic native-actuator replacement is disabled explicitly.
- Normalized openings use the upstream `alex_purdue_ezgripper_targets` helper.
  Both leader knuckles receive commands; the URDF mimic followers remain passive.
- Joint mode interpolates joint targets and supplies their time-scaled velocity
  targets. EE mode interpolates position and shortest-arc quaternion orientation,
  then runs one damped-least-squares `DifferentialIKController` per arm at every
  physics step. Neck and gripper commands still follow the dataset. IK uses actual
  simulated poses, transforms the PhysX Jacobian to the base frame, and includes
  the fixed TCP offset. Its joint targets are limited to the URDF position bounds
  and `--ik-max-joint-step` radians away from the current state.
- Gravity compensation is enabled, following the upstream asset readiness
  workflow. `--no-gravity-compensation` disables this feedforward term.
- Each episode resets once to its first joint configuration and settles under PD
  control for `--settle-time` seconds (default 1, minimum one physics step).
  Subsequent motion is simulated;
  joints are not teleported to the trajectory samples.
- All poses use **PEDESTAL_LINK**, metres, and **xyzw** quaternions. Hands are
  ordered left, right. Initialization checks the simulated TCP transforms and
  offset Jacobians against independent URDF kinematics.

For Sim 6.2, a local USD override attaches the root joint to the world and scopes
the articulation correctly. The newer mesh importer rejects 16 unused vertices
in the torso's visual OBJ. A generated copy removes only those unused positions
and remaps face indices, retaining normals, UVs, materials, and all face geometry.
The original URDF, collision meshes, inertias, and actuator settings stay intact.
Generated import files and the repair record are included in each run directory.

## Recordings

Each run contains `run.json`, `summary.json`, an `audit/` directory, and one folder
per playback mode and episode. Each episode folder contains:

| File | Contents |
| --- | --- |
| `steps.npz` | Every physics sample: desired/actual joint positions, desired/actual TCP poses, dataset/simulation times, frame indices, contact flags |
| `frames.npz` | The same measurements at original dataset frame boundaries, plus source video indices, action-valid masks, and collision labels |
| `frames.csv` | Frame/source-frame indices, collision flags, mask status, and TCP position errors |
| `collision_pairs.json` | Nominal and measured mesh pairs, dataset pairs, and PhysX contacts with minimum separation and accumulated normal impulse |
| `metadata.json` | Timing, actuator types, asset identity, import repairs, pose conventions, and completion status |
| `summary.json` | Joint/TCP tracking errors, collision confusion matrices, and inclusive source-frame collision ranges |

Arrays `desired_joint_position` and `actual_joint_position` use the explicit
`joint_names` array, including measured mimic followers. Poses have shape
`(samples, 2, 7)`. In joint mode desired TCPs are FK of the commanded joint
configuration. In IK mode they are the interpolated `action.ee` targets.
Settling is excluded from the recorded trajectory clock; frame zero records its
final state. The last original dataset timestamp is retained exactly.

```python
import numpy as np

log = np.load("outputs/comparison/joint/episode-0/frames.npz")
q_desired = log["desired_joint_position"]
q_actual = log["actual_joint_position"]
tcp_desired = log["desired_tcp_pose_base"]
tcp_actual = log["actual_tcp_pose_base"]
```

## Collision interpretation

`quality.action_valid` is a composite mask: tracking, observations, IK, joint
limits, and collisions can all invalidate actions. An all-false row is **not**
necessarily a collision. The collision reference comes from `collision_free` in
each `benchmark_frame_metrics.npz`, joined using **source_frame_index**, and is
cross-checked against the pair lists in `score.json`.

The independent nominal audit uses Pinocchio/Coal on the original collision
meshes, excluding same rigid assemblies, adjacent assemblies, and the upstream
three documented convex-mesh pairs. It reproduces all 173 dataset collision
frames exactly: 24/269 in episode 0, 18/461 in episode 1, and 131/339 in episode 2.

| Episode | Collision source video frames (inclusive) | Fully invalid frames without collision |
| --- | --- | --- |
| 0 | 186–190, 192–202, 205–206, 211–216 | 190 |
| 1 | 857–859, 877–889, 891–892 | 250 |
| 2 | 947–967, 994–1053, 1059–1078, 1166–1181, 1204–1211, 1234–1239 | 180 |

All 173 collision frames have every action component masked invalid. The 620
additional fully invalid frames demonstrate why the composite mask and collision
mask are reported separately.

Playback reports measured-pose mesh intersections separately from PhysX contacts.
PhysX contact flags aggregate all substeps leading to a frame; mesh checks sample
the exact frame boundary. Contact offsets can report contact before meshes
intersect; tracking lag, physical collision response, and IK's redundant solutions
can change which frames collide. `physx_penetration` requires reported separation
below -1 micrometre. Mesh checks are discrete, not continuous collision detection.
The recorder uses PhysX's native pairwise contact tensors and enables contact
processing explicitly. Kit's contact callbacks returned no events in this GPU
configuration despite nonzero native contact impulses.

Finite masked actions are deliberately retained for this simulation analysis,
including the dataset's overridden failed region. Masks are recorded unchanged.
This script is a simulation player, not a hardware execution interface.

## Validated run

The full headless run in `outputs/final-validation` executed all 1,069 frames in
each mode at `--speed 1`, recording 2,138 frame samples and 17,062 physics samples.
The saved measured TCP positions agree with independent FK of the measured joints
within 1 micrometre; orientations agree within 2 microradians. The 2× speed check
also preserved every input frame and the expected simulation timestamps.

| Episode | Joint TCP RMSE, left/right (mm) | IK TCP RMSE, left/right (mm) | Measured mesh frames, joint/IK | PhysX contact frames, joint/IK |
| --- | --- | --- | --- | --- |
| 0 | 35.0 / 38.2 | 98.1 / 105.0 | 15 / 2 | 19 / 5 |
| 1 | 32.6 / 70.0 | 68.8 / 96.6 | 17 / 8 | 19 / 8 |
| 2 | 50.9 / 49.9 | 87.1 / 75.6 | 98 / 89 | 128 / 108 |

The dynamic collision frames **do not exactly match** the nominal dataset labels.
Tracking errors are substantial at 1× with the supplied gains. In particular,
masked source frame 892 reaches right-TCP position errors of 733 mm in joint mode
and 705 mm in IK mode; the desired TCP moves 122 mm during the final 1/30 s interval.
The input is retained faithfully, including this masked motion.

See the run's `summary.json` for confusion matrices and inclusive frame ranges,
`verification.json` for recording checks, and `collision-comparison.png` and
`tracking-errors.png` for plots. Every episode's `frames.csv` identifies the exact
source frames, and `collision_pairs.json` identifies the contacting links.

## Checks

```bash
python -m unittest discover -s tests
python playback.py --mode both --headless --episodes 0 --max-frames 4
```

The small regression suite covers speed/frame alignment, quaternion sign changes,
and the lossless visual-mesh repair. Full-run collision and kinematic checks are
part of playback itself.
