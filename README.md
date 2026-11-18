# Alex trajectory playback

Playback of LeRobot v3 trajectories on fixed-base Purdue Alex
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

# All three included joint trajectories, one GUI.
./scripts/play_dataset.sh

# Also include all three EE trajectories when wanted.
./scripts/play_dataset.sh --mode both

# Joint trajectory playback, all episodes, GUI at half speed.
python playback.py --mode joint --speed 0.5

# Closed-loop DifferentialIKController playback of action.ee.
python playback.py --mode ee --speed 0.5

# Any Alex LeRobot v3 episode, with an optional frame interval.
python playback.py --dataset /path/to/lerobot-v3 --episodes 7 \
  --mode ee --start-frame 120 --end-frame 300 --speed 0.5

# Execute both modes and record a full comparison without a GUI.
python playback.py --mode both --headless --output outputs/comparison
python analyze.py outputs/comparison

# Recheck nominal URDF collisions without launching Isaac Sim.
python playback.py --mode audit
```

The launcher uses `~/IsaacLab/.venv/bin/python`; set `ISAACLAB_PYTHON` to use another
installation. Arguments such as `--dataset`, `--episodes`, `--speed`, `--headless`,
and `--exit-after` pass through to playback. Joint playback is the default;
`--mode both` runs joint and EE playback in the same simulation window.

The GUI shows the physical robot driven by the supplied PD actuators and a cyan,
transparent reference at the dataset's joint actions, including during EE playback.
The reference has no rigid bodies or collision geometry. Intersecting reference
links turn orange; simulated links turn red for measured mesh intersections or
PhysX penetration. The panel shows the episode, original frame/source indices,
collision pairs, and Pause/Resume. Use normal Isaac Sim viewport camera navigation.
The visibility checkboxes let you inspect the full desired robot or PD robot alone.
Every reference mesh and material subset uses the transparent material, including
the head, torso, and both arms.
The final pose stays visible until the window closes; `--exit-after` closes
automatically. `--capture-preview` saves each trajectory's final viewport.

`--ghost-opacity 0.25` controls reference transparency. The default `raytraced`
renderer avoids the default Isaac Sim path-tracing pipeline; `minimal` and
`pathtraced` are also available. Initial shader-cache warmup can take several
minutes; subsequent launches reuse the cache. `--render-fps` controls intermediate
rendering cadence (default 30); every original trajectory frame is displayed.

The local dataset lives in `datasets/iphone-robot`; its source-frame collision
diagnostics live in `datasets/iphone-export-report`. These inputs and generated
outputs are Git-ignored. `--dataset`, `--reports`, and `--assets` override their
locations. `--episodes 0 2` selects episodes; `--max-frames 10` runs a short check.
`--start-frame` is inclusive and `--end-frame` is exclusive, in each episode's
frame-index space. Original timestamps and source indices are preserved in logs.
Output directories must be new so previous recordings cannot be overwritten.

`--speed` scales dataset time relative to simulated time. It does not skip frames
or merely change GUI delays. The integrator uses enough substeps to retain every
original 30 Hz sample while respecting `--physics-dt` (default maximum 1/240 s).
GUI playback is paced to wall time by default; use `--no-realtime` to run as fast
as possible. Headless playback runs without wall-time pacing by default. Actual
wall-time playback can be slower than requested if the machine cannot keep up.

## LeRobot v3 inputs

Playback reads `meta/info.json` and selected episodes from `data/**/*.parquet`;
video decoding, LeRobot installation, export reports, and a dataset README are
not required. Nonzero episode IDs and reordered joint columns are supported.
The episode's `frame_index` must start at zero and be contiguous, with timestamps
matching the declared FPS. Source-frame indices may contain gaps.

- `--joint-key action`: a vector with `features.<key>.names` listing Alex names.
  Names are the seven `LEFT_*` and seven `RIGHT_*` arm joints, `NECK_Z`, `NECK_Y`,
  `left_gripper_opening`, and `right_gripper_opening`; see
  [embodiment.py](alex_playback/embodiment.py). Any ordering or named subset is
  accepted. Unspecified joints hold zero and unspecified grippers stay open.
- `--ee-key action.ee`: shape `(2, 7)` or `(2, 8)` per frame, ordered left/right:
  `[x, y, z, qx, qy, qz, qw, optional_opening]`. Positions are metres in
  `PEDESTAL_LINK`, quaternions are unit **xyzw**, and openings range from 0 to 1.
  For EE poses stored in `action`, use `--mode ee --ee-key action`.
- With joint actions only, EE targets come from FK. With EE actions only, IK
  settles from a zero-joint seed; the overlay shows **IK joint targets**, explicitly
  labeled as such because there is no recorded joint reference.
- `quality.action_valid` and source-video indices are optional. Collision reports
  are auto-selected only for the bundled dataset; use `--reports` for compatible
  exports elsewhere. Missing masks and collision labels are marked unavailable,
  and comparisons are `null`, rather than interpreted as collision-free frames.

These inputs must describe the Purdue Alex embodiment and absolute TCP poses;
relative EE deltas or trajectories for another robot require conversion first.

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
| `steps.npz` | Every physics sample: reference/desired/actual joint positions, desired/actual TCP poses, dataset/simulation times, frame indices, contact flags |
| `frames.npz` | The same measurements at original dataset frame boundaries, plus source video indices, action-valid masks, and collision labels |
| `frames.csv` | Frame/source-frame indices, collision flags, mask status, and TCP position errors |
| `collision_pairs.json` | Nominal and measured mesh pairs, dataset pairs, and PhysX contacts with minimum separation and accumulated normal impulse |
| `metadata.json` | Timing, actuator types, asset identity, import repairs, pose conventions, and completion status |
| `summary.json` | Joint/TCP tracking errors, collision confusion matrices, and inclusive source-frame collision ranges |
| `preview.png` | Final viewport when GUI playback uses `--capture-preview` |

Arrays `desired_joint_position` and `actual_joint_position` use the explicit
`joint_names` array, including measured mimic followers. Poses have shape
`(samples, 2, 7)`. In joint mode desired TCPs are FK of the commanded joint
configuration. In IK mode they are the interpolated `action.ee` targets.
Settling is excluded from the recorded trajectory clock; frame zero records its
final state. The last original dataset timestamp is retained exactly.
`reference_joint_position` stores the overlay configuration: original joint actions
when available, otherwise IK targets. `metadata.json` identifies which. NPZ files
also include `has_joint_reference`, `has_collision_labels`, and `has_valid_mask`;
read these flags before interpreting optional reference arrays.

```python
import numpy as np

log = np.load("outputs/comparison/joint/episode-0/frames.npz")
q_desired = log["desired_joint_position"]
q_actual = log["actual_joint_position"]
tcp_desired = log["desired_tcp_pose_base"]
tcp_actual = log["actual_tcp_pose_base"]
```

## TCP tracking and fixed-target diagnostics

```bash
# X/Y/Z desired vs actual, both TCPs, each episode and recorded control mode.
python analyze.py outputs/archive/2026-11-17/final-validation --output outputs/tcp-plots

# Abruptly switch from episode 0 frame 0 to frame 35, then hold for 8 seconds.
python playback.py --mode joint --headless --episodes 0 --end-frame 36 \
  --step-target-frame 35 --step-delay 1 --step-hold 8 --settle-time 2 \
  --output outputs/step-response
python analyze.py outputs/step-response
```

Analysis writes a six-panel PNG for each episode/mode, a combined
`tcp-position-tracking.pdf`, and per-axis/3D RMSE and peak errors in
`tcp-position-summary.json`. RMSE and peaks use every recorded physics sample;
3D error is the Euclidean distance between actual and desired TCP positions.
Plots use every physics sample, simulation seconds, and TCP positions in metres
in `PEDESTAL_LINK`. Desired TCPs in joint mode come from FK of joint commands;
actual TCPs come from simulated link poses. Pink shading marks PhysX contact.
The existing collision and position-error plots are also generated for playback.

The step experiment requires headless joint mode at speed 1. It initializes at
`--start-frame`, settles under PD control, records a baseline for `--step-delay`,
then switches all joint targets to `--step-target-frame` without interpolation.
Joint velocity targets remain zero, and the upstream gains, torque limits,
gravity compensation, and contacts are unchanged. Durations round up to physics
steps. The target frame must be within each selected episode/frame interval.
The defaults are a 1-second baseline and an 8-second hold.

Step recordings use `joint/episode-N/steps.npz` and `metadata.json`, with the
source frames and synthetic experiment timing recorded explicitly. At time `t`,
the log pairs the measured state with the command applied over `[t, t+dt)`;
the command changes at the marked step time, before the robot responds.
`summary.json` includes final errors averaged over the last 0.5 seconds, contact
counts, and time after the switch to enter and remain within 2 mm of the target.
Settling is `null` unless the final in-band interval lasts at least 0.5 seconds.
Analysis saves `fixed-target-step-response-episode-N.png` and a PDF; original
episode collision audits are not plotted against this synthetic clock.

For damping experiments, `--kd-scale` multiplies the upstream damping gains of
all 20 actuated arm, neck, and gripper joints. Position gains, armature, limits,
and the four passive mimic followers are unchanged. The default is `1.0`.
Resolved gains for each active joint are saved in `metadata.json` as
`pd_gains_by_joint`, using N m/rad for Kp and N m s/rad for Kd.

```bash
python playback.py --mode joint --headless --episodes 0 --kd-scale 1.5 \
  --output outputs/episode-0-kd-1p5
python analyze.py outputs/episode-0-kd-1p5
```

Normal playback still supplies the finite-difference desired joint velocities.
Scaling Kd therefore scales both velocity feedback and desired-velocity
feedforward in the PD velocity-error term. The fixed-target step experiment
continues to use zero desired velocity, including with a damping override.

Previous-day output folders are organized under `outputs/archive/YYYY-MM-DD/`.
The archive contains a move manifest; recorded paths inside older run metadata
retain their original provenance. The 2026-11-18 damping sweep, joint-window
plots, plotting script, and CSV/JSON metrics are in
`outputs/damping-sweep-20261118/`. Its comparison uses all 2,145 physics samples
of episode 0, with the archived 1× run as the baseline. The joint plots cover
all 24 recorded joints in episode 1 from 2.5 through 5.0 simulation seconds.

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

GUI joint playback completed all three episodes in
`outputs/archive/2026-11-17/gui-validation` with joint states identical to the
headless recordings. The corrected full-reference
materials and Pause/Resume were checked on episode 0, frames 50–57, at half speed
in `outputs/archive/2026-11-17/full-reference-check`. Its `reference-only.png`,
`pd-only.png`, and `preview.png` show the separate robots and combined overlay. The viewer applies
the reference material directly to every mesh and material subset, including
deinstancing visual geometry where necessary; collision geometry is unaffected.

The full headless run in `outputs/archive/2026-11-17/final-validation` executed
all 1,069 frames in each mode at `--speed 1`, recording 2,138 frame samples and
17,062 physics samples.
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
lossless visual-mesh repair, LeRobot episode/slice selection, named joint ordering,
and EE-only inputs. Full-run collision and kinematic checks are part of playback.
