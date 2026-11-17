"""Kit viewer: measured robot, nonphysical joint-reference overlay, and collision status."""

import asyncio
import time

import numpy as np
import pinocchio as pin


def configure_renderer(mode):
    from isaacsim import SimulationApp

    # Lab 3's launcher forwards only a small subset of SimulationApp settings.
    # Set public defaults for this process before it constructs the app. No installed
    # package or persistent user configuration is edited.
    SimulationApp.DEFAULT_LAUNCHER_CONFIG.update(
        renderer={
            "minimal": "MinimalRendering",
            "raytraced": "RaytracedLighting",
            "pathtraced": "RealTimePathTracing",
        }[mode],
        minimal_shading_mode=2,
        window_width=1440,
        window_height=1000,
    )


class Viewer:
    def __init__(self, stage, model, opacity, speed, renderer):
        from omni import ui
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        self.stage, self.model = stage, model
        self.paused = False
        self.finished = False
        self.actual_visuals = {}
        self.original_bindings = {}
        self.ghost_visuals = {}
        self.ghost_links = {}
        self.highlighted = set()
        self.ghost_highlighted = set()
        self.reference_label = "Dataset joint actions"
        self.base_transform = np.eye(4)
        self.renderer = renderer

        def material(name, color, alpha):
            path = f"/World/PlaybackMaterials/{name}"
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(alpha)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat

        self.ghost_material = material("Reference", (0.05, 0.7, 1.0), opacity)
        self.ghost_collision_material = material(
            "ReferenceCollision", (1.0, 0.25, 0.02), max(opacity, 0.4)
        )
        self.collision_material = material("MeasuredCollision", (1.0, 0.03, 0.03), 1.0)
        UsdGeom.Xform.Define(stage, "/World/Reference")
        robot = stage.GetPrimAtPath("/World/Robot")
        bodies = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        for body in bodies:
            visuals = []
            for child in body.GetChildren():
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                descendants = list(Usd.PrimRange(child, Usd.TraverseInstanceProxies()))
                if any(p.HasAPI(UsdPhysics.CollisionAPI) for p in descendants):
                    continue
                if any(p.IsA(UsdGeom.Mesh) for p in descendants):
                    visuals.append(child)
            if not visuals:
                continue
            name = body.GetName()
            self.actual_visuals[name] = []
            for visual in visuals:
                self.make_editable(visual)
                for prim in Usd.PrimRange(visual):
                    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):
                        self.actual_visuals[name].append(prim)
                        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
                        self.original_bindings[str(prim.GetPath())] = (
                            rel.GetTargets() if rel else [],
                            UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel)
                            if rel
                            else UsdShade.Tokens.weakerThanDescendants,
                        )
            ghost = UsdGeom.Xform.Define(stage, f"/World/Reference/{name}")
            op = ghost.AddTransformOp()
            for visual in visuals:
                prim = stage.DefinePrim(ghost.GetPath().AppendChild(visual.GetName()))
                prim.GetReferences().AddInternalReference(visual.GetPath())
            self.ghost_visuals[name] = [
                prim
                for prim in Usd.PrimRange(ghost.GetPrim())
                if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)
            ]
            # RTX resolves per-face subset materials separately from link bindings.
            # Override every mesh/subset, including the multi-material arms and head.
            for prim in self.ghost_visuals[name]:
                self.bind(prim, self.ghost_material)
            self.ghost_links[name] = (ghost.GetPrim(), op, model.model.getFrameId(name))
        if not self.ghost_links:
            raise ValueError("No Alex visual geometry found for the reference overlay")
        # The reference contains only visual subtrees; it must never affect dynamics.
        for prim in Usd.PrimRange(
            stage.GetPrimAtPath("/World/Reference"), Usd.TraverseInstanceProxies()
        ):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError("Reference overlay unexpectedly contains physics geometry")

        self.window = ui.Window(
            "Alex playback", width=440, height=430, position_x=16, position_y=100
        )
        with self.window.frame, ui.VStack(spacing=8, height=0):
            ui.Label("ALEX  /  TRAJECTORY PLAYBACK", height=28, style={"font_size": 20})
            self.title = ui.Label("Preparing scene…", height=24)
            self.progress = ui.Label("", height=24)
            ui.Label(f"Playback speed: {speed:g}×  ·  CLI --speed changes motion timing", height=20)
            ui.Label("Solid: simulated robot with supplied PD actuators", height=20)
            self.legend = ui.Label(
                "Cyan transparent: dataset joint actions",
                height=36,
                word_wrap=True,
                style={"color": 0xFFFFCC55},
            )
            with ui.HStack(height=24, spacing=8):
                for label, path in (
                    ("Show PD robot", "/World/Robot"),
                    ("Show desired robot", "/World/Reference"),
                ):
                    checkbox = ui.CheckBox(width=20)
                    checkbox.model.set_value(True)
                    checkbox.model.add_value_changed_fn(
                        lambda value, path=path: self.set_visible(path, value.as_bool)
                    )
                    ui.Label(label)
            self.status = ui.Label("Checking self-collisions…", height=24, style={"font_size": 18})
            self.pair_text = ui.Label("", word_wrap=True, height=95)
            self.pause_button = ui.Button("Pause", clicked_fn=self.toggle_pause, height=30)
        for name in (
            "Content",
            "Content Browser",
            "Stage",
            "Property",
            "Layer",
            "Render Settings",
            "Console",
        ):
            window = ui.Workspace.get_window(name)
            if window:
                window.visible = False

    @staticmethod
    def make_editable(prim):
        """Deinstance visual geometry so subset bindings can be overridden locally."""
        from pxr import UsdShade

        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader):
            return
        if prim.IsInstance():
            prim.SetInstanceable(False)
        for child in prim.GetChildren():
            Viewer.make_editable(child)

    def set_visible(self, path, visible):
        from pxr import UsdGeom

        UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
        )

    @staticmethod
    def bind(prim, material):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    def set_base_pose(self, pose):
        from scipy.spatial.transform import Rotation

        self.base_transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        self.base_transform[:3, 3] = pose[:3]

    def configure_viewport(self):
        import carb
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        mode = {
            "minimal": "MinimalRendering",
            "raytraced": "RaytracedLighting",
            "pathtraced": "RealTimePathTracing",
        }[self.renderer]
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", mode)
        settings.set("/rtx/minimal/mode", 2)
        viewport = get_active_viewport()
        product = viewport.stage.GetPrimAtPath(viewport.render_product_path)
        product.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set(mode)
        product.CreateAttribute("omni:rtx:minimal:mode", Sdf.ValueTypeNames.Int).Set(2)
        print(
            f"GUI renderer: {mode}; reference overlay: {len(self.ghost_links)} visual links",
            flush=True,
        )

    def set_episode(self, mode, episode):
        self.title.text = f"Episode {episode.index}  ·  {'Joint' if mode == 'joint' else 'End effector / Differential IK'}"
        self.reference_label = (
            "dataset joint actions"
            if episode.has_joint_actions
            else "IK joint targets (no dataset joint reference)"
        )
        self.legend.text = f"Cyan transparent: {self.reference_label}"

    def set_reference(self, configuration):
        from pxr import Gf

        pin.framesForwardKinematics(self.model.model, self.model.data, configuration)
        base = self.model.data.oMf[self.model.base].inverse()
        for _, op, frame_id in self.ghost_links.values():
            transform = self.base_transform @ (base * self.model.data.oMf[frame_id]).homogeneous
            op.Set(Gf.Matrix4d(*transform.T.ravel().tolist()))

    def highlight(self, reference_pairs, actual_pairs, contacts):
        from pxr import UsdShade

        penetrating = [pair for pair, data in contacts.items() if data["min_separation_m"] < -1e-6]
        actual = {
            self.model.assembly(link) for pair in [*actual_pairs, *penetrating] for link in pair
        }
        reference = {self.model.assembly(link) for pair in reference_pairs for link in pair}
        for name in self.highlighted - actual:
            for prim in self.actual_visuals.get(name, []):
                targets, strength = self.original_bindings[str(prim.GetPath())]
                if targets:
                    rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
                    rel.SetTargets(targets)
                    UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(rel, strength)
                else:
                    prim.RemoveProperty("material:binding")
        for name in actual - self.highlighted:
            for prim in self.actual_visuals.get(name, []):
                self.bind(prim, self.collision_material)
        for name in self.ghost_highlighted ^ reference:
            for prim in self.ghost_visuals.get(name, []):
                self.bind(
                    prim,
                    self.ghost_collision_material if name in reference else self.ghost_material,
                )
        self.highlighted, self.ghost_highlighted = actual, reference
        if actual:
            self.status.text = "SELF COLLISION  ·  simulated robot"
            self.status.style = {"color": 0xFF5555FF, "font_size": 18}
        elif reference:
            self.status.text = "SELF COLLISION  ·  reference actions"
            self.status.style = {"color": 0xFF3399FF, "font_size": 18}
        else:
            self.status.text = "No mesh self-collision" + (
                "  ·  contact margin active" if contacts else ""
            )
            self.status.style = {"color": 0xFF99DD99, "font_size": 18}
        pairs = [("PD", p) for p in sorted(set(actual_pairs) | set(penetrating))]
        pairs += [("REF", p) for p in reference_pairs]
        self.pair_text.text = "\n".join(f"{kind}: {a} ↔ {b}" for kind, (a, b) in pairs[:3])
        if len(pairs) > 3:
            self.pair_text.text += (
                f"\n+ {len(pairs) - 3} more pairs (recorded in collision_pairs.json)"
            )

    def set_frame(self, episode, index):
        self.progress.text = (
            f"Frame {episode.frame[index]} / {episode.full_length - 1}"
            f"  ·  source {episode.source_frame[index]}  ·  {episode.time[index]:.2f} s"
        )

    def toggle_pause(self):
        if not self.finished:
            self.paused = not self.paused
            self.pause_button.text = "Resume" if self.paused else "Pause"

    def wait_if_paused(self, simulation):
        if not self.paused:
            return 0.0
        start = time.monotonic()
        while self.paused and simulation.is_running():
            simulation.render()
            time.sleep(0.02)
        return time.monotonic() - start

    def capture(self, simulation, path):
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.widget.viewport.capture import FileCapture

        path.parent.mkdir(parents=True, exist_ok=True)
        # Empty AOV name selects whichever color buffer this renderer provides.
        capture = get_active_viewport().schedule_capture(FileCapture(str(path)))
        task = asyncio.ensure_future(capture.wait_for_result())
        deadline = time.monotonic() + 30
        while (
            (not task.done() or not path.exists())
            and simulation.is_running()
            and time.monotonic() < deadline
        ):
            simulation.render()
            time.sleep(0.01)
        if not task.done():
            task.cancel()
            raise RuntimeError("Viewport capture did not complete within 30 seconds")
        result = task.result()
        if not path.exists():
            raise RuntimeError(
                f"Viewport capture did not create the requested image (buffers: {result})"
            )

    def finish(self, simulation, keep_open):
        self.finished = True
        self.pause_button.enabled = False
        self.pause_button.text = "Playback complete"
        self.progress.text += "  ·  finished"
        if keep_open:
            print("Playback complete. Close the GUI window to exit.", flush=True)
            while simulation.is_running():
                simulation.render()
                time.sleep(0.02)
