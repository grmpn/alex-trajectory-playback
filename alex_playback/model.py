"""URDF FK and independent discrete mesh collision checks (Pinocchio/Coal)."""

import ast
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np
import pinocchio as pin
from ihmc_alex_isaaclab.end_effectors.sake_ezgripper import (
    alex_purdue_ezgripper_targets,
)
from scipy.spatial.transform import Rotation


class RobotModel:
    def __init__(self, urdf: Path, assets: Path, base_frame: str, tool_frames: list[str]):
        self.urdf = urdf.resolve()
        self.model = pin.buildModelFromUrdf(str(self.urdf))
        self.data = self.model.createData()
        self.names = list(self.model.names)[1:]
        self.indices = {
            name: self.model.joints[self.model.getJointId(name)].idx_q for name in self.names
        }
        self.base = self.model.getFrameId(base_frame)
        self.tools = [self.model.getFrameId(name) for name in tool_frames]
        self.tool_names = tool_frames
        if max([self.base, *self.tools]) >= len(self.model.frames):
            raise ValueError("Dataset base/TCP frames are absent from the URDF")
        tree = ET.parse(self.urdf).getroot()
        self.parents = {j.find("child").get("link"): j for j in tree.findall("joint")}
        self.mimics = {}
        for joint in tree.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None:
                self.mimics[joint.get("name")] = (
                    mimic.get("joint"),
                    float(mimic.get("multiplier", 1)),
                    float(mimic.get("offset", 0)),
                )
        adjacent = {
            frozenset((self.assembly(child), self.assembly(j.find("parent").get("link"))))
            for child, j in self.parents.items()
            if j.get("type") != "fixed"
        }
        module = ast.parse((assets / "src/ihmc_alex_isaaclab/robots/purdue_physics.py").read_text())
        exclusions = next(
            ast.literal_eval(n.value)
            for n in module.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS"
                for t in n.targets
            )
        )
        self.excluded_assemblies = adjacent | {frozenset(map(self.assembly, p)) for p in exclusions}
        self.geometry = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf),
            pin.GeometryType.COLLISION,
            package_dirs=[str(self.urdf.parent)],
        )
        self.geometry_links = [
            self.model.frames[g.parentFrame].name for g in self.geometry.geometryObjects
        ]
        for a, b in combinations(range(len(self.geometry_links)), 2):
            if not self.excluded(self.geometry_links[a], self.geometry_links[b]):
                self.geometry.addCollisionPair(pin.CollisionPair(a, b))
        self.geometry_data = pin.GeometryData(self.geometry)

    def assembly(self, link: str) -> str:
        while link in self.parents and self.parents[link].get("type") == "fixed":
            link = self.parents[link].find("parent").get("link")
        return link

    def excluded(self, first: str, second: str) -> bool:
        pair = frozenset((self.assembly(first), self.assembly(second)))
        return len(pair) == 1 or pair in self.excluded_assemblies

    def configuration(self, names, positions) -> np.ndarray:
        """Map a complete measured configuration without imposing mimic coupling."""
        if len(names) != len(positions) or set(names) != set(self.names):
            raise ValueError("Measured joint names must cover every movable URDF joint")
        q = np.zeros(self.model.nq)
        for name, value in zip(names, positions, strict=True):
            q[self.indices[name]] = value
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite simulated joint configuration")
        return q

    def action_configuration(self, action_names, action) -> np.ndarray:
        values = dict(zip(action_names[:16], action[:16], strict=True))
        for side, opening in zip(("left", "right"), action[-2:], strict=True):
            values.update(alex_purdue_ezgripper_targets(float(opening), side))
        for follower, (leader, multiplier, offset) in self.mimics.items():
            values[follower] = values[leader] * multiplier + offset
        return self.configuration(list(values), list(values.values()))

    def ordered(self, q, names) -> np.ndarray:
        return np.array([q[self.indices[name]] for name in names])

    def tcp(self, q: np.ndarray) -> np.ndarray:
        pin.framesForwardKinematics(self.model, self.data, q)
        inverse_base = self.data.oMf[self.base].inverse()
        poses = []
        for tool in self.tools:
            transform = inverse_base * self.data.oMf[tool]
            poses.append(
                np.r_[transform.translation, Rotation.from_matrix(transform.rotation).as_quat()]
            )
        return np.array(poses)

    def tcp_anchor(self, tool: str) -> tuple[str, np.ndarray]:
        """TCP offset from the unmerged rigid link used by PhysX and its Jacobian."""
        anchor = self.assembly(tool)
        pin.framesForwardKinematics(self.model, self.data, np.zeros(self.model.nq))
        transform = (
            self.data.oMf[self.model.getFrameId(anchor)].inverse()
            * self.data.oMf[self.model.getFrameId(tool)]
        )
        return anchor, np.r_[
            transform.translation, Rotation.from_matrix(transform.rotation).as_quat()
        ]

    def collisions(self, q: np.ndarray) -> list[tuple[str, str]]:
        pin.computeCollisions(self.model, self.data, self.geometry, self.geometry_data, q, False)
        return sorted(
            {
                tuple(sorted((self.geometry_links[p.first], self.geometry_links[p.second])))
                for p, result in zip(
                    self.geometry.collisionPairs, self.geometry_data.collisionResults, strict=True
                )
                if result.isCollision()
            }
        )


def pose_error(actual: np.ndarray, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Float32 normalization can create spurious milliradian errors near identical rotations.
    actual, desired = np.asarray(actual, dtype=np.float64), np.asarray(desired, dtype=np.float64)
    position = np.linalg.norm(actual[..., :3] - desired[..., :3], axis=-1)
    a = actual[..., 3:7] / np.linalg.norm(actual[..., 3:7], axis=-1, keepdims=True)
    b = desired[..., 3:7] / np.linalg.norm(desired[..., 3:7], axis=-1, keepdims=True)
    angle = 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))
    return position, angle
