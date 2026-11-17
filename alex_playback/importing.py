"""Small, local adaptations for the Isaac Sim 6.2 URDF importer."""

import xml.etree.ElementTree as ET
from pathlib import Path


def compact_obj(source: Path, destination: Path) -> int:
    """Drop only unused position vertices; preserve faces, normals, UVs and materials."""
    lines = source.read_text().splitlines(keepends=True)
    count = 0
    faces = {}
    used = set()
    for number, line in enumerate(lines):
        if line.startswith("v "):
            count += 1
        elif line.startswith("f "):
            corners = [part.split("/") for part in line.split()[1:]]
            for corner in corners:
                index = int(corner[0])
                corner[0] = index if index > 0 else count + index + 1
                used.add(corner[0])
            faces[number] = corners
    if len(used) == count:
        return 0
    if not used or min(used) < 1 or max(used) > count:
        raise ValueError(f"Invalid OBJ vertex indices in {source}")
    mapping = {old: new for new, old in enumerate(sorted(used), start=1)}
    result = []
    index = 0
    for number, line in enumerate(lines):
        if line.startswith("v "):
            index += 1
            if index not in used:
                continue
        elif number in faces:
            line = (
                "f "
                + " ".join("/".join([str(mapping[c[0]]), *c[1:]]) for c in faces[number])
                + "\n"
            )
        elif line.startswith("mtllib "):
            line = (
                "mtllib "
                + str((source.parent / line.split(maxsplit=1)[1].strip()).resolve())
                + "\n"
            )
        result.append(line)
    destination.write_text("".join(result))
    return count - len(used)


def prepare_urdf(urdf: Path, directory: Path):
    """Keep canonical physics untouched; the new USD importer rejects unused OBJ vertices."""
    directory.mkdir(parents=True)
    tree = ET.parse(urdf)
    repairs = []
    for number, mesh in enumerate(tree.findall(".//visual/geometry/mesh")):
        source = (urdf.parent / mesh.get("filename")).resolve()
        if source.suffix.lower() == ".obj":
            destination = directory / f"visual-{number}-{source.name}"
            removed = compact_obj(source, destination)
            if removed:
                mesh.set("filename", str(destination))
                repairs.append(
                    {
                        "source": str(source),
                        "generated": str(destination),
                        "unused_vertices_removed": removed,
                    }
                )
    if not repairs:
        return urdf, repairs
    for mesh in tree.findall(".//mesh"):
        mesh.set("filename", str((urdf.parent / mesh.get("filename")).resolve()))
    generated = directory / urdf.name
    tree.write(generated, encoding="utf-8", xml_declaration=True)
    return generated, repairs


def fix_world_joint(stage, root_path: str, base_frame: str):
    """Make the imported fixed joint attach to world and scope the articulation above it.

    Sim 6.2 points body0 at the non-rigid asset Xform and places the articulation
    API on the base body. PhysX then reports a floating base. Author local USD
    overrides without editing the upstream URDF or removing payload properties.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    prims = list(Usd.PrimRange(root))
    bases = [p for p in prims if p.GetName() == base_frame and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bases) != 1:
        raise ValueError("Cannot identify the imported dataset base link")
    base = bases[0]
    joints = [
        UsdPhysics.FixedJoint(p)
        for p in prims
        if p.IsA(UsdPhysics.FixedJoint)
        and UsdPhysics.FixedJoint(p).GetBody1Rel().GetTargets() == [base.GetPath()]
    ]
    if len(joints) != 1:
        raise ValueError("Expected the URDF importer's single fixed root joint")
    joints[0].GetBody0Rel().SetTargets([])
    joints[0].GetJointEnabledAttr().Set(True)
    for prim in prims:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAppliedSchema("PhysxArticulationAPI")
            prim.RemoveAppliedSchema("NewtonArticulationRootAPI")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    api = PhysxSchema.PhysxArticulationAPI.Apply(root)
    api.CreateEnabledSelfCollisionsAttr().Set(True)
    api.CreateSolverPositionIterationCountAttr().Set(64)
    api.CreateSolverVelocityIterationCountAttr().Set(8)
    root.AddAppliedSchema("NewtonArticulationRootAPI")
    root.GetAttribute("newton:selfCollisionEnabled").Set(True)
