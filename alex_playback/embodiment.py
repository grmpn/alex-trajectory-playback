"""Names and coordinate conventions of the Purdue Alex playback embodiment."""

ARM_JOINTS = [
    [
        f"{side}_{joint}"
        for joint in (
            "SHOULDER_Y",
            "SHOULDER_X",
            "SHOULDER_Z",
            "ELBOW_Y",
            "WRIST_Z",
            "WRIST_X",
            "GRIPPER_Y",
        )
    ]
    for side in ("LEFT", "RIGHT")
]
ACTION_NAMES = [
    *ARM_JOINTS[0],
    *ARM_JOINTS[1],
    "NECK_Z",
    "NECK_Y",
    "left_gripper_opening",
    "right_gripper_opening",
]
URDF = "assets/robots/alex_purdue/urdf/baseline/alex_purdue_full_convex.urdf"
BASE_FRAME = "PEDESTAL_LINK"
TOOL_FRAMES = ["LEFT_EZGRIPPER_TCP_LINK", "RIGHT_EZGRIPPER_TCP_LINK"]
