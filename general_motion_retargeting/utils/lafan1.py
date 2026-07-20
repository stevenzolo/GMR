import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


SUPPORTED_BVH_FORMATS = (
    "lafan1",
    "nokov",
    "noitom",
    "noi",
    "motive5spine",
    "motive_5spine",
    "motive2spine",
    "motive_2spine",
    "vicon",
)

_FORMAT_ALIASES = {
    "noi": "noitom",
    "motive_5spine": "motive5spine",
    "motive_2spine": "motive2spine",
}

_FOOT_ORIENTATION_BONES = {
    "lafan1": ("LeftToe", "RightToe"),
    "nokov": ("LeftToeBase", "RightToeBase"),
    "noitom": ("LeftFoot", "RightFoot"),
    "motive5spine": ("LeftToeBase", "RightToeBase"),
    "motive2spine": ("LeftToeBase", "RightToeBase"),
    "vicon": ("LeftToeBase", "RightToeBase"),
}


def normalize_bvh_format(format):
    format = format.lower()
    return _FORMAT_ALIASES.get(format, format)


def load_bvh_file(bvh_file, format="lafan1"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    format = normalize_bvh_format(format)
    if format not in _FOOT_ORIENTATION_BONES:
        supported = ", ".join(SUPPORTED_BVH_FORMATS)
        raise ValueError(f"Invalid format: {format}. Supported formats: {supported}")

    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
            
        left_orientation_bone, right_orientation_bone = _FOOT_ORIENTATION_BONES[format]
        for bone in ("LeftFoot", "RightFoot", left_orientation_bone, right_orientation_bone):
            if bone not in result:
                raise KeyError(f"BVH file {bvh_file} is missing required {format} bone: {bone}")

        # Add modified foot poses expected by the BVH IK config files. Some
        # sources have toe joints while Noitom does not, so it reuses foot
        # orientation directly.
        result["LeftFootMod"] = [
            result["LeftFoot"][0],
            result[left_orientation_bone][1],
        ]
        result["RightFootMod"] = [
            result["RightFoot"][0],
            result[right_orientation_bone][1],
        ]

        frames.append(result)

    # human_height = result["Head"][0][2] - min(
    #     result["LeftFootMod"][0][2], result["RightFootMod"][0][2]
    # )
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height
