import pickle
from pathlib import Path

import mujoco as mj
import numpy as np

from .params import ROBOT_BASE_DICT, ROBOT_XML_DICT


def _load_pickle_motion(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)

    motion_fps = motion_data["fps"]
    motion_root_pos = motion_data["root_pos"]
    # GMR pickle files store quaternions as xyzw; MuJoCo expects wxyz.
    motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]
    motion_dof_pos = motion_data["dof_pos"]
    motion_local_body_pos = motion_data.get("local_body_pos")
    motion_link_body_list = motion_data.get("link_body_list")
    return (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    )


def _load_npz_motion(motion_file, robot_type):
    if robot_type is None:
        raise ValueError("robot_type is required when loading an NPZ robot motion")
    if robot_type not in ROBOT_XML_DICT:
        raise ValueError(f"Unknown robot type: {robot_type}")

    required_keys = {"fps", "joint_pos", "body_pos_w", "body_quat_w", "body_names"}
    with np.load(motion_file, allow_pickle=False) as npz:
        missing_keys = required_keys.difference(npz.files)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"NPZ motion is missing required arrays: {missing}")
        # Materialize the arrays before the NpzFile is closed.
        motion_data = {key: npz[key] for key in npz.files}

    body_names = [str(name) for name in motion_data["body_names"]]
    root_body_name = ROBOT_BASE_DICT[robot_type]
    if root_body_name not in body_names:
        raise ValueError(
            f"NPZ motion has no root body {root_body_name!r}; "
            f"available bodies: {body_names}"
        )
    root_body_index = body_names.index(root_body_name)

    joint_pos = motion_data["joint_pos"]
    joint_body_names = [
        name for index, name in enumerate(body_names) if index != root_body_index
    ]
    if len(joint_body_names) != joint_pos.shape[1]:
        raise ValueError(
            "Cannot infer NPZ joint order: expected one joint per non-root body, "
            f"but found {joint_pos.shape[1]} joints and {len(joint_body_names)} bodies"
        )

    # Isaac/PhysX motion exports order joints like body_names (typically breadth
    # first), whereas RobotMotionViewer writes qpos in MuJoCo XML order. Map each
    # 1-DoF joint through the body it moves to make the ordering unambiguous.
    source_column_by_body = {
        body_name: column for column, body_name in enumerate(joint_body_names)
    }
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_type]))
    source_columns = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
            continue
        if model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_HINGE:
            joint_name = model.joint(joint_id).name
            raise ValueError(f"NPZ loading does not support joint {joint_name!r}")
        child_body_name = model.body(int(model.jnt_bodyid[joint_id])).name
        if child_body_name not in source_column_by_body:
            raise ValueError(
                f"Cannot map MuJoCo joint {model.joint(joint_id).name!r}: "
                f"body {child_body_name!r} is absent from the NPZ body_names"
            )
        source_columns.append(source_column_by_body[child_body_name])

    if len(source_columns) != joint_pos.shape[1]:
        raise ValueError(
            f"Robot model has {len(source_columns)} hinge joints, "
            f"but the NPZ motion has {joint_pos.shape[1]} joint positions"
        )

    motion_fps = int(np.asarray(motion_data["fps"]).reshape(-1)[0])
    motion_root_pos = motion_data["body_pos_w"][:, root_body_index]
    # Isaac/PhysX world quaternions are already scalar-first (wxyz).
    motion_root_rot = motion_data["body_quat_w"][:, root_body_index]
    motion_dof_pos = joint_pos[:, source_columns]

    # Add canonical aliases so callers inspecting motion_data can use the same
    # playback fields for either source format without discarding NPZ metadata.
    motion_data.update(
        {
            "root_pos": motion_root_pos,
            "root_rot_wxyz": motion_root_rot,
            "dof_pos": motion_dof_pos,
            "link_body_list": body_names,
        }
    )
    return (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        None,
        body_names,
    )


def load_robot_motion(motion_file, robot_type=None):
    """Load a GMR pickle or an Isaac/PhysX-style NPZ robot motion."""
    suffix = Path(motion_file).suffix.lower()
    if suffix == ".pkl":
        return _load_pickle_motion(motion_file)
    if suffix == ".npz":
        return _load_npz_motion(motion_file, robot_type)
    raise ValueError(f"Unsupported robot motion format {suffix!r}; use .pkl or .npz")
