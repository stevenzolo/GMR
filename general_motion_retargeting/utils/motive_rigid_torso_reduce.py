#!/usr/bin/env python3
"""Convert Motive 5-spine BVH files to a robot rigid-torso 2-spine BVH.

This is a local reducer used by ``csv_filter_retarget_to_npz.py`` so batch
retargeting does not depend on another repository.

Algorithm summary:
- output hierarchy/channel declarations come from the standard Motive 2-spine
  template;
- Hips channels are copied exactly from the source;
- ordinary retained joints keep same-name local rotations and positions;
- Spine is forced onto the source Hips->Neck line using the standard 2-spine
  template proportion;
- Spine1 is sampled from the source 5-spine torso centerline;
- Spine carries the fitted chest orientation while Spine1 is kept neutral for a
  robot-friendly rigid torso;
- Neck, Head, LeftShoulder and RightShoulder are re-anchored by source global
  transform because their parents change when Spine2/3/4 and Neck1 are removed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

EXPECTED_ROOT_CHANNELS = ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"]
EXPECTED_ROTATION_CHANNELS = ["Zrotation", "Xrotation", "Yrotation"]
ROTATION_CHANNELS = {"Xrotation", "Yrotation", "Zrotation"}
POSITION_CHANNELS = {"Xposition", "Yposition", "Zposition"}
REMOVED_SOURCE_JOINTS = {"Spine2", "Spine3", "Spine4", "Neck1"}
ANCHOR_WEIGHTS = {"Neck": 2.0, "LeftShoulder": 2.0, "RightShoulder": 2.0, "Head": 0.7}
REANCHORED_JOINTS = ("Neck", "LeftShoulder", "RightShoulder", "Head")
SOURCE_TORSO_CHAIN = ("Hips", "Spine", "Spine1", "Spine2", "Spine3", "Spine4", "Neck")
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "motive_2spine"
    / "lu_narrow_shoulder_20260330000016_1_1775009766.bvh"
)
LINE_TOLERANCE = 1e-5


@dataclass
class EndSite:
    offset: np.ndarray
    lines: list[str] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    kind: str
    parent: str | None
    offset: np.ndarray
    channels: list[str]
    children: list[str] = field(default_factory=list)
    end_sites: list[EndSite] = field(default_factory=list)
    line_no: int = 0
    channel_start: int = 0

    @property
    def channel_end(self) -> int:
        return self.channel_start + len(self.channels)


@dataclass
class BVH:
    path: Path
    hierarchy_lines: list[str]
    joints: list[Joint]
    frames: int
    frame_time: float
    motion: np.ndarray

    def __post_init__(self) -> None:
        self.joint_map = {joint.name: joint for joint in self.joints}
        if len(self.joint_map) != len(self.joints):
            raise ValueError(f"duplicate joint names in {self.path}")
        self.total_channels = sum(len(joint.channels) for joint in self.joints)

    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    def channel_slice(self, joint_name: str) -> slice:
        joint = self.joint_map[joint_name]
        return slice(joint.channel_start, joint.channel_end)

    def channel_index(self, joint_name: str, channel: str) -> int:
        joint = self.joint_map[joint_name]
        return joint.channel_start + joint.channels.index(channel)


def parse_bvh(path: Path) -> BVH:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        motion_idx = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as exc:
        raise ValueError(f"{path} does not contain a MOTION section") from exc

    hierarchy_lines = lines[:motion_idx]
    joints: list[Joint] = []
    stack: list[Joint] = []
    pending_joint: Joint | None = None
    pending_end: EndSite | None = None
    in_end_site = False

    for line_no, raw_line in enumerate(hierarchy_lines, start=1):
        stripped = raw_line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            kind, name = stripped.split(None, 1)
            parent = stack[-1].name if stack else None
            joint = Joint(name=name, kind=kind, parent=parent, offset=np.zeros(3), channels=[], line_no=line_no)
            if stack:
                stack[-1].children.append(name)
            joints.append(joint)
            pending_joint = joint
            pending_end = None
            in_end_site = False
        elif stripped == "End Site":
            if not stack:
                raise ValueError(f"End Site without parent at {path}:{line_no}")
            pending_end = EndSite(offset=np.zeros(3), lines=[])
            stack[-1].end_sites.append(pending_end)
            pending_joint = None
            in_end_site = True
        elif stripped == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end is not None:
                pending_end.lines.append(raw_line)
        elif stripped == "}":
            if in_end_site and pending_end is not None:
                pending_end.lines.append(raw_line)
                pending_end = None
                in_end_site = False
            elif stack:
                stack.pop()
        elif stripped.startswith("OFFSET"):
            values = np.array([float(part) for part in stripped.split()[1:]], dtype=float)
            if len(values) != 3:
                raise ValueError(f"OFFSET must contain 3 values at {path}:{line_no}")
            if in_end_site and pending_end is not None:
                pending_end.offset = values
                pending_end.lines.append(raw_line)
            elif stack:
                stack[-1].offset = values
        elif stripped.startswith("CHANNELS"):
            if not stack:
                raise ValueError(f"CHANNELS without current joint at {path}:{line_no}")
            parts = stripped.split()
            count = int(parts[1])
            channels = parts[2:]
            if len(channels) != count:
                raise ValueError(f"CHANNEL count mismatch at {path}:{line_no}")
            stack[-1].channels = channels

    channel_start = 0
    for joint in joints:
        joint.channel_start = channel_start
        channel_start += len(joint.channels)
        if joint.channels not in (EXPECTED_ROOT_CHANNELS, EXPECTED_ROTATION_CHANNELS):
            raise ValueError(
                f"{path}: joint {joint.name} has channels {joint.channels}; "
                f"expected either {EXPECTED_ROOT_CHANNELS} or {EXPECTED_ROTATION_CHANNELS}"
            )

    frames_line = lines[motion_idx + 1].strip()
    frame_time_line = lines[motion_idx + 2].strip()
    if not frames_line.startswith("Frames:") or not frame_time_line.startswith("Frame Time:"):
        raise ValueError(f"{path}: malformed MOTION header")
    frames = int(frames_line.split(":", 1)[1].strip())
    frame_time = float(frame_time_line.split(":", 1)[1].strip())

    rows = [[float(part) for part in line.split()] for line in lines[motion_idx + 3 :] if line.strip()]
    motion = np.array(rows, dtype=float)
    if motion.shape != (frames, channel_start):
        raise ValueError(f"{path}: expected motion shape {(frames, channel_start)}, got {motion.shape}")
    return BVH(path=path, hierarchy_lines=hierarchy_lines, joints=joints, frames=frames, frame_time=frame_time, motion=motion)


def rotation_sequence(joint: Joint) -> str:
    """Return scipy's intrinsic Euler sequence matching the BVH rotation order."""

    return "".join(channel[0].upper() for channel in joint.channels if channel in ROTATION_CHANNELS)


def joint_position_from_frame(joint: Joint, frame: np.ndarray) -> np.ndarray:
    position = joint.offset.astype(float).copy()
    for local_idx, channel in enumerate(joint.channels):
        value = frame[joint.channel_start + local_idx]
        if channel == "Xposition":
            position[0] = value
        elif channel == "Yposition":
            position[1] = value
        elif channel == "Zposition":
            position[2] = value
    return position


def joint_rotation_from_frame(joint: Joint, frame: np.ndarray) -> Rotation:
    angles = [frame[joint.channel_start + i] for i, channel in enumerate(joint.channels) if channel in ROTATION_CHANNELS]
    return Rotation.from_euler(rotation_sequence(joint), angles, degrees=True) if angles else Rotation.identity()


def set_joint_position(frame: np.ndarray, bvh: BVH, joint_name: str, position: np.ndarray) -> None:
    joint = bvh.joint_map[joint_name]
    for local_idx, channel in enumerate(joint.channels):
        idx = joint.channel_start + local_idx
        if channel == "Xposition":
            frame[idx] = position[0]
        elif channel == "Yposition":
            frame[idx] = position[1]
        elif channel == "Zposition":
            frame[idx] = position[2]


def set_joint_rotation(frame: np.ndarray, bvh: BVH, joint_name: str, rotation: Rotation | Iterable[float]) -> None:
    joint = bvh.joint_map[joint_name]
    angles = rotation.as_euler(rotation_sequence(joint), degrees=True) if isinstance(rotation, Rotation) else np.array(list(rotation))
    angle_idx = 0
    for local_idx, channel in enumerate(joint.channels):
        if channel in ROTATION_CHANNELS:
            frame[joint.channel_start + local_idx] = angles[angle_idx]
            angle_idx += 1


def copy_channels(dst_frame: np.ndarray, dst_bvh: BVH, src_frame: np.ndarray, src_bvh: BVH, joint_name: str, channels: set[str]) -> None:
    dst_joint = dst_bvh.joint_map[joint_name]
    src_joint = src_bvh.joint_map[joint_name]
    for dst_local_idx, channel in enumerate(dst_joint.channels):
        if channel in channels:
            src_local_idx = src_joint.channels.index(channel)
            dst_frame[dst_joint.channel_start + dst_local_idx] = src_frame[src_joint.channel_start + src_local_idx]


def fk(bvh: BVH, frame: np.ndarray) -> dict[str, np.ndarray]:
    """Compute global transforms with T_child = T_parent @ Translate @ Rotate."""

    transforms: dict[str, np.ndarray] = {}
    for joint in bvh.joints:
        local = np.eye(4)
        local[:3, :3] = joint_rotation_from_frame(joint, frame).as_matrix()
        local[:3, 3] = joint_position_from_frame(joint, frame)
        transforms[joint.name] = local if joint.parent is None else transforms[joint.parent] @ local
    return transforms


def normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        if fallback is None:
            raise ValueError("cannot normalize near-zero vector")
        return normalize(fallback)
    return vec / norm


def weighted_kabsch(q_points: np.ndarray, p_points: np.ndarray, weights: np.ndarray) -> tuple[Rotation, np.ndarray]:
    """Fit a rigid transform from target-local anchor points q to source points p."""

    w = weights / np.sum(weights)
    q_centroid = np.sum(q_points * w[:, None], axis=0)
    p_centroid = np.sum(p_points * w[:, None], axis=0)
    q_centered = q_points - q_centroid
    p_centered = p_points - p_centroid
    u, _s, vt = np.linalg.svd((q_centered * w[:, None]).T @ p_centered)
    rot_matrix = vt.T @ u.T
    if np.linalg.det(rot_matrix) < 0:
        vt[-1, :] *= -1.0
        rot_matrix = vt.T @ u.T
    return Rotation.from_matrix(rot_matrix), p_centroid - rot_matrix @ q_centroid


def subtree_anchor_positions_from_spine1(target_bvh: BVH, frame: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate target anchors in a temporary coordinate system where Spine1 is root."""

    transforms: dict[str, np.ndarray] = {"Spine1": np.eye(4)}
    positions: dict[str, np.ndarray] = {}
    start_index = target_bvh.joint_names().index("Spine1") + 1
    for joint in target_bvh.joints[start_index:]:
        if joint.parent not in transforms:
            continue
        local = np.eye(4)
        local[:3, :3] = joint_rotation_from_frame(joint, frame).as_matrix()
        local[:3, 3] = joint_position_from_frame(joint, frame)
        transforms[joint.name] = transforms[joint.parent] @ local
        if joint.name in ANCHOR_WEIGHTS:
            positions[joint.name] = transforms[joint.name][:3, 3].copy()
    return positions


def solve_chest_rotation(source_bvh: BVH, target_bvh: BVH, source_frame: np.ndarray, target_frame: np.ndarray) -> Rotation:
    """Return the desired target chest orientation relative to source Hips."""

    source_fk = fk(source_bvh, source_frame)
    anchors = list(ANCHOR_WEIGHTS)
    p_points = np.array([source_fk[name][:3, 3] for name in anchors])
    q_positions = subtree_anchor_positions_from_spine1(target_bvh, target_frame)
    q_points = np.array([q_positions[name] for name in anchors])
    weights = np.array([ANCHOR_WEIGHTS[name] for name in anchors], dtype=float)
    r_star, _t_star = weighted_kabsch(q_points, p_points, weights)
    r_hips = Rotation.from_matrix(source_fk["Hips"][:3, :3])
    return r_hips.inv() * r_star


def rotation_angle_degrees(rotation: Rotation) -> float:
    return float(np.linalg.norm(rotation.as_rotvec()) * 180.0 / math.pi)


def sample_polyline(points: list[np.ndarray], distance: float) -> np.ndarray:
    if distance <= 0.0:
        return points[0].copy()
    remaining = distance
    for start, end in zip(points, points[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-12:
            continue
        if remaining <= length:
            return start + segment * (remaining / length)
        remaining -= length
    return points[-1].copy()


def reduced_torso_ratios(target_bvh: BVH, template_frame: np.ndarray) -> tuple[float, float]:
    """Read Motive 2-spine Hips->Spine and Hips->Spine1 ratios from template."""

    spine_len = float(np.linalg.norm(joint_position_from_frame(target_bvh.joint_map["Spine"], template_frame)))
    spine1_len = float(np.linalg.norm(joint_position_from_frame(target_bvh.joint_map["Spine1"], template_frame)))
    neck_len = float(np.linalg.norm(joint_position_from_frame(target_bvh.joint_map["Neck"], template_frame)))
    total = spine_len + spine1_len + neck_len
    if total < 1e-12:
        raise ValueError("target torso template has near-zero Hips->Neck length")
    return spine_len / total, (spine_len + spine1_len) / total


def set_joint_global_position(frame: np.ndarray, bvh: BVH, joint_name: str, global_position: np.ndarray) -> None:
    joint = bvh.joint_map[joint_name]
    if joint.parent is None:
        set_joint_position(frame, bvh, joint_name, global_position)
        return
    parent_transform = fk(bvh, frame)[joint.parent]
    parent_rotation = Rotation.from_matrix(parent_transform[:3, :3])
    local_position = parent_rotation.inv().apply(global_position - parent_transform[:3, 3])
    set_joint_position(frame, bvh, joint_name, local_position)


def set_joint_global_rotation(frame: np.ndarray, bvh: BVH, joint_name: str, global_rotation: Rotation) -> None:
    joint = bvh.joint_map[joint_name]
    if joint.parent is None:
        local_rotation = global_rotation
    else:
        parent_rotation = Rotation.from_matrix(fk(bvh, frame)[joint.parent][:3, :3])
        local_rotation = parent_rotation.inv() * global_rotation
    set_joint_rotation(frame, bvh, joint_name, local_rotation)


def set_reduced_spine_positions(source_bvh: BVH, target_bvh: BVH, source_frame: np.ndarray, target_frame: np.ndarray, template_frame: np.ndarray) -> None:
    """Place target Spine/Spine1 by sampling the source 5-spine torso centerline."""

    source_fk = fk(source_bvh, source_frame)
    points = [source_fk[name][:3, 3] for name in SOURCE_TORSO_CHAIN]
    total_length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
    if total_length < 1e-12:
        return
    spine_ratio, spine1_ratio = reduced_torso_ratios(target_bvh, template_frame)
    set_joint_global_position(target_frame, target_bvh, "Spine", sample_polyline(points, total_length * spine_ratio))
    set_joint_global_position(target_frame, target_bvh, "Spine1", sample_polyline(points, total_length * spine1_ratio))


def point_on_hips_neck_line(hips: np.ndarray, neck: np.ndarray, ratio: float) -> np.ndarray:
    return hips + (neck - hips) * ratio


def point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        raise ValueError("cannot constrain Spine because Hips and Neck are coincident")
    return float(np.linalg.norm(np.cross(direction, point - start)) / length)


def assert_spine_on_hips_neck_line(target_bvh: BVH, frame: np.ndarray) -> None:
    transforms = fk(target_bvh, frame)
    distance = point_line_distance(
        transforms["Spine"][:3, 3],
        transforms["Hips"][:3, 3],
        transforms["Neck"][:3, 3],
    )
    if distance > LINE_TOLERANCE:
        raise ValueError(f"Spine is {distance:.6g} away from the Hips->Neck line")


def set_robot_rigid_torso_positions(
    source_bvh: BVH,
    target_bvh: BVH,
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    template_frame: np.ndarray,
) -> None:
    """Place Spine on Hips->Neck line and Spine1 by centerline ratio."""

    spine_ratio, spine1_ratio = reduced_torso_ratios(target_bvh, template_frame)
    source_fk = fk(source_bvh, source_frame)
    hips = source_fk["Hips"][:3, 3]
    neck = source_fk["Neck"][:3, 3]
    set_joint_global_position(
        target_frame,
        target_bvh,
        "Spine",
        point_on_hips_neck_line(hips, neck, spine_ratio),
    )

    centerline = [source_fk[name][:3, 3] for name in SOURCE_TORSO_CHAIN]
    centerline_length = sum(float(np.linalg.norm(b - a)) for a, b in zip(centerline, centerline[1:]))
    if centerline_length >= 1e-12:
        set_joint_global_position(
            target_frame,
            target_bvh,
            "Spine1",
            sample_polyline(centerline, centerline_length * spine1_ratio),
        )


def reanchor_upper_body(source_bvh: BVH, target_bvh: BVH, source_frame: np.ndarray, target_frame: np.ndarray) -> None:
    """Keep topology-change connection joints source-aligned after reparenting."""

    source_fk = fk(source_bvh, source_frame)
    for joint_name in ("Neck", "LeftShoulder", "RightShoulder"):
        source_transform = source_fk[joint_name]
        set_joint_global_position(target_frame, target_bvh, joint_name, source_transform[:3, 3])
        set_joint_global_rotation(target_frame, target_bvh, joint_name, Rotation.from_matrix(source_transform[:3, :3]))

    source_head = source_fk["Head"]
    set_joint_global_position(target_frame, target_bvh, "Head", source_head[:3, 3])
    set_joint_global_rotation(target_frame, target_bvh, "Head", Rotation.from_matrix(source_head[:3, :3]))


def make_recommended_frame(source_bvh: BVH, target_bvh: BVH, source_frame: np.ndarray, template_frame: np.ndarray) -> np.ndarray:
    frame = np.zeros(target_bvh.total_channels, dtype=float)
    for joint in target_bvh.joints:
        if joint.name == "Hips":
            set_joint_position(frame, target_bvh, "Hips", joint_position_from_frame(source_bvh.joint_map["Hips"], source_frame))
            set_joint_rotation(frame, target_bvh, "Hips", joint_rotation_from_frame(source_bvh.joint_map["Hips"], source_frame))
            continue
        if joint.name in source_bvh.joint_map:
            set_joint_position(frame, target_bvh, joint.name, joint_position_from_frame(source_bvh.joint_map[joint.name], source_frame))
        else:
            set_joint_position(frame, target_bvh, joint.name, joint_position_from_frame(joint, template_frame))
        if joint.name in source_bvh.joint_map and joint.name not in {"Spine", "Spine1"}:
            copy_channels(frame, target_bvh, source_frame, source_bvh, joint.name, ROTATION_CHANNELS)
        else:
            set_joint_rotation(frame, target_bvh, joint.name, [0.0, 0.0, 0.0])

    # Start reduced spine translations from the standard template, then replace
    # them with source-centerline samples. This keeps the BVH schema standard but
    # avoids treating the template's fixed segment lengths as the actor geometry.
    for joint_name in ("Spine", "Spine1"):
        set_joint_position(frame, target_bvh, joint_name, joint_position_from_frame(target_bvh.joint_map[joint_name], template_frame))

    chest_rotation = solve_chest_rotation(source_bvh, target_bvh, source_frame, frame)
    set_joint_rotation(frame, target_bvh, "Spine", chest_rotation)
    set_joint_rotation(frame, target_bvh, "Spine1", Rotation.identity())
    set_reduced_spine_positions(source_bvh, target_bvh, source_frame, frame, template_frame)
    reanchor_upper_body(source_bvh, target_bvh, source_frame, frame)
    return frame


def convert_motion_to_2spine(source_bvh: BVH, target_bvh: BVH) -> np.ndarray:
    validate_bvhs(source_bvh, target_bvh)
    output = np.zeros((source_bvh.frames, target_bvh.total_channels), dtype=float)
    for frame_idx, source_frame in enumerate(source_bvh.motion):
        template_frame = target_bvh.motion[min(frame_idx, target_bvh.frames - 1)]
        output[frame_idx] = make_recommended_frame(source_bvh, target_bvh, source_frame, template_frame)
    unwrap_spine_eulers(output, target_bvh)
    return output


def make_robot_rigid_torso_frame(
    source_bvh: BVH,
    target_bvh: BVH,
    source_frame: np.ndarray,
    template_frame: np.ndarray,
) -> np.ndarray:
    frame = np.zeros(target_bvh.total_channels, dtype=float)
    for joint in target_bvh.joints:
        if joint.name == "Hips":
            set_joint_position(frame, target_bvh, "Hips", joint_position_from_frame(source_bvh.joint_map["Hips"], source_frame))
            set_joint_rotation(frame, target_bvh, "Hips", joint_rotation_from_frame(source_bvh.joint_map["Hips"], source_frame))
            continue
        if joint.name in source_bvh.joint_map:
            set_joint_position(frame, target_bvh, joint.name, joint_position_from_frame(source_bvh.joint_map[joint.name], source_frame))
        else:
            set_joint_position(frame, target_bvh, joint.name, joint_position_from_frame(joint, template_frame))
        if joint.name in source_bvh.joint_map and joint.name not in {"Spine", "Spine1"}:
            copy_channels(frame, target_bvh, source_frame, source_bvh, joint.name, ROTATION_CHANNELS)
        else:
            set_joint_rotation(frame, target_bvh, joint.name, [0.0, 0.0, 0.0])

    for joint_name in ("Spine", "Spine1"):
        set_joint_position(frame, target_bvh, joint_name, joint_position_from_frame(target_bvh.joint_map[joint_name], template_frame))

    chest_rotation = solve_chest_rotation(source_bvh, target_bvh, source_frame, frame)
    set_joint_rotation(frame, target_bvh, "Spine", chest_rotation)
    set_joint_rotation(frame, target_bvh, "Spine1", Rotation.identity())
    set_robot_rigid_torso_positions(source_bvh, target_bvh, source_frame, frame, template_frame)
    reanchor_upper_body(source_bvh, target_bvh, source_frame, frame)
    assert_spine_on_hips_neck_line(target_bvh, frame)
    return frame


def convert_motion_to_robot_rigid_torso(source_bvh: BVH, target_bvh: BVH) -> np.ndarray:
    validate_bvhs(source_bvh, target_bvh)
    output = np.zeros((source_bvh.frames, target_bvh.total_channels), dtype=float)
    for frame_idx, source_frame in enumerate(source_bvh.motion):
        template_frame = target_bvh.motion[min(frame_idx, target_bvh.frames - 1)]
        output[frame_idx] = make_robot_rigid_torso_frame(
            source_bvh,
            target_bvh,
            source_frame,
            template_frame,
        )
    unwrap_spine_eulers(output, target_bvh)
    return output


def unwrap_spine_eulers(motion: np.ndarray, target_bvh: BVH) -> None:
    """Avoid artificial +/-360 degree jumps in solved Spine Euler channels."""

    for joint_name in ("Spine", "Spine1"):
        joint = target_bvh.joint_map[joint_name]
        indices = [joint.channel_start + i for i, channel in enumerate(joint.channels) if channel in ROTATION_CHANNELS]
        motion[:, indices] = np.rad2deg(np.unwrap(np.deg2rad(motion[:, indices]), axis=0))


def is_5spine_source(bvh: BVH) -> bool:
    return all(name in bvh.joint_map for name in SOURCE_TORSO_CHAIN + tuple(ANCHOR_WEIGHTS))


def validate_bvhs(source_bvh: BVH, target_bvh: BVH) -> None:
    missing_source = [name for name in SOURCE_TORSO_CHAIN + tuple(ANCHOR_WEIGHTS) if name not in source_bvh.joint_map]
    if missing_source:
        raise ValueError(f"{source_bvh.path}: source is missing required 5-spine joints: {missing_source}")
    present_removed = sorted(REMOVED_SOURCE_JOINTS & set(target_bvh.joint_map))
    if present_removed:
        raise ValueError(f"{target_bvh.path}: target template still contains removed joints: {present_removed}")
    missing_target = [name for name in target_bvh.joint_names() if name not in source_bvh.joint_map and name not in REMOVED_SOURCE_JOINTS]
    if missing_target:
        raise ValueError(f"{source_bvh.path}: target joints missing from source: {missing_target}")
    if target_bvh.total_channels != 306:
        raise ValueError(f"{target_bvh.path}: expected 306 target channels, got {target_bvh.total_channels}")


def format_number(value: float) -> str:
    return f"{value:.10g}"


def write_bvh(path: Path, target_template: BVH, motion: np.ndarray, frame_time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in target_template.hierarchy_lines:
            handle.write(line + "\n")
        handle.write("MOTION\n")
        handle.write(f"Frames:\t{motion.shape[0]}\n")
        handle.write(f"Frame Time:\t{format_number(frame_time)}\n")
        for row in motion:
            handle.write(" ".join(format_number(float(value)) for value in row) + "\n")


def convert_file(source_path: Path, output_path: Path, target_template_path: Path = DEFAULT_TEMPLATE) -> None:
    source_bvh = parse_bvh(source_path)
    target_bvh = parse_bvh(target_template_path)
    motion = convert_motion_to_robot_rigid_torso(source_bvh, target_bvh)
    write_bvh(output_path, target_bvh, motion, source_bvh.frame_time)


def convert_directory(input_dir: Path, output_dir: Path, target_template_path: Path = DEFAULT_TEMPLATE) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    target_bvh = parse_bvh(target_template_path)
    written: list[Path] = []
    for source_path in sorted(input_dir.rglob("*.bvh")):
        source_bvh = parse_bvh(source_path)
        if not is_5spine_source(source_bvh):
            # Input directories may contain the 2-spine template or already
            # converted files. They are not source motions, so leave them out
            # silently to keep the batch command output-only.
            continue
        motion = convert_motion_to_robot_rigid_torso(source_bvh, target_bvh)
        output_path = output_dir / source_path.relative_to(input_dir)
        write_bvh(output_path, target_bvh, motion, source_bvh.frame_time)
        written.append(output_path)
    if not written:
        raise ValueError(f"no 5-spine BVH files found under {input_dir}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively convert Motive 5-spine BVH files to robot rigid-torso 2-spine BVH files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing source 5-spine .bvh files; searched recursively.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where converted 2-spine .bvh files are written with mirrored relative paths.")
    parser.add_argument("--target-template", type=Path, default=DEFAULT_TEMPLATE, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_directory(args.input_dir, args.output_dir, args.target_template)


if __name__ == "__main__":
    main()
