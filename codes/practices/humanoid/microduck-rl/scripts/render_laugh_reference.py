#!/usr/bin/env python3
"""Render the designed arm-enabled MicroDuck supine laugh choreography.

This is a deterministic reference clip for reviewing the motion design while
the phase-conditioned RL policy is still being trained.  It uses the same
MJCF and keyframes as ``Mjlab-LaughChoreo-Flat-MicroDuck`` and adds only a
visual floor; no dynamics or learned policy are involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from mjlab_microduck.tasks.microduck_laugh_choreo_env_cfg import (
    LAUGH_KEYFRAMES,
    LEFT_FOOT_TAP_END,
    LEFT_FOOT_TAP_START,
    RIGHT_FOOT_TAP_END,
    RIGHT_FOOT_TAP_START,
    ROOT_PITCH_KEYFRAMES,
    ROOT_Z_KEYFRAMES,
)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _interpolate_pose(phase: float) -> dict[str, float]:
    for (lo, pose_lo), (hi, pose_hi) in zip(LAUGH_KEYFRAMES, LAUGH_KEYFRAMES[1:]):
        if lo <= phase <= hi:
            weight = _smoothstep((phase - lo) / (hi - lo))
            return {
                name: pose_lo[name] + weight * (pose_hi[name] - pose_lo[name])
                for name in pose_lo
            }
    return dict(LAUGH_KEYFRAMES[-1][1])


def _interpolate_scalar(phase: float, keyframes: tuple) -> float:
    for (lo, value_lo), (hi, value_hi) in zip(keyframes, keyframes[1:]):
        if lo <= phase <= hi:
            weight = _smoothstep((phase - lo) / (hi - lo))
            return value_lo + weight * (value_hi - value_lo)
    return float(keyframes[-1][1])


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.08)
    camera.distance = 0.48
    # A front-side view keeps both arms visible while retaining enough depth
    # to read the forward/backward body throw.
    camera.azimuth = 135.0
    camera.elevation = -8.0
    return camera


def render(output: Path, frames: int, width: int, height: int) -> dict[str, float]:
    xml = Path(__file__).resolve().parents[1] / (
        "src/mjlab_microduck/robot/microduck/robot_allcollisions_laugh.xml"
    )
    spec = mujoco.MjSpec.from_file(str(xml))
    floor = spec.worldbody.add_geom()
    floor.name = "reference_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.01]
    floor.rgba = [0.12, 0.14, 0.17, 1.0]
    model = spec.compile()
    data = mujoco.MjData(model)

    joint_ids = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )])
        for name in LAUGH_KEYFRAMES[0][1]
    }
    hand_sites = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand"),
    ]
    foot_sites = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
    ]
    foot_geoms = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"),
    ]
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "reference_floor")
    root_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root_qpos = int(model.jnt_qposadr[root_joint])
    data.qpos[:] = 0.0
    data.qpos[root_qpos : root_qpos + 3] = (0.0, 0.0, 0.12)
    data.qpos[root_qpos + 3] = 1.0

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _camera()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output, fps=50, codec="libx264", quality=8, macro_block_size=1
    )
    min_hand_z = float("inf")
    max_hand_z = float("-inf")
    min_root_z = float("inf")
    max_root_z = float("-inf")
    foot_tap_contact_frames = [0, 0]
    foot_z_min = [float("inf"), float("inf")]
    foot_z_max = [float("-inf"), float("-inf")]
    try:
        for frame_index in range(frames):
            phase = (frame_index % frames) / frames
            pose = _interpolate_pose(phase)
            for name, value in pose.items():
                data.qpos[joint_ids[name]] = value

            # The root is free in the MJCF.  The reference therefore applies
            # the validated 90-degree backward orientation directly while the
            # RL task learns the same target through projected gravity.
            root_pitch = _interpolate_scalar(phase, ROOT_PITCH_KEYFRAMES)
            data.qpos[root_qpos + 2] = _interpolate_scalar(phase, ROOT_Z_KEYFRAMES)
            data.qpos[root_qpos + 3 : root_qpos + 7] = (
                np.cos(root_pitch / 2.0), 0.0, np.sin(root_pitch / 2.0), 0.0
            )
            mujoco.mj_forward(model, data)

            hand_z = data.site_xpos[hand_sites, 2]
            foot_z = data.site_xpos[foot_sites, 2]
            min_hand_z = min(min_hand_z, float(hand_z.min()))
            max_hand_z = max(max_hand_z, float(hand_z.max()))
            for index, value in enumerate(foot_z):
                foot_z_min[index] = min(foot_z_min[index], float(value))
                foot_z_max[index] = max(foot_z_max[index], float(value))
            root_z = float(data.qpos[root_qpos + 2])
            min_root_z = min(min_root_z, root_z)
            max_root_z = max(max_root_z, root_z)
            for index, (start, end) in enumerate(
                (
                    (LEFT_FOOT_TAP_START, LEFT_FOOT_TAP_END),
                    (RIGHT_FOOT_TAP_START, RIGHT_FOOT_TAP_END),
                )
            ):
                distance = mujoco.mj_geomDistance(
                    model, data, foot_geoms[index], floor_geom, 1.0, np.zeros(6)
                )
                if start <= phase < end and distance <= 0.0:
                    foot_tap_contact_frames[index] += 1

            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    stats = {
        "frames": float(frames),
        "duration_s": frames / 50.0,
        "min_root_z_m": min_root_z,
        "max_root_z_m": max_root_z,
        "min_hand_z_m": min_hand_z,
        "max_hand_z_m": max_hand_z,
        "left_foot_min_z_m": foot_z_min[0],
        "left_foot_max_z_m": foot_z_max[0],
        "right_foot_min_z_m": foot_z_min[1],
        "right_foot_max_z_m": foot_z_max[1],
        "left_foot_tap_contact_frames": float(foot_tap_contact_frames[0]),
        "right_foot_tap_contact_frames": float(foot_tap_contact_frames[1]),
    }
    output.with_suffix(".json").write_text(
        json.dumps({"type": "kinematic_reference_supine", "stats": stats}, indent=2)
    )
    print(json.dumps(stats, indent=2))
    print(f"saved {output}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()
    render(args.output, args.frames, args.width, args.height)


if __name__ == "__main__":
    main()
