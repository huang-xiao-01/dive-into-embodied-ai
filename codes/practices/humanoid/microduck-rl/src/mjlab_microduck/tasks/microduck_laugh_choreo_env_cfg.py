"""Full laugh choreography for the arm-equipped MicroDuck model.

The legacy laugh task stays on the original 14-servo robot.  This task uses a
separate 18-servo XML with procedural arms, allowing the policy to learn the
visible sequence requested for the tutorial: belly-hug and forward laugh,
throw the body all the way onto its back, then alternate left/right foot taps.
"""

import math
from copy import deepcopy

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_LAUGH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_laugh_env_cfg import (
    make_microduck_laugh_env_cfg,
    MicroduckLaughRlCfg,
)


LAUGH_CHOREO_PERIOD = 5.0


def _pose(**values: float) -> dict[str, float]:
    """Keep every choreography frame in one explicit, stable joint order."""

    return values


# Source pose: hands are already folded toward the belly, so the two palms do
# not touch the floor at reset.  A tap frame sets the corresponding arm to
# shoulder=elbow=0, placing its palm on the plane.
HOME_POSE = _pose(
    left_hip_yaw=0.0,
    left_hip_roll=-0.0873,
    left_hip_pitch=-0.4579,
    left_knee=-0.0049,
    left_ankle=0.4530,
    left_shoulder_pitch=0.82,
    left_elbow_pitch=-1.62,
    right_shoulder_pitch=0.82,
    right_elbow_pitch=-1.62,
    neck_pitch=0.3491,
    head_pitch=0.3491,
    head_yaw=0.0,
    head_roll=0.0,
    right_hip_yaw=0.0,
    right_hip_roll=0.0873,
    right_hip_pitch=0.4579,
    right_knee=0.0049,
    right_ankle=-0.4530,
)

# The first half reads as “捧腹前仰后倒”.  Once the trunk is horizontal, the
# arms stay folded on the belly and the feet perform the visible taps.
HUG_POSE = {**HOME_POSE, "left_shoulder_pitch": 1.05, "left_elbow_pitch": -1.85,
            "right_shoulder_pitch": 1.05, "right_elbow_pitch": -1.85}
FORWARD_POSE = {
    **HUG_POSE,
    "left_hip_pitch": -0.54,
    "left_knee": -0.13,
    "left_ankle": 0.52,
    "right_hip_pitch": 0.54,
    "right_knee": 0.13,
    "right_ankle": -0.52,
    "neck_pitch": 0.24,
    "head_pitch": 0.52,
    "head_yaw": 0.12,
    "head_roll": -0.06,
}
BACK_POSE = {
    **HUG_POSE,
    "left_hip_pitch": -0.38,
    "left_knee": 0.12,
    "left_ankle": 0.37,
    "right_hip_pitch": 0.38,
    "right_knee": -0.12,
    "right_ankle": -0.37,
    "neck_pitch": 0.43,
    "head_pitch": 0.19,
    "head_yaw": -0.12,
    "head_roll": 0.08,
}
LEFT_TAP_POSE = {
    **HUG_POSE,
    # Validated against the actual left sole collision geometry at the
    # horizontal root pose; the other leg stays in the raised supine pose.
    "left_hip_yaw": 0.1721,
    "left_hip_roll": -0.1575,
    "left_hip_pitch": 0.3445,
    "left_knee": -0.3685,
    "left_ankle": -1.4933,
}
RIGHT_TAP_POSE = {
    **HUG_POSE,
    # Counterpart validated against the right sole collision geometry.
    "right_hip_yaw": -0.2365,
    "right_hip_roll": 0.3702,
    "right_hip_pitch": 0.4368,
    "right_knee": 1.2200,
    "right_ankle": 0.9791,
}

# Keep the arms folded over the belly after the fall.  The base leg pose leaves
# both feet visibly raised; each tap pose above brings exactly one sole down.
SUPINE_POSE = {**HUG_POSE}

LAUGH_KEYFRAMES = (
    (0.00, HOME_POSE),
    (0.10, HUG_POSE),
    (0.20, FORWARD_POSE),
    (0.30, BACK_POSE),
    (0.40, SUPINE_POSE),
    (0.48, LEFT_TAP_POSE),
    (0.56, LEFT_TAP_POSE),
    (0.60, SUPINE_POSE),
    (0.64, RIGHT_TAP_POSE),
    (0.72, RIGHT_TAP_POSE),
    (0.76, SUPINE_POSE),
    (0.90, SUPINE_POSE),
    (0.98, HUG_POSE),
    (1.00, HOME_POSE),
)

ARM_JOINTS = (
    "left_shoulder_pitch",
    "left_elbow_pitch",
    "right_shoulder_pitch",
    "right_elbow_pitch",
)
ARM_KEYFRAMES = tuple(
    (phase, {name: pose[name] for name in ARM_JOINTS})
    for phase, pose in LAUGH_KEYFRAMES
)
BODY_JOINTS = tuple(name for name in HOME_POSE if name not in ARM_JOINTS)
BODY_HOME_POSE = {name: HOME_POSE[name] for name in BODY_JOINTS}
BODY_LOCK_KEYFRAMES = ((0.0, BODY_HOME_POSE), (1.0, BODY_HOME_POSE))

# Projected-gravity x targets: upright → forward laugh → full backward lay-down
# → upright recovery.  Values are dimensionless pitch proxies; -1.0 is the
# measured value at the validated -90-degree supine root orientation.
TRUNK_LEAN_KEYFRAMES = (
    (0.00, 0.00),
    (0.10, 0.00),
    (0.20, 0.12),
    (0.30, -0.10),
    (0.40, -1.00),
    (0.48, -1.00),
    (0.56, -1.00),
    (0.64, -1.00),
    (0.72, -1.00),
    (0.90, -1.00),
    (0.98, 0.00),
    (1.00, 0.00),
)

# Reference renderer targets.  The root is free rather than actuated, so these
# are kept separate from the joint reward's projected-gravity proxy.
ROOT_PITCH_KEYFRAMES = (
    (0.00, 0.00),
    (0.20, 0.00),
    (0.30, -0.10),
    (0.40, -math.pi / 2.0),
    (0.90, -math.pi / 2.0),
    (0.98, 0.00),
    (1.00, 0.00),
)
ROOT_Z_KEYFRAMES = (
    (0.00, 0.120),
    (0.30, 0.120),
    (0.40, 0.055),
    (0.90, 0.055),
    (0.98, 0.120),
    (1.00, 0.120),
)

LEFT_FOOT_TAP_START = 0.48
LEFT_FOOT_TAP_END = 0.56
RIGHT_FOOT_TAP_START = 0.64
RIGHT_FOOT_TAP_END = 0.72


def make_microduck_laugh_choreo_env_cfg(play: bool = False, rough: bool = False):
    """Create the arm-enabled, phase-conditioned laugh choreography."""

    cfg = make_microduck_laugh_env_cfg(play=play, rough=rough)
    cfg.scene.entities["robot"] = MICRODUCK_LAUGH_ROBOT_CFG

    hands_ground_cfg = ContactSensorCfg(
        name="laugh_hand_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_hand_collision|right_hand_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (hands_ground_cfg,)

    cfg.rewards.pop("laugh_pose", None)
    cfg.rewards.pop("laugh_pose_l1", None)
    # The legacy helper indexes the original 14-joint neck slice.  This model
    # inserts four arm joints, so the all-joint action-rate term below is the
    # correct regularizer for this task.
    cfg.rewards.pop("neck_action_rate_l2", None)
    cfg.rewards["laugh_choreography"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track,
        weight=14.0,
        params={
            "command_name": "twist",
            "keyframes": LAUGH_KEYFRAMES,
            "std": 0.22,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_choreography_l1"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track_l1,
        weight=3.0,
        params={
            "command_name": "twist",
            "keyframes": LAUGH_KEYFRAMES,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_arm_choreography"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track,
        weight=60.0,
        params={
            "command_name": "twist",
            "keyframes": ARM_KEYFRAMES,
            "std": 0.18,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_arm_choreography_l1"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track_l1,
        weight=20.0,
        params={
            "command_name": "twist",
            "keyframes": ARM_KEYFRAMES,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_hand_taps"] = RewardTermCfg(
        func=microduck_mdp.laugh_alternating_hand_contact_reward,
        weight=20.0,
        params={
            "sensor_name": hands_ground_cfg.name,
            "command_name": "twist",
            "left_start": LEFT_FOOT_TAP_START,
            "left_end": LEFT_FOOT_TAP_END,
            "right_start": RIGHT_FOOT_TAP_START,
            "right_end": RIGHT_FOOT_TAP_END,
        },
    )
    cfg.rewards["laugh_hand_height"] = RewardTermCfg(
        func=microduck_mdp.laugh_hand_height_track,
        weight=30.0,
        params={
            "command_name": "twist",
            "left_start": LEFT_FOOT_TAP_START,
            "left_end": LEFT_FOOT_TAP_END,
            "right_start": RIGHT_FOOT_TAP_START,
            "right_end": RIGHT_FOOT_TAP_END,
            "target_height": 0.011,
            "std": 0.025,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_hand", "right_hand")
            ),
        },
    )
    cfg.rewards["laugh_foot_taps"] = RewardTermCfg(
        func=microduck_mdp.laugh_alternating_foot_contact_reward,
        weight=28.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "left_start": LEFT_FOOT_TAP_START,
            "left_end": LEFT_FOOT_TAP_END,
            "right_start": RIGHT_FOOT_TAP_START,
            "right_end": RIGHT_FOOT_TAP_END,
            "right_weight": 2.0,
        },
    )
    cfg.rewards["laugh_foot_height"] = RewardTermCfg(
        func=microduck_mdp.laugh_foot_height_track,
        weight=35.0,
        params={
            "command_name": "twist",
            "left_start": LEFT_FOOT_TAP_START,
            "left_end": LEFT_FOOT_TAP_END,
            "right_start": RIGHT_FOOT_TAP_START,
            "right_end": RIGHT_FOOT_TAP_END,
            "target_height": 0.012,
            "std": 0.025,
            "right_weight": 2.0,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
        },
    )
    cfg.rewards["laugh_trunk_lean"] = RewardTermCfg(
        func=microduck_mdp.laugh_trunk_lean_track,
        weight=2.5,
        params={
            "command_name": "twist",
            "keyframes": TRUNK_LEAN_KEYFRAMES,
            "std": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # This gesture deliberately enters the orientation that velocity tasks
    # call a fall.  Standing-specific terms would fight the choreography, so
    # disable them for the full-lie task and rely on the phase targets below.
    cfg.terminations.pop("fell_over", None)
    cfg.rewards["upright"].weight = 0.0
    cfg.rewards["feet_grounded"].weight = 0.0
    cfg.rewards["feet_flat"].weight = 0.0
    cfg.rewards["laugh_hand_taps"].weight = 0.0
    cfg.rewards["laugh_hand_height"].weight = 0.0
    cfg.rewards["laugh_trunk_lean"].weight = 8.0
    cfg.rewards["laugh_trunk_lean"].params["std"] = 0.18
    cfg.rewards["laugh_root_height"] = RewardTermCfg(
        func=microduck_mdp.laugh_root_height_track,
        weight=5.0,
        params={
            "command_name": "twist",
            "keyframes": ROOT_Z_KEYFRAMES,
            "std": 0.025,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["action_rate_l2"].weight = -0.40 if not play else -0.55
    cfg.rewards["joint_torques_l2"].weight = -2e-3
    cfg.rewards["self_collisions"].weight = -1.5

    command = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": LAUGH_CHOREO_PERIOD,
            "randomize_phase": False,
        }
    )
    cfg.viewer.body_name = "trunk_base"
    return cfg


MicroduckLaughChoreoRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckLaughRlCfg)
MicroduckLaughChoreoRlCfg.experiment_name = "laugh_choreo"
MicroduckLaughChoreoRlCfg.run_name = "laugh_choreo"


def make_microduck_laugh_arm_stage_env_cfg(play: bool = False, rough: bool = False):
    """Bootstrap arm contacts while the lower body tracks a fixed home pose."""

    cfg = make_microduck_laugh_choreo_env_cfg(play=play, rough=rough)
    cfg.rewards.pop("laugh_choreography", None)
    cfg.rewards.pop("laugh_choreography_l1", None)
    cfg.rewards.pop("laugh_trunk_lean", None)
    cfg.rewards["laugh_body_lock"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track,
        weight=35.0,
        params={
            "command_name": "twist",
            "keyframes": BODY_LOCK_KEYFRAMES,
            "std": 0.20,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_arm_choreography"].weight = 80.0
    cfg.rewards["laugh_arm_choreography_l1"].weight = 20.0
    cfg.rewards["laugh_hand_taps"].weight = 40.0
    cfg.rewards["laugh_hand_height"].weight = 50.0
    cfg.rewards["upright"].weight = 3.0
    cfg.rewards["feet_grounded"].weight = 2.0
    cfg.rewards["feet_flat"].weight = -1.0
    cfg.rewards["action_rate_l2"].weight = -0.25 if not play else -0.35
    return cfg


MicroduckLaughArmStageRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(
    MicroduckLaughChoreoRlCfg
)
MicroduckLaughArmStageRlCfg.experiment_name = "laugh_arm_stage"
MicroduckLaughArmStageRlCfg.run_name = "laugh_arm_stage"
