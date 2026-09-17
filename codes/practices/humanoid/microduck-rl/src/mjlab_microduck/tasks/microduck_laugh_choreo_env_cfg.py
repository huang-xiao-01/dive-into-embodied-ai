"""Full laugh choreography for the arm-equipped MicroDuck model.

The legacy laugh task stays on the original 14-servo robot.  This task uses a
separate 18-servo XML with procedural arms, allowing the policy to learn the
visible sequence requested for the tutorial: a long belly-hug forward laugh,
a held backward laugh, a full supine laugh with rapid alternating hand and
foot taps, and a clean return to standing.
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


LAUGH_CHOREO_PERIOD = 10.0


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
# arms and feet perform the rapid alternating taps.
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
    # The leg values were validated against the actual left sole collision
    # geometry at the horizontal root pose.
    "left_hip_yaw": 0.1721,
    "left_hip_roll": -0.1575,
    "left_hip_pitch": 0.3445,
    "left_knee": -0.3685,
    "left_ankle": -1.4933,
    # The left palm taps while the right arm stays folded over the belly.
    # This is a small move from HUG_POSE, so the fast window remains learnable.
    "left_shoulder_pitch": 1.22,
    "left_elbow_pitch": -1.30,
}
RIGHT_TAP_POSE = {
    **HUG_POSE,
    # Counterpart validated against the right sole collision geometry.
    "right_hip_yaw": -0.2365,
    "right_hip_roll": 0.3702,
    "right_hip_pitch": 0.4368,
    "right_knee": 1.2200,
    "right_ankle": 0.9791,
    # The right palm taps while the left arm stays folded over the belly.
    "right_shoulder_pitch": 1.22,
    "right_elbow_pitch": -1.30,
}

# The base leg pose leaves both feet visibly raised; each tap pose above brings
# exactly one sole and the matching palm down.
SUPINE_POSE = {**HUG_POSE}

# Six short windows make the taps read as a fast left-right-left-right sequence.
# Each window is shared by the hand and foot contact/height terms. The small
# gaps between windows are intentional: they give the lifted side room to
# leave the floor before the other side lands.
TAP_WINDOWS = (
    (0.68, 0.701, "left"),
    (0.72, 0.741, "right"),
    (0.76, 0.781, "left"),
    (0.80, 0.821, "right"),
    (0.84, 0.861, "left"),
    (0.88, 0.901, "right"),
)

LAUGH_KEYFRAMES = (
    (0.00, HOME_POSE),
    (0.08, HUG_POSE),
    (0.20, FORWARD_POSE),
    (0.38, FORWARD_POSE),
    (0.46, BACK_POSE),
    (0.64, BACK_POSE),
    (0.66, SUPINE_POSE),
    (0.68, LEFT_TAP_POSE),
    (0.701, LEFT_TAP_POSE),
    (0.72, RIGHT_TAP_POSE),
    (0.741, RIGHT_TAP_POSE),
    (0.76, LEFT_TAP_POSE),
    (0.781, LEFT_TAP_POSE),
    (0.80, RIGHT_TAP_POSE),
    (0.821, RIGHT_TAP_POSE),
    (0.84, LEFT_TAP_POSE),
    (0.861, LEFT_TAP_POSE),
    (0.88, RIGHT_TAP_POSE),
    (0.901, RIGHT_TAP_POSE),
    (0.96, SUPINE_POSE),
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
    (0.08, 0.00),
    (0.20, 0.12),
    (0.38, 0.12),
    (0.46, -0.10),
    (0.64, -0.10),
    (0.66, -1.00),
    (0.96, -1.00),
    (0.98, 0.00),
    (1.00, 0.00),
)

# Reference renderer targets.  The root is free rather than actuated, so these
# are kept separate from the joint reward's projected-gravity proxy.
ROOT_PITCH_KEYFRAMES = (
    (0.00, 0.00),
    (0.20, 0.00),
    (0.38, 0.00),
    (0.46, -0.10),
    (0.64, -0.10),
    (0.66, -math.pi / 2.0),
    (0.96, -math.pi / 2.0),
    (0.98, 0.00),
    (1.00, 0.00),
)
ROOT_Z_KEYFRAMES = (
    (0.00, 0.120),
    (0.64, 0.120),
    (0.66, 0.055),
    (0.96, 0.055),
    (0.98, 0.120),
    (1.00, 0.120),
)

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
        weight=24.0,
        params={
            "command_name": "twist",
            "keyframes": LAUGH_KEYFRAMES,
            "std": 0.22,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_choreography_l1"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track_l1,
        weight=5.0,
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
            "tap_windows": TAP_WINDOWS,
        },
    )
    cfg.rewards["laugh_hand_height"] = RewardTermCfg(
        func=microduck_mdp.laugh_hand_height_track,
        weight=16.0,
        params={
            "command_name": "twist",
            "tap_windows": TAP_WINDOWS,
            "target_height": 0.011,
            "std": 0.035,
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
            "tap_windows": TAP_WINDOWS,
            "right_weight": 1.0,
        },
    )
    cfg.rewards["laugh_foot_height"] = RewardTermCfg(
        func=microduck_mdp.laugh_foot_height_track,
        weight=35.0,
        params={
            "command_name": "twist",
            "tap_windows": TAP_WINDOWS,
            "target_height": 0.012,
            "std": 0.025,
            "right_weight": 1.0,
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
    cfg.rewards["laugh_hand_taps"].weight = 20.0
    cfg.rewards["laugh_trunk_lean"].weight = 8.0
    cfg.rewards["laugh_trunk_lean"].params["std"] = 0.18
    cfg.rewards["laugh_root_height"] = RewardTermCfg(
        func=microduck_mdp.laugh_root_height_track,
        weight=8.0,
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
    # Ground-pick's curriculum gradually increases smoothness to -2.0.  That
    # is appropriate for a slow reach but suppresses this intentionally fast
    # alternating tap sequence, so keep the choreography's fixed weight.
    cfg.curriculum.pop("action_rate_weight", None)

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
