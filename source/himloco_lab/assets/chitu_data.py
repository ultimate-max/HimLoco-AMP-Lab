"""RobotLoad motion paths and canonical ChituMax kinematics."""

from pathlib import Path

ROBOTLOAD_ROOT = Path(__file__).resolve().parents[3] / "resources" / "robotload"
CHITU_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("LF", "LH", "RF", "RH")
    for joint in ("HAA", "HFE", "KFE")
]
CHITU_FOOT_NAMES = [f"{leg}_foot" for leg in ("LF", "LH", "RF", "RH")]
CHITU_FK_PARAMS = dict(
    hip_offsets=[[0.301, 0.09557, 0.0], [-0.301, 0.09557, 0.0],
                 [0.301, -0.09557, 0.0], [-0.301, -0.09557, 0.0]],
    hfe_offsets=[[0.0795, 0.0272, 0.0], [-0.0795, 0.0272, 0.0],
                 [0.0795, -0.0272, 0.0], [-0.0795, -0.0272, 0.0]],
    kfe_offsets=[[-0.24748, 0.12015, -0.24752], [-0.24748, 0.12015, -0.24752],
                 [-0.2475, -0.12015, -0.2475], [-0.24748, -0.12015, -0.24752]],
    foot_offsets=[[0.275, 0.0, -0.278]] * 4,
)
CHITU_MOTION_FILES = sorted(str(path) for path in
                            (ROBOTLOAD_ROOT / "datasets/chitu/mocap_motions").glob("*.txt"))
