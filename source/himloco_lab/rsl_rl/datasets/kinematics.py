"""RobotLoad forward kinematics shared by expert and policy AMP states."""

import torch


def foot_positions(joint_pos: torch.Tensor, params: dict) -> torch.Tensor:
    angles = joint_pos.reshape(-1, 4, 3)
    hip, thigh, calf = angles.unbind(-1)
    offsets = {
        name: torch.as_tensor(value, dtype=joint_pos.dtype, device=joint_pos.device)
        for name, value in params.items()
    }

    def rotate_y(angle, vector):
        coord_x, coord_y, coord_z = vector.unbind(-1)
        return torch.stack((
            angle.cos() * coord_x + angle.sin() * coord_z,
            coord_y.expand_as(angle),
            -angle.sin() * coord_x + angle.cos() * coord_z,
        ), dim=-1)

    leg = offsets["hfe_offsets"] + rotate_y(
        thigh, offsets["kfe_offsets"] + rotate_y(calf, offsets["foot_offsets"])
    )
    coord_x, coord_y, coord_z = leg.unbind(-1)
    rotated = torch.stack((
        coord_x,
        hip.cos() * coord_y - hip.sin() * coord_z,
        hip.sin() * coord_y + hip.cos() * coord_z,
    ), dim=-1)
    return (offsets["hip_offsets"] + rotated).reshape(-1, 12)
