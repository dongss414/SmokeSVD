import numpy as np
import torch
import torch.nn.functional as F


def rho(density_field, n=0, f=64, is_side = False):
    # density_field = density_field.detach().cpu().numpy()
    if is_side:
        integral = torch.cumsum(density_field, dim=2)
    else:
        integral = torch.cumsum(density_field, dim=0)

    return integral


def compute_light_source_field(density_field, light_positions, i_p, intensity_a):
    return density_field * intensity_a #light_source_field

def render(density_field,light_positions,intensity_p,intensity_a, n=0, f=64,is_side = False):
    integral_rho = rho(density_field, n, f, is_side)
    # exp_neg_rho = np.exp(-integral_rho)
    exp_neg_rho = torch.exp(torch.clip(-integral_rho, -80, 80))
    light = compute_light_source_field(density_field, light_positions, intensity_p, intensity_a)
    if is_side:
        R = torch.sum(light[n:f] * exp_neg_rho[n:f], axis=2)
    else:
        R = torch.sum(light[n:f] * exp_neg_rho[n:f], axis=0)

    return R


def render_scene(density_field,light_positions,intensity_p,intensity_a, n=0, f=64,is_side = False):
    if not is_side:
        R = render(density_field,light_positions,intensity_p,intensity_a, n, f, is_side)

    else:
        R = render(density_field, light_positions, intensity_p, intensity_a, n, f, is_side)

    return R




def render_with_y_axis_rotation_cuda(density_field, light_positions, intensity_p, intensity_a, n=0, f=64, rotation_angle=45):
    """
    Render the density field with a rotation around the y-axis.

    Parameters:
    - density_field: 3D density field tensor (shape: [D, H, W])
    - light_positions: Positions of light sources
    - intensity_p: Point light intensity
    - intensity_a: Ambient light intensity
    - rotation_angle: Angle to rotate the field in degrees (e.g., 45 for diagonal view)

    Returns:
    - R: Rendered 2D image
    """
    # Convert rotation angle to radians
    angle_rad = np.radians(rotation_angle)

    # Create a rotation matrix for the y-axis
    rotation_matrix = torch.tensor([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ], dtype=torch.float32)

    # Generate a rotated grid for the density field
    grid = generate_rotated_grid(density_field.shape, rotation_matrix).cuda()
    rotated_density_field = torch.nn.functional.grid_sample(
        density_field.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True
    ).squeeze()

    # Now render the rotated density field as usual
    integral_rho = rho(rotated_density_field, n, f)
    exp_neg_rho = torch.exp(torch.clip(-integral_rho, -80, 80))
    light = compute_light_source_field(rotated_density_field, light_positions, intensity_p, intensity_a)

    R = torch.sum(light[n:f] * exp_neg_rho[n:f], axis=0)
    return R


def generate_rotated_grid(shape, rotation_matrix):
    """
    Generate a grid for rotating the density field.
    """
    D, H, W = shape
    device = rotation_matrix.device
    grid = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack(grid, dim=-1).reshape(-1, 3)
    rotated_grid = grid @ rotation_matrix.T
    rotated_grid = rotated_grid.reshape(D, H, W, 3).unsqueeze(0)
    return rotated_grid

def only_rotated_density_field(density_field,angle):
    # angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32))
    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device=density_field.device))
    # rotation_matrix = torch.tensor([
    #     [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
    #     [0, 1, 0],
    #     [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    # ], dtype=torch.float32)
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
        [0, 1, 0],
        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    ], dtype=torch.float32, device=density_field.device)
    grid = generate_rotated_grid(density_field.shape, rotation_matrix).to(density_field.device)
    rotated_density_field = F.grid_sample(
        density_field.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True
    )
    return rotated_density_field



