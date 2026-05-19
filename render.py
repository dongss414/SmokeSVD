import sys

import numpy as np
from scipy.integrate import quad
import torch
# from torchvision.utils import save_image
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def rho(density_field, n=0, f=64, is_side = False):
    # density_field = density_field.detach().cpu().numpy()
    if is_side:
        integral = torch.cumsum(density_field, dim=2)
    else:
        integral = torch.cumsum(density_field, dim=0)

    return integral


def compute_light_source_field(density_field, light_positions, i_p, intensity_a):
    # depth, height, width = density_field.shape
    # light_source_field = np.zeros((depth, height, width))
    #
    # for z in range(depth):
    #     for y in range(height):
    #         for x in range(width):
    #             rho_x = density_field[z, y, x]
    #             L = intensity_a * rho_x
    #             for (x_p, y_p, z_p) in light_positions:
    #                 dist_sq = (x_p - x) ** 2 + (y_p - y) ** 2 + (z_p - z) ** 2
    #                 dist = np.sqrt(dist_sq)
    #                 integral_rho = np.sum(density_field[z_p:z + 1, y_p, x_p])
    #                 contribution = i_p * rho_x * (1 / (1 + dist)) * np.exp(-integral_rho)
    #                 L += contribution
    #
    #             light_source_field[z, y, x] = L
    # density_field = density_field.detach().cpu().numpy()
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
        # front = np.mean(density_field, axis=0)
        # R = torch.from_numpy(R)
        # m = R.max()
        # print(R.max())
        # R = torch.from_numpy(R) / m
        # front = torch.from_numpy(front) * 25.0
        # save_image(R, os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction", f"render_front.png"), nrow=1)
        # save_image(front, os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction", f"front.png"),nrow=1)
    else:
        R = render(density_field, light_positions, intensity_p, intensity_a, n, f, is_side)
        # side = np.mean(density_field, axis=2)
        # R = torch.from_numpy(R)
        # m = R.max()
        # print(R.max())
        # R = torch.from_numpy(R) / m
        # side = torch.from_numpy(side) * 25.0
        # save_image(R, os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction", f"render_side.png"), nrow=1)
        # save_image(side, os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction", f"side.png"),nrow=1)
    return R


def render_with_y_axis_rotation(density_field, light_positions, intensity_p, intensity_a, n=0, f=64, rotation_angle=45):
    """
    Render the density_ field with a rotation around the y-axis.

    Parameters:
    - density_field: 3D density_ field tensor (shape: [D, H, W])
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

    # Generate a rotated grid for the density_ field
    grid = generate_rotated_grid(density_field.shape, rotation_matrix)
    rotated_density_field = torch.nn.functional.grid_sample(
        density_field.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True
    ).squeeze()

    # Now render the rotated density_ field as usual
    integral_rho = rho(rotated_density_field, n, f)
    exp_neg_rho = torch.exp(torch.clip(-integral_rho, -80, 80))
    light = compute_light_source_field(rotated_density_field, light_positions, intensity_p, intensity_a)

    R = torch.sum(light[n:f] * exp_neg_rho[n:f], axis=0)
    return R

def render_with_y_axis_rotation_cuda(density_field, light_positions, intensity_p, intensity_a, n=0, f=64, rotation_angle=45):
    """
    Render the density_ field with a rotation around the y-axis.

    Parameters:
    - density_field: 3D density_ field tensor (shape: [D, H, W])
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

    # Generate a rotated grid for the density_ field
    grid = generate_rotated_grid(density_field.shape, rotation_matrix).cuda()
    rotated_density_field = torch.nn.functional.grid_sample(
        density_field.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True
    ).squeeze()

    # Now render the rotated density_ field as usual
    integral_rho = rho(rotated_density_field, n, f)
    exp_neg_rho = torch.exp(torch.clip(-integral_rho, -80, 80))
    light = compute_light_source_field(rotated_density_field, light_positions, intensity_p, intensity_a)

    R = torch.sum(light[n:f] * exp_neg_rho[n:f], axis=0)
    return R

def render_with_y_axis_rotation_torch(density_field, light_positions, intensity_p, intensity_a, n=0, f=64, rotation_angle=45):
    """
    Render the density_ field with a rotation around the y-axis.

    Parameters:
    - density_field: 3D density_ field tensor (shape: [D, H, W])
    - light_positions: Positions of light sources
    - intensity_p: Point light intensity
    - intensity_a: Ambient light intensity
    - rotation_angle: Angle to rotate the field in degrees (e.g., 45 for diagonal view)

    Returns:
    - R: Rendered 2D image
    """
    # Convert rotation angle to radians
    angle_rad = torch.deg2rad(torch.tensor(rotation_angle, dtype=torch.float32))

    # Create a rotation matrix for the y-axis
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
        [0, 1, 0],
        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    ], dtype=torch.float32)

    # Generate a rotated grid for the density_ field
    grid = generate_rotated_grid(density_field.shape, rotation_matrix)

    # Reshape the density_ field to match the expected shape for grid_sample (B, C, D, H, W)
    rotated_density_field = F.grid_sample(
        density_field.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True
    ).squeeze()

    # Now render the rotated density_ field as usual
    integral_rho = rho(rotated_density_field, n, f)
    exp_neg_rho = torch.exp(torch.clip(-integral_rho, -80, 80))
    light = compute_light_source_field(rotated_density_field, light_positions, intensity_p, intensity_a)

    R = torch.sum(light[n:f] * exp_neg_rho[n:f], dim=0)
    return R

def render_batch_with_y_axis_rotation(density_field, light_positions, intensity_p, intensity_a, n=0, f=64, rotation_angle=45):
    """
    Render the density_ field with a rotation around the y-axis.

    Parameters:
    - density_field: 3D density_ field tensor (shape: [B,C,D, H, W])
    - light_positions: Positions of light sources
    - intensity_p: Point light intensity
    - intensity_a: Ambient light intensity
    - rotation_angle: Angle to rotate the field in degrees (e.g., 45 for diagonal view)

    Returns:
    - R: Rendered 2D image
    """
    # Convert rotation angle to radians
    # angle_rad = np.radians(rotation_angle)
    angle_rad = torch.deg2rad(torch.tensor(rotation_angle, dtype=torch.float32))

    # Create a rotation matrix for the y-axis
    # rotation_matrix = torch.tensor([
    #     [np.cos(angle_rad), 0, np.sin(angle_rad)],
    #     [0, 1, 0],
    #     [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    # ], dtype=torch.float32)
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
        [0, 1, 0],
        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    ], dtype=torch.float32)

    B, C, D, H, W = density_field.shape
    # Generate a rotated grid for the density_ field
    grid = generate_rotated_grid((D, H, W), rotation_matrix).cuda()
    grid = grid.expand(B, *grid.shape[1:])
    rotated_density_field = torch.nn.functional.grid_sample(
        density_field, grid, mode='bilinear', align_corners=True
    )

    Rs = []
    for b in range(B):
        rotated_field = rotated_density_field[b, 0]  # [D,H,W]

        integral_rho = rho(rotated_field, n, f)
        exp_neg_rho = torch.exp(torch.clamp(-integral_rho, -80, 80))
        light = compute_light_source_field(rotated_field, light_positions, intensity_p, intensity_a)

        R = torch.sum(light[n:f] * exp_neg_rho[n:f], axis=0)  # [H,W]
        Rs.append(R)

    return torch.stack(Rs)


def generate_rotated_grid(shape, rotation_matrix):
    """
    Generate a grid for rotating the density_ field.
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

def only_rotated_batch_density_field(density_field,angle):
    B, D, H, W = density_field.shape
    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device=density_field.device))
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
        [0, 1, 0],
        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    ], dtype=torch.float32, device=density_field.device)
    grid = generate_rotated_grid((D, H, W), rotation_matrix).to(density_field.device)
    grid = grid.expand(B, *grid.shape[1:])
    rotated_density_field = F.grid_sample(
        density_field.unsqueeze(1), grid, mode='bilinear', align_corners=True
    ).squeeze(1)
    return rotated_density_field

if __name__ == '__main__':
    def resize3D(x, size):
        resized_data = F.interpolate(x, size=size, mode='trilinear',
                                     align_corners=False)
        return resized_data
    path = "/home/dongss/pycharmProjects/pinf_smoke-main/log/syn1920x1080_test/volumeout_400000/d_0100.npz"
    # path = "/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction/density_/AdvectedDensity_000100.npz"
    data = np.load(path)['vel'][:,:,:,0]
    # print(data.shape)
    data = torch.from_numpy(data).to(torch.float32)
    density = resize3D(data.unsqueeze(0).unsqueeze(0), (100, 178, 100)).cuda()#[0,0,:,:,:]
    print(density.shape)
    # sys.exit()

    light_positions = [(100, 178, -100)]
    intensity_p = 0
    intensity_a = 1

    rotation_angle = 180  # 90=front ,180=side
    scale = 0.005
    rendered_image = render_batch_with_y_axis_rotation(density * scale, light_positions, intensity_p, intensity_a, n=0, f=100,rotation_angle=rotation_angle)
    # rendered_image = torch.rot90(rendered_image.unsqueeze(0), k=2, dims=(1, 2)).squeeze(0)
    # rendered_image = F.interpolate(rendered_image.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear',
    #                                align_corners=False).squeeze(0).squeeze(0)
    print(rendered_image.shape)
    plt.imshow(rendered_image[0].cpu(),'gray')
    plt.show()
    # save_image(rendered_image,
    #            os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction", f"test.png"),
    #            nrow=1)
    # from PIL import Image
    # import torchvision.transforms as transforms
    # filepath = "/home/dongss/pycharmProjects/pinf_smoke-main/data/frame/frame_000100.png"
    # image = Image.open(filepath)
    # transform = transforms.ToTensor()
    # image_tensor = transform(image)
    # image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(256, 256), mode='bilinear',
    #                                align_corners=False).squeeze(0)
    # save_image(image_tensor,
    #            os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction", f"true.png"),
    #            nrow=1)
    # plt.imshow(rendered_image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # save_image(rendered_image, os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/Reconstruction", f"test.png"),
    #            nrow=1)
