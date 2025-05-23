import torch
import math

def clampExtrema(old_f, new_f, vx, vy, vz, dt):
# Move field f according to x and y velocities (u and v) using an implicit Euler integrator.
    zres, yres, xres = old_f.shape
    cell_zs, cell_ys, cell_xs = torch.meshgrid(torch.arange(zres), torch.arange(yres), torch.arange(xres))
    cell_zs, cell_ys, cell_xs = cell_zs.float().cuda(), cell_ys.float().cuda(), cell_xs.float().cuda()
    center_xs = (cell_xs - vx*dt).flatten()
    center_ys = (cell_ys - vy*dt).flatten()
    center_zs = (cell_zs - vz*dt).flatten()

    # Compute indices of source cells.
    x = torch.floor(center_xs).long()
    y = torch.floor(center_ys).long()
    z = torch.floor(center_zs).long()
    xw = center_xs - x.float()
    yw = center_ys - y.float()
    zw = center_zs - z.float()
    x0 = torch.remainder(x, xres)
    x1 = torch.remainder(x+1, xres)
    y0 = torch.remainder(y, yres)
    y1 = torch.remainder(y+1, yres)
    z0 = torch.remainder(z, zres)
    z1 = torch.remainder(z+1, zres)

    # cal min and max
    # min_f = max_f = old_f.flatten()
    min_f, max_f = torch.min(old_f[z0,y0,x0], old_f[z0,y0,x1]), torch.max(old_f[z0,y0,x0], old_f[z0,y0,x1])
    min_f, max_f = torch.min(min_f, old_f[z0,y1,x0]), torch.max(max_f, old_f[z0,y1,x0])
    min_f, max_f = torch.min(min_f, old_f[z0,y1,x1]), torch.max(max_f, old_f[z0,y1,x1])
    min_f, max_f = torch.min(min_f, old_f[z1,y0,x0]), torch.max(max_f, old_f[z1,y0,x0])
    min_f, max_f = torch.min(min_f, old_f[z1,y0,x1]), torch.max(max_f, old_f[z1,y0,x1])
    min_f, max_f = torch.min(min_f, old_f[z1,y1,x0]), torch.max(max_f, old_f[z1,y1,x0])
    min_f, max_f = torch.min(min_f, old_f[z1,y1,x1]), torch.max(max_f, old_f[z1,y1,x1])

    # limit new_f
    new_f = torch.min(new_f.flatten(), max_f)
    new_f = torch.max(new_f.flatten(), min_f)
    return torch.reshape(new_f, (zres, yres, xres))

def copyBorder(f):
    zres, yres, xres = f.shape
    f[:,:,0] = f[:,:,1] = 0
    f[:,:,xres-1] = f[:,:,xres-2] = 0
    f[:,0,:] = f[:,1,:] = 0
    f[:,yres-1,:] = f[:,yres-2,:] = 0
    f[0,:,:] = f[1,:,:] = 0
    f[zres-1,:,:] = f[zres-2,:,:]

# 3D
def advectSL(f, vx, vy, vz, dt):
# Move field f according to x and y velocities (u and v) using an implicit Euler integrator.
    zres, yres, xres = f.shape
    cell_zs, cell_ys, cell_xs = torch.meshgrid(torch.arange(zres), torch.arange(yres), torch.arange(xres))
    cell_zs, cell_ys, cell_xs = cell_zs.float().cuda(), cell_ys.float().cuda(), cell_xs.float().cuda()
    center_xs = (cell_xs - vx*dt).flatten()
    center_ys = (cell_ys - vy*dt).flatten()
    center_zs = (cell_zs - vz*dt).flatten()

    # Compute indices of source cells.
    x = torch.floor(center_xs).long()
    y = torch.floor(center_ys).long()
    z = torch.floor(center_zs).long()
    xw = 1 - (center_xs - x.float())
    yw = 1 - (center_ys - y.float())
    zw = 1 - (center_zs - z.float())
    x0 = torch.remainder(x, xres)
    x1 = torch.remainder(x+1, xres)
    y0 = torch.remainder(y, yres)
    y1 = torch.remainder(y+1, yres)
    z0 = torch.remainder(z, zres)
    z1 = torch.remainder(z+1, zres)

    # A linearly-weighted sum of the 8 surrounding cells.
    flat_f = zw * (yw * (xw * f[z0,y0,x0] + (1-xw) * f[z0,y0,x1]) + (1-yw) * (xw * f[z0,y1,x0] + (1-xw) * f[z0,y1,x1])) + \
             (1-zw) * (yw * (xw * f[z1,y0,x0] + (1-xw) * f[z1,y0,x1]) + (1-yw) * (xw * f[z1,y1,x0] + (1-xw) * f[z1,y1,x1]))
    return torch.reshape(flat_f, (zres, yres, xres))


def advectMacCormack(f, vx, vy, vz, dt):
    phiN = f
    # phiHatN1 = A(phiN)
    phiHatN1 = advectSL(phiN, vx, vy, vz, dt)
    # phiHatN = A^R(phiHatN1)
    phiHatN = advectSL(phiHatN1, vx, vy, vz, -1.0*dt)
    # phiN1 = phiHatN1 + (phiN - phiHatN) / 2
    phiN1 = phiHatN1 + (phiN - phiHatN) * 0.5
    copyBorder(f)
    # clamp any newly created extrema
    phiN1 = clampExtrema(phiN, phiN1, vx, vy, vz, dt)
    return phiN1

def my_advection_mac(f, v, source, dt=0.1):
    zres, yres, xres = f[0,0].shape
    vx, vy, vz = v[0,0]*xres, v[0,1]*yres, v[0,2]*zres
    return torch.max(advectMacCormack(f[0,0], vx, vy, vz, dt), source.reshape(zres, yres, xres)).reshape(f.shape)

def my_advection_sl(f, v, source, dt=0.1):
    zres, yres, xres = f[0,0].shape
    vx, vy, vz = v[0,0]*xres, v[0,1]*yres, v[0,2]*zres
    return torch.max(advectSL(f[0,0], vx, vy, vz, dt), source.reshape(zres, yres, xres)).reshape(f.shape)

def my_advection_mac_noinflow(f, v, dt=0.1):
    zres, yres, xres = f[0,0].shape
    vx, vy, vz = v[0,0], v[0,1], v[0,2]
    return advectMacCormack(f[0,0], vx, vy, vz, dt).reshape(f.shape)

def my_advection_sl_noinflow(f, v, dt=0.1):
    zres, yres, xres = f[0,0].shape
    vx, vy, vz = v[0,0]*xres, v[0,1]*yres, v[0,2]*zres
    return advectSL(f[0,0], vx, vy, vz, dt).reshape(f.shape)

def advection(f, v, source,dt=0.1):
    zres, yres, xres = f.shape
    vx, vy, vz = v[0] * xres, v[1] * yres, v[2] * zres
    return torch.add(advectMacCormack(f, vx, vy, vz, dt), source.reshape(zres, yres, xres)).reshape(f.shape)
def trainAdvection(f, v, source,dt=0.1):
    zres, yres, xres = f.shape
    vx, vy, vz = v[0] * xres, v[1] * yres, v[2] * zres
    return torch.max(advectMacCormack(f, vx, vy, vz, dt), source.reshape(zres, yres, xres)).reshape(f.shape)


def set_zero_border(field):
    new_field = field.clone()
    new_field[0, :, :] = 0.0
    new_field[-1, :, :] = 0.0
    new_field[:, 0, :] = 0.0
    new_field[:, -1, :] = 0.0
    new_field[:, :, 0] = 0.0
    new_field[:, :, -1] = 0.0

    return new_field
def set_vel_zero_border(field):
    new_field = field.clone()
    new_field[:, :, 0, :, :] = 0.0
    new_field[:, :,-1, :, :] = 0.0
    new_field[:, :, :, 0, :] = 0.0
    new_field[:, :, :, -1, :] = 0.0
    new_field[:, :, :, :, 0] = 0.0
    new_field[:, :, :, :, -1] = 0.0

    return new_field

def smoke_source(field = None, res = None):
    max_res = max(res)
    dx = 1.0 / max_res

    x_total = dx * res[2]
    z_total = dx * res[0]

    height_min = 0.05
    height_max = 0.10
    if field is None:
        field = torch.zeros(res, dtype=torch.float32)

    for z in range(res[0]):
        for y in range(int(height_min * res[1]), int(height_max * res[1]) + 1):
            for x in range(res[2]):
                x_length = x * dx - x_total * 0.5
                z_length = z * dx - z_total * 0.5
                radius = math.sqrt(x_length * x_length + z_length * z_length)

                if radius < 0.075 * x_total:
                    field[x, y, z] = 1.0

    return field




