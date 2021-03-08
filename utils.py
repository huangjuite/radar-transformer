import torch
import open3d as o3d
import numpy as np



def laser_to_pc(laser):

    a = np.arange(start=-120, stop=121, step=1)
    a = np.radians(a)
    x = np.cos(a) * laser
    y = np.sin(a) * laser
    z = np.ones(a.shape)
    pc = np.dstack((x, y, z))
    pc = np.squeeze(pc)

    return pc


def draw_dataset(laser, radar):
    rpc = radar.numpy()[:, 3:6]
    lpc = laser_to_pc(laser.numpy())

    rpcd = o3d.geometry.PointCloud()
    lpcd = o3d.geometry.PointCloud()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=4, origin=[0, 0, 0])

    rpcd.points = o3d.utility.Vector3dVector(rpc)
    lpcd.points = o3d.utility.Vector3dVector(lpc)

    o3d.visualization.draw_geometries([lpcd, rpcd, frame])
