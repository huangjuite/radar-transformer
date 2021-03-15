import torch
import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import io
import math

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

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def laser_visual(lasers=[], show=False, range_limit=6):
    fig = plt.figure(figsize=(8, 8))
    for l in reversed(lasers):
        angle = 120
        xp = []
        yp = []
        for r in l:
            if r <= range_limit:
                yp.append(r * math.cos(math.radians(angle)))
                xp.append(r * math.sin(math.radians(angle)))
            angle -= 1
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.plot(xp, yp, 'x')
    img = get_img_from_fig(fig)
    if not show:
        plt.close()
    return img