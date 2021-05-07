import math
import io
import os
import cv2
import matplotlib.pyplot as plt
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


def pc_to_laser(pc):
    max_dis = 5
    min_angle = math.radians(-120)
    max_angle = math.radians(120)
    angle_incre = math.radians(0.9999)
    start_a = min_angle
    end_a = start_a + angle_incre

    angles = np.zeros(pc.shape[0])
    for i, p in enumerate(pc):
        angles[i] = math.atan2(p[1], p[0])

    laser = []
    while start_a < max_angle:
        bundle = pc[np.where((angles > start_a) & (angles < end_a))][:, :-1]
        if len(bundle) == 0:
            d = max_dis
        else:
            bundle = np.linalg.norm(bundle, axis=1)
            d = np.min(bundle)
            d = max_dis if d > max_dis else d
        laser.append(d)
        start_a += angle_incre
        end_a += angle_incre

    laser = np.array(laser)
    return laser


def filter(radar):
    min_h = 0.15
    max_h = 2
    pcd = o3d.geometry.PointCloud()
    pt = radar[:, 3:6]
    pcd.points = o3d.utility.Vector3dVector(pt)
    cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=1.2)
    radar = radar[ind]
    radar = radar[np.where(radar[:, 5] > min_h)]
    radar = radar[np.where(radar[:, 5] < max_h)]
    return radar


def draw_laser(laser, radar):
    rpc = laser_to_pc(radar)
    lpc = laser_to_pc(laser)

    rpcd = o3d.geometry.PointCloud()
    lpcd = o3d.geometry.PointCloud()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=4, origin=[0, 0, 0])

    rpcd.points = o3d.utility.Vector3dVector(rpc)
    colors = [[1, 0, 0] for i in range(len(rpcd.points))]
    rpcd.colors = o3d.utility.Vector3dVector(colors)
    lpcd.points = o3d.utility.Vector3dVector(lpc)

    o3d.visualization.draw_geometries([lpcd, rpcd, frame])


def draw_dataset(laser, radar):
    rpc = radar[:, 3:6]
    lpc = laser_to_pc(laser)

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


def laser_visual_one(laser, axs, label=None, range_limit=4.9, m_size=5, f_size=20, marker='.'):
    angle = 120
    xp = []
    yp = []
    for r in laser:
        if r <= range_limit:
            yp.append(r * math.cos(math.radians(angle)))
            xp.append(r * math.sin(math.radians(angle)))
        angle -= 1
    axs.set_xlim([-6, 6])
    axs.set_ylim([-6, 6])
    axs.plot(xp, yp, marker, markersize=m_size, label=label)

    if label is not None:
        axs.set_title(label, fontsize=f_size)
