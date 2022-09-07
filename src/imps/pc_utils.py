import os
import pickle

import open3d as o3d
import numpy as np

from .sensors import OVCamera


def from_mv_rgb_depth(data_dir, crop_height=None, put_ceil=None, voxel_size=0.01, depth_scale=1, depth_trunc=10):
    """
    forms point clouds from multiview rgb images collected from the omniverse simulation via the script in ../script/multiview-few-cameras.py
    """
    params = pickle.load(open(os.path.join(data_dir, 'params.pk'), 'rb'))
    
    frames = list(range(7))
    pcd_combined = o3d.geometry.PointCloud()

    for f in frames:
        f_rgb_dir = os.path.join(data_dir, f'images/frame.{f}.png')
        f_depth_dir = os.path.join(data_dir, f'depth/frame.{f}.depthLinear.npy')
        cam = OVCamera(params['cameras'][f])
        
        color_raw = o3d.io.read_image(f_rgb_dir)
        depth_raw = o3d.geometry.Image(np.load(f_depth_dir).squeeze())
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 
                                                                        convert_rgb_to_intensity=False,
                                                                        depth_scale=depth_scale, depth_trunc=depth_trunc)
        
        H, W, _ = np.array(color_raw).shape

        Fx = cam.focal_px(W)
        Fy = Fx
        Cx = W / 2 - 0.5
        Cy = H / 2 - 0.5
        
        cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, Fx, Fy, Cx, Cy)
        
        # cam.w2c already converts y-up to z-up
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_intrinsic, cam.w2c)
        pcd_combined += pcd
        
    points = np.array(pcd_combined.points)
    colors = np.array(pcd_combined.colors)

    if crop_height is not None:
        non_ceil_idxs = points[:, -1] < crop_height
        points = points[non_ceil_idxs]
        colors = colors[non_ceil_idxs]
    
    if put_ceil is not None:
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        x_ = np.arange(x_min, x_max, voxel_size)
        y_ = np.arange(y_min, y_max, voxel_size)
        x, y = np.meshgrid(x_, y_, indexing='ij')
        z = np.ones((len(x_), len(y_))) * put_ceil

        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1).reshape(-1 ,3)
        ceil_colors = np.ones_like(xyz) * np.array([[100/255, 100/255, 100/255]])

        points = np.concatenate([points, xyz], axis=0)
        colors = np.concatenate([colors, ceil_colors], axis=0)

    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(points)
    pcd_combined.colors = o3d.utility.Vector3dVector(colors)

    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    return pcd_combined