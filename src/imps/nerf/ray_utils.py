import numpy as np

import tensorflow as tf


def generate_rays(cam_extrinsics, h, w, f, pixel_center=0.5):
    """
    cam_extrinsics: a N_image x 3 x 4 transformation tensor, camera2world
    h, w, f: height and width of the image and focal length of the camera (in pixels)
    near, far: near and far bounds of the rays
    pixel_center: where should the rays be shooted. By default it is the middle of each pixel.

    IMPORTANT: This assumes a z-up coordinate system with cameras facing +z by default.
    """
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32) + pixel_center,
        np.arange(h, dtype=np.float32) + pixel_center,
        indexing="xy")

    camera_dirs = np.stack([(x - w * 0.5) / f,
                            (y - h * 0.5) / f, 
                            np.ones_like(x)],
                           axis=-1)
    
    directions = (camera_dirs[None, :, :, None, :] * cam_extrinsics[:, None, None, :3, :3]).sum(axis=-1)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(cam_extrinsics[:, None, None, :3, -1], directions.shape)
    
    return origins, viewdirs


def sample_points_on_ray(bounds, N_sample, perturb=True):
    """
    Bounds is a tensor with shape (N_rays, 2). First column is near, second columns is far values
    If perturb is True samples uniformly from the interval within the point on rays
    Output shape is (N_rays, N_sample)
    """
    near = tf.expand_dims(bounds[:, 0], 1)
    far = tf.expand_dims(bounds[:, 1], 1)

    coarse_pts = tf.expand_dims(tf.linspace(0., 1., N_sample), 0)
    coarse_locs = near*(1-coarse_pts) + far*coarse_pts

    if perturb:
        mids = .5 * (coarse_locs[..., 1:] + coarse_locs[..., :-1])
        upper = tf.concat([mids, coarse_locs[..., -1:]], -1)
        lower = tf.concat([coarse_locs[..., :1], mids], -1)
        
        t_rand = tf.random.uniform(coarse_locs.shape)
        coarse_locs = lower + (upper - lower) * t_rand
    
    return coarse_locs
