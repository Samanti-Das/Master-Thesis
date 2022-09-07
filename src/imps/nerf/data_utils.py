import os
import numpy as np
import pickle

import PIL

import llff.poses.colmap_read_model as read_model
from ..sensors import OVCamera
from .ray_utils import generate_rays
from ..colmap import utils as colmap_utils
from ..colmap.run_colmap import run_automoatic_reconstructor


def get_poses_bounds(data_dir, colmap, near=None, far=None, img_format='.png', img_prefix='frame.'):
    if colmap:
        print("Using COLMAP parameters")

        if colmap_utils.does_exist(data_dir):
            print("Found COLMAP parameters")
        else:
            if colmap_utils.is_known_params(data_dir):
                print("Camera parameters are already provided. Not running colmap")
            else:
                print("Camera parameters are not found. Running COLMAP on the provided images.")
                run_automoatic_reconstructor(data_dir) # TODO: Test this

        poses_bounds = load_colmap_poses_bounds(data_dir)

    else:
        print("Using known parameters from params.pk file")
        assert (near is not None) and (far is not None), "If --colmap is not given --near and --far must be provided"

        poses_bounds = load_ov_poses_bounds(data_dir, near, far, img_format=img_format, frame_prefix=img_prefix)

    return poses_bounds

# TODO: This is a big question: How do we normalize the poses?
def generate_data(img_list, poses_bounds, down_scale=1, scale=False):
    cam_nf = poses_bounds[:, -2:]
    cam_params = poses_bounds[:, :-2].reshape(-1, 3, 5)

    cam_intrinsics = cam_params[:, :, -1]
    cam_extrinsics = cam_params[:, :, :-1]
    h,w,f = cam_intrinsics[0]

    h = int(h // down_scale)
    w = int(w // down_scale)
    f = f / down_scale

    if scale:
        near_original = cam_nf.min()
        far_original = cam_nf.max()
        scale_factor = near_original*100

        cam_nf[..., 0] = near_original
        cam_nf[..., 1] = far_original
        
        cam_nf /= scale_factor
        cam_extrinsics[..., -1] /= scale_factor # Scale down the origin as well

    origins, viewdirs = generate_rays(cam_extrinsics, h, w, f)

    all_imgs = np.zeros((len(img_list), h, w, 3))
    for i, img_f in enumerate(img_list):
        img = PIL.Image.open(img_f)
        # PIL Image convention (width, height) and numpy image convention (height, width).
        # This makes sense since from a matrix point of view, rows represent the height.
        img.thumbnail((w, h), PIL.Image.ANTIALIAS)
        img = (np.array(img) / 255)[:, :, :-1]
        all_imgs[i] = img

    bounds = np.broadcast_to(cam_nf[:, None, None, :], (len(all_imgs), h, w, 2))

    return origins, viewdirs, all_imgs, bounds


def cams2nerf(cameras, h, w, bounds, perm=None, transpose_rot=False):
    '''
    Converts camera extrinsics and intrinsics to a file format that NeRF utilizes.
    Cameras is a list of sensors.Camera objects.
    For now assume height, width, far and near are same for every image
    transpose_rot converts the rotation matrix into a format that the original NeRF repository requires. This
    repository does not require such a thing.
    '''

    poses = np.array([cam.c2w for cam in cameras])[:, :-1, :]

    if transpose_rot:
        poses = np.concatenate([poses[:, :, 1:2], poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:4]], -1)[:, :-1, :]

    intrinsics = np.array(list(map(lambda x: np.array([h, w, x.focal_px(w)]), cameras))).reshape(-1, 3, 1)
    nerf_poses = np.concatenate([poses, intrinsics], axis=-1).reshape(-1, 15)

    if perm is not None:
        nerf_poses = nerf_poses[perm]

    poses_bounds = np.concatenate([nerf_poses, bounds], axis=-1)

    return poses_bounds

def load_ov_poses_bounds(realdir, near, far, transpose_rot=False, img_format='png', frame_prefix='frame.'):
    '''
    transpose_rot converts the rotation matrix into a format that the original NeRF repository requires. This
    # repository does not require such a thing.
    '''
    # Here the camera parameters are ordered as integer naming
    cameras = load_ov_cams(realdir)
    # Convert it to string naming, NeRF training will use ordering wrt. string
    frame_name = lambda x: f"{frame_prefix}{x}{img_format}"
    frame_names = list(map(frame_name, range(len(cameras))))
    perm = np.argsort(frame_names)

    # This assumes all near and fars and image resolutions are same. Maybe we can change it in the future
    nf = np.array([near, far])
    bounds = np.repeat(nf[None, :], len(cameras), axis=0)
 
    w, h = PIL.Image.open(os.path.join(realdir, 'images', frame_names[0])).size

    poses_bounds = cams2nerf(cameras, h, w, bounds, perm, transpose_rot=transpose_rot)

    return poses_bounds


def load_ov_cams(data_dir, fname='params.pk'):
    params = pickle.load(open(os.path.join(data_dir, fname), 'rb'))
    cams = []

    for cam_params in params['cameras']:
        cams.append(OVCamera(cam_params))

    return cams


def load_colmap_poses_bounds(realdir, transposed_rot=False):
    """
    https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py
    """
    # Here the camera parameters are ordered as string naming

    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                raise ValueError('ERROR: the correct camera poses for current points cannot be accessed')

            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)

        pose = poses[..., i]
        if not transposed_rot:
            # The depth calculation is conducted with the changed rotation ordering, keep the change but revert it after the depth calculations if
            # we want the regular rotation ordering.
            pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]], 1)

        save_arr.append(np.concatenate([pose.ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    return save_arr


def load_colmap_poses(realdir, c2w=True):    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    names = [imdata[k].name for k in imdata]
    perm = np.argsort(names)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    poses = np.stack(w2c_mats, 0)
    
    if c2w:
        poses = np.linalg.inv(poses)
    
    return poses[perm]