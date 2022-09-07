import os
import asyncio
import pickle
import signal

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import omni
from omni.isaac.python_app import OmniKitHelper


NUCLEUS_IP = os.environ['NUCLEUS_IP']
SCENARIO_PATH = f'omniverse://{NUCLEUS_IP}/Projects/DT-lab/main.usd'

DATA_NAME = 'ov-v3'

CAM_ORIGIN = np.array([-160, 0, 0])
CAM_INIT = np.array([-160,  295.,  260.])
CAM_PER_LINE = 12
NUM_LINE = 8
LINE_DESCENT = 20
ORIGIN_ASCENT = 20
CAM_ROT = 2*np.pi / CAM_PER_LINE

FOCAL_LEN = 4
HORIZONTAL_APERTURE = 5
VERTICAL_APERTURE = 3.8
FOCUS_DIST = 400

# Default rendering parameters
RENDER_CONFIG = {
    "renderer": "RayTracedLighting",
    "samples_per_pixel_per_frame": 12, # Only for pathtracing.
    "headless": True,
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "width": 1280,
    "height": 720,
}


exiting = False
def handle_exit(self, *args, **kwargs):
    global exiting
    print("exiting dataset generation...")
    exiting = True


kit = OmniKitHelper(config=RENDER_CONFIG)
signal.signal(signal.SIGINT, handle_exit)
### Perform any omniverse imports here after the helper loads ###
from omni.isaac.synthetic_utils import SyntheticDataHelper, DataWriter
sdh = SyntheticDataHelper()
from omni.isaac.synthetic_utils import visualization as vis
from omni.syntheticdata import visualize, helpers


async def load_stage(path):
    await omni.usd.get_context().open_stage_async(path)

def setup_world(kit, scenario_path):
    # Load scenario
    setup_task = asyncio.ensure_future(load_stage(scenario_path))
    while not setup_task.done():
        kit.update()
    kit.setup_renderer()
    kit.update()

def create_camera(kit, name):
    from pxr import UsdGeom

    camera_rig = UsdGeom.Xformable(kit.create_prim("/World/CameraRig", "Xform"))

    camera_prim = kit.create_prim(
        f"/World/CameraRig/{name}",
        "Camera",
        attributes={
            "focusDistance": FOCUS_DIST,
            "focalLength": FOCAL_LEN,
            "horizontalAperture": HORIZONTAL_APERTURE,
            "verticalAperture": VERTICAL_APERTURE,
        },
    )

    return camera_rig, camera_prim

def rotY(th):
    R = np.array([
        [np.cos(th) , 0, np.sin(th)],
        [    0      , 1,     0     ],
        [-np.sin(th), 0, np.cos(th)]
    ])
    return R

def rotZ(ph):
    R = np.array([
        [np.cos(ph), -np.sin(ph), 0],
        [np.sin(ph),  np.cos(ph), 0],
        [    0     ,      0     , 1]
    ])
    return R


if __name__ == "__main__":
    setup_world(kit, SCENARIO_PATH)

    data_dir = f'../data/{DATA_NAME}'
    img_dir = os.path.join(data_dir, 'images')
    depth_dir = os.path.join(data_dir, 'depth')
    seg_dir = os.path.join(data_dir, 'seg')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    vpi = omni.kit.viewport.get_viewport_interface()
    vpw = vpi.get_viewport_window()
    
    while kit.is_loading():
        kit.update()

    print("Loaded scenario")

    # Load cameras
    camera_rig, camera_prim = create_camera(kit, 'Camera')
    camera_path = str(camera_prim.GetPath())

    vpw.set_active_camera(camera_path)

    kit.update()

    R_y = rotY(CAM_ROT)
    cam_pos = np.copy(CAM_INIT)
    cam_orig = np.copy(CAM_ORIGIN)

    info_dict = {}
    info_dict['n_line'] = NUM_LINE
    info_dict['line_descent'] = LINE_DESCENT
    info_dict['cam_per_line'] = CAM_PER_LINE
    info_dict['cameras'] = []

    for j in range(NUM_LINE):
        for i in range(CAM_PER_LINE):
            vpw.set_camera_position(camera_path, cam_pos[0], cam_pos[1], cam_pos[2], True)
            vpw.set_camera_target(camera_path, cam_orig[0], cam_orig[1], cam_orig[2], True)

            kit.update()

            # This is twice on purpose. DON'T REMOVE!!
            gt = sdh.get_groundtruth(["rgb", "depth", "depthLinear", "semanticSegmentation"], vpw)
            gt = sdh.get_groundtruth(["rgb", "depth", "depthLinear","semanticSegmentation"], vpw)

            rgb = gt['rgb'][..., :3]
            depth = gt['depth']
            depth_linear = gt['depthLinear']
            # depth_data = np.clip(gt["depth"], 0, 255)
            # depth_data = visualize.colorize_depth(depth_data.squeeze())[..., 0].astype(np.uint8)

            semantic_segmentation = gt['semanticSegmentation']

            im = Image.fromarray(rgb, 'RGB')
            # im_depth = Image.fromarray(depth_data, 'L')

            im.save(os.path.join(img_dir, f"frame.{i+j*CAM_PER_LINE}.png"))
            # im_depth.save(os.path.join(depth_dir, f"frame.{i+j*CAM_PER_LINE}.png"))
            np.save(os.path.join(depth_dir, f"frame.{i+j*CAM_PER_LINE}.depth.npy"), depth)
            np.save(os.path.join(depth_dir, f"frame.{i+j*CAM_PER_LINE}.depthLinear.npy"), depth_linear)
            np.save(os.path.join(seg_dir, f"frame.{i+j*CAM_PER_LINE}.seg.npy"), semantic_segmentation)

            info_dict['cameras'].append(
                sdh.get_camera_params(vpw)
            )

            cam_pos = R_y @ (cam_pos - CAM_ORIGIN) + CAM_ORIGIN

        cam_pos[1] = cam_pos[1] - LINE_DESCENT
        cam_orig[1] = cam_orig[1] + ORIGIN_ASCENT


    pickle.dump(info_dict, open(os.path.join(data_dir, 'params.pk'), 'wb'))

    kit.stop()  # Stop Simulation
    kit.shutdown()  # Cleanup application
