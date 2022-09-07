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

CAMERA_POSITIONS = np.array([
    [275, 298, -305],
    [-400, 298, -210],
    [-40, 298, -305],
    [-420, 298, 20],
    [-415, 298, 315],
    [245, 298, 310],
    [230, 298, -180],

])

CAMERA_TARGETS = np.array([
    [0, 0, 0],
    [140, 100, 60],
    [-240, 85, 20],
    [-95, 160, 20],
    [-160, 140, -10],
    [-290, 115, -10],
    [-165, 145, 265],
])

DATA_NAME = 'ov-v4'

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

    info_dict = {}
    info_dict['cameras'] = []

    for i, (pos, tar) in enumerate(zip(CAMERA_POSITIONS, CAMERA_TARGETS)):
        vpw.set_camera_position(camera_path, pos[0], pos[1], pos[2], True)
        vpw.set_camera_target(camera_path, tar[0], tar[1], tar[2], True)

        kit.update()

        # This is twice on purpose. DON'T REMOVE!!
        gt = sdh.get_groundtruth(["rgb", "depth", "depthLinear", "semanticSegmentation"], vpw)
        gt = sdh.get_groundtruth(["rgb", "depth", "depthLinear","semanticSegmentation"], vpw)

        rgb, depth, depth_linear, semantic_segmentation = gt['rgb'][..., :3], gt['depth'], gt['depthLinear'], gt['semanticSegmentation']

        im = Image.fromarray(rgb, 'RGB')

        im.save(os.path.join(img_dir, f"frame.{i}.png"))
        np.save(os.path.join(depth_dir, f"frame.{i}.depth.npy"), depth)
        np.save(os.path.join(depth_dir, f"frame.{i}.depthLinear.npy"), depth_linear)
        np.save(os.path.join(seg_dir, f"frame.{i}.seg.npy"), semantic_segmentation)

        info_dict['cameras'].append(
            sdh.get_camera_params(vpw)
        )

    pickle.dump(info_dict, open(os.path.join(data_dir, 'params.pk'), 'wb'))

    kit.stop()  # Stop Simulation
    kit.shutdown()  # Cleanup application
