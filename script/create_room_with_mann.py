import omni
from pxr import Gf, Sdf, UsdPhysics, PhysicsSchemaTools
from omni.physx.scripts import utils
from omni.isaac.range_sensor import _range_sensor

import asyncio
from scipy.spatial.transform import Rotation as R


import numpy as np

SCENE_SIZE = 10
WALL_HEIGHT = 3
WALL_THICKNESS = 0.2

N_MANN = 25

plane_size = SCENE_SIZE*100/2
wall_size_z= WALL_HEIGHT*100/2
wall_size_x = WALL_THICKNESS*100/2

NUCLEUS_IP = "10.152.4.112"
mannequin_path = f"omniverse://{NUCLEUS_IP}/Projects/DT-lab/guy-man-dude.usd"

stage = omni.usd.get_context().get_stage()
# Add a physics scene prim to stage
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
# Set gravity vector
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(981.0)

# Create ground plane
stage = omni.usd.get_context().get_stage()
PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", plane_size, Gf.Vec3f(0, 0, 0), Gf.Vec3f(1.0))
ground_prim = stage.GetPrimAtPath("/World/groundPlane")
utils.setCollider(ground_prim, approximationShape="None")

### Spawn the Room ###
# Create walls
prim = stage.DefinePrim("/World/wall0", "Cube")
UsdGeom.XformCommonAPI(prim).SetTranslate((-plane_size-WALL_THICKNESS, 0.0, wall_size_z))
UsdGeom.XformCommonAPI(prim).SetScale((wall_size_x, plane_size, wall_size_z))
utils.setCollider(prim, approximationShape="None")

prim = stage.DefinePrim("/World/wall1", "Cube")
UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, -plane_size-WALL_THICKNESS, wall_size_z))
UsdGeom.XformCommonAPI(prim).SetRotate((0.0, 0.0, 90.0))
UsdGeom.XformCommonAPI(prim).SetScale((wall_size_x, plane_size, wall_size_z))
utils.setCollider(prim, approximationShape="None")

prim = stage.DefinePrim("/World/wall2", "Cube")
UsdGeom.XformCommonAPI(prim).SetTranslate((plane_size+WALL_THICKNESS, 0.0, wall_size_z))
UsdGeom.XformCommonAPI(prim).SetRotate((0.0, 0.0, 0.0))
UsdGeom.XformCommonAPI(prim).SetScale((wall_size_x, plane_size, wall_size_z))
utils.setCollider(prim, approximationShape="None")

prim = stage.DefinePrim("/World/wall3", "Cube")
UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, plane_size+WALL_THICKNESS, wall_size_z))
UsdGeom.XformCommonAPI(prim).SetRotate((0.0, 0.0, 90.0))
UsdGeom.XformCommonAPI(prim).SetScale((wall_size_x, plane_size, wall_size_z))
utils.setCollider(prim, approximationShape="None")

### Spawn Mannequins ###
FLOOR_Z = 0
MANN_DEFAULT_HEIGHT = 120
MANN_SCALE = np.array([MANN_DEFAULT_HEIGHT, MANN_DEFAULT_HEIGHT, MANN_DEFAULT_HEIGHT])
MANN_ROTATE = np.array([0, 180, 0])

room_bound_x = [-plane_size, plane_size]
room_bound_y = [-plane_size, plane_size]

all_mann_pos = []

def create_prim_from_usd(stage, prim_env_path, prim_usd_path):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")  # create an empty Xform at the given path
    envPrim.GetReferences().AddReference(prim_usd_path)  # attach the USD to the given path
    return stage.GetPrimAtPath(envPrim.GetPath().pathString)
    
def spawn_mann(stage, prim_path, usd_path, bound_x=None, bound_y=None, scale_rand=20, rot_max=360, wall_offset=100, other_mann_offset=30):
    rotation = MANN_ROTATE
    translation = np.array([0,0,0])
    scale = MANN_SCALE + np.random.uniform(0,scale_rand,3)
    
    rotation[-1] += np.random.uniform(0,rot_max)
    is_collided = True
    check_counter = 0

    while is_collided:
        if check_counter > 1000:
            raise Exception("Can't find a place without collision. Check the offset parameters and bounds.")

        check_counter += 1

        if bound_x is not None:
            translation[0] = np.random.uniform(bound_x[0]+wall_offset, bound_x[1]-wall_offset)
        if bound_y is not None:
            translation[1] = np.random.uniform(bound_y[0]+wall_offset, bound_y[1]-wall_offset)
    
        mann_z = (scale/MANN_SCALE)[-1] * MANN_DEFAULT_HEIGHT
        translation[-1] = mann_z

        if not check_collision(translation, all_mann_pos, other_mann_offset):
            is_collided = False

    
    prim = create_prim_from_usd(stage, prim_path, usd_path)
    UsdGeom.XformCommonAPI(prim).SetTranslate(translation.tolist())
    UsdGeom.XformCommonAPI(prim).SetRotate(rotation.tolist())
    UsdGeom.XformCommonAPI(prim).SetScale(scale.tolist())
    utils.setCollider(prim, approximationShape="None")
    
    return prim

def check_collision(pos, other_manns, other_mann_offset):
    if len(other_manns) == 0:
        return False
    
    manns = np.array(other_manns)[:, :2]
    return np.any(np.linalg.norm(manns - np.array(pos[:2]).reshape(1, 2), axis=-1) < other_mann_offset)
    

for i in range(N_MANN):
    mann_prim = spawn_mann(stage, f"/World/mann{i}", mannequin_path, bound_x=room_bound_x, bound_y=room_bound_y)
    mann_pos = stage.GetPrimAtPath(mann_prim.GetPath().pathString).GetAttribute("xformOp:translate").Get()
    all_mann_pos.append(mann_pos)

### Spawn LiDAR sensors ###
lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
timeline = omni.timeline.get_timeline_interface()

# For now don't assume rotation rate of the LiDAR is 0
fps = 20 # of Ouster
angular_velocity = 90 # deg/sec
bottom = np.array([0,0,0,1])[None, :]

def spawn_lidar(stage, stage_path, position, rotation):
    # This spawns Ouster OS1-128. 

    _, _ = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=stage_path,
            parent="/World",
            min_range=0.3,
            max_range=100,
            draw_points=True,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=45.0,
            horizontal_resolution=0.4,
            vertical_resolution=0.35,
            rotation_rate=0, # For now assume rotation rate is 0
            high_lod=True,
            yaw_offset=0.0,
            enable_semantics=False
        )

    prim = stage.GetPrimAtPath("/World" + stage_path)
    trs_att = prim.GetAttribute("xformOp:translate")
    rot_att = prim.GetAttribute("xformOp:rotateXYZ")
    trs_att.Set(Gf.Vec3d(*position))
    rot_att.Set(Gf.Vec3f(*rotation))

    return prim

async def save_scan_data(path, goal_rot):
    await asyncio.sleep(1)
    
    trs_xyz_att = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate")
    rot_xyz_att = stage.GetPrimAtPath(path).GetAttribute("xformOp:rotateXYZ")

    initial_rot = rot_xyz_att.Get()
    
    increment = angular_velocity * (1/fps)
    if goal_rot < initial_rot[-1]:
        increment *= -1
    
    scans = []
    xform_mats = []
    # Complete a full rotation
    current_rot = rot_xyz_att.Get()
    while np.abs(current_rot[0] - goal_rot) > 1e-5:
        scans += [lidarInterface.get_point_cloud_data(path)]
    
        rot_mat = R.from_euler('xyz', current_rot, degrees=True).as_matrix()
        trans = np.array(trs_xyz_att.Get())[:, None] / 100 # Convert to meters

        l2w = np.concatenate([rot_mat, trans], axis=1)
        l2w = np.concatenate([l2w, bottom], axis=0)
        xform_mats += [l2w]

        current_rot[0] += increment
        rot_xyz_att.Set(Gf.Vec3f(current_rot[0], current_rot[1], current_rot[2]))
        await asyncio.sleep(1/fps)

    # Scan is complete roll-back to the original state
    rot_xyz_att.Set(Gf.Vec3f(initial_rot[0], initial_rot[1], initial_rot[2]))
    name = path.split("/")[-1]
    np.save("/home/cem/Desktop/"+name+"-scan.npy", np.array(scans))
    np.save("/home/cem/Desktop/"+name+"-xform.npy", np.array(xform_mats))

spawn_lidar(stage, '/Lidar1', [0, -plane_size+20, WALL_HEIGHT*100-20], [-22.5,0,0])
spawn_lidar(stage, '/Lidar2', [0, plane_size-20, WALL_HEIGHT*100-20], [22.5,0,0])

timeline.play()

asyncio.ensure_future(save_scan_data("/World/Lidar1", -67.5))
asyncio.ensure_future(save_scan_data("/World/Lidar2", 67.5))