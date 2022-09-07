import omni
# used to run sample asynchronously to not block rendering thread
import asyncio
# import the python bindings to interact with lidar sensor
from omni.isaac.range_sensor import _range_sensor
# pxr usd imports used to create cube
from pxr import UsdGeom, Gf, UsdPhysics
import numpy as np

import threading
import time

from scipy.spatial.transform import Rotation as R

stage = omni.usd.get_context().get_stage()
lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
timeline = omni.timeline.get_timeline_interface()

# For now don't assume rotation rate of the LiDAR is 0
fps = 20 # of Ouster
angular_velocity = 90 # deg/sec
bottom = np.array([0,0,0,1])[None, :]

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
    while np.abs(current_rot[-1] - goal_rot) > 1e-5:
        scans += [lidarInterface.get_point_cloud_data(path)]
    
        rot_mat = R.from_euler('xyz', current_rot, degrees=True).as_matrix()
        trans = np.array(trs_xyz_att.Get())[:, None] / 100 # Convert to meters

        l2w = np.concatenate([rot_mat, trans], axis=1)
        l2w = np.concatenate([l2w, bottom], axis=0)
        xform_mats += [l2w]

        current_rot[-1] += increment
        rot_xyz_att.Set(Gf.Vec3f(current_rot[0], current_rot[1], current_rot[2]))
        await asyncio.sleep(1/fps)

    # Scan is complete roll-back to the original state
    rot_xyz_att.Set(Gf.Vec3f(initial_rot[0], initial_rot[1], initial_rot[2]))
    name = path.split("/")[-1]
    np.save("/home/cem/Desktop/"+name+"-scan.npy", np.array(scans))
    np.save("/home/cem/Desktop/"+name+"-xform.npy", np.array(xform_mats))

timeline.play()

asyncio.ensure_future(save_scan_data("/Lidar1", 0))
asyncio.ensure_future(save_scan_data("/Lidar2", -90))


# # provides the core omniverse apis
# import omni
# # used to run sample asynchronously to not block rendering thread
# import asyncio
# # import the python bindings to interact with lidar sensor
# from omni.isaac.range_sensor import _range_sensor
# # pxr usd imports used to create cube
# from pxr import UsdGeom, Gf, UsdPhysics
# import numpy as np

# stage = omni.usd.get_context().get_stage()
# lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
# timeline = omni.timeline.get_timeline_interface()

# # Create lidar prim
# result, prim = omni.kit.commands.execute(
#             "RangeSensorCreateLidar",
#             path="/Lidar",
#             parent="",
#             min_range=0.3,
#             max_range=5,
#             draw_points=True,
#             draw_lines=False,
#             horizontal_fov=360.0,
#             vertical_fov=45.0,
#             horizontal_resolution=0.4,
#             vertical_resolution=0.35,
#             rotation_rate=20.0,
#             high_lod=True,
#             yaw_offset=0.0,
#             enable_semantics=False
#         )

# trs_att = stage.GetPrimAtPath("/Lidar").GetAttribute("xformOp:translate")
# rot_att = stage.GetPrimAtPath("/Lidar").GetAttribute("xformOp:rotateXYZ")
# trs_att.Set(Gf.Vec3d(245,298,0))
# rot_att.Set(Gf.Vec3f(270,0,45))

## Scan with already available LiDAR ###

# import omni
# # used to run sample asynchronously to not block rendering thread
# import asyncio
# # import the python bindings to interact with lidar sensor
# from omni.isaac.range_sensor import _range_sensor
# # pxr usd imports used to create cube
# from pxr import UsdGeom, Gf, UsdPhysics
# import numpy as np

# stage = omni.usd.get_context().get_stage()
# lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
# timeline = omni.timeline.get_timeline_interface()

# # For now don't assume rotation rate of the LiDAR is 0
# fps = 20 # of Ouster
# angular_velocity = 45 # deg/sec

# async def get_lidar_param(path, goal_rot):
#     await asyncio.sleep(1.0)
    
#     rot_xyz_att = stage.GetPrimAtPath(path).GetAttribute("xformOp:rotateXYZ")
#     initial_z_rot = rot_xyz_att.Get()[-1]
    
#     completion_sec = np.abs(initial_z_rot - goal_rot) / angular_velocity
#     increment = angular_velocity * (1/fps)
#     if goal_rot < initial_z_rot:
#         increment *= -1
    
#     # Complete a full rotation
#     current_rot = rot_xyz_att.Get()
#     while np.abs(current_rot[-1] - goal_rot) > 1e-5:
#         print(current_z_rot)
#         current_rot[-1] += increment
#         rot_xyz_att.Set(Gf.Vec3f(current_rot[0], current_rot[1], current_rot[2]))
#         await asyncio.sleep(1/fps)
        
# timeline.play()
# asyncio.ensure_future(get_lidar_param("/Lidar", 0))
