import pickle
import time

import open3d as o3d


if __name__ == '__main__':
    hsheets = pickle.load(open('../data/ndf_steps.pkl', 'rb'))
    #mesh = o3d.io.read_triangle_mesh('../data/ndf/mesh.ply')

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('./ScreenCamera_2021-11-10-10-57-27.json')
    # vis.add_geometry(mesh)

    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(hsheets[0])

    vis.add_geometry(geometry)
    ctr.convert_from_pinhole_camera_parameters(param)

    for s in hsheets:
        ctr.convert_from_pinhole_camera_parameters(param)
        geometry.points = o3d.utility.Vector3dVector(s[s[:,2] < 1.5/8])
        vis.update_geometry(geometry)
        time.sleep(1/5)
        vis.poll_events()
        vis.update_renderer()

    vis.run()
    vis.destroy_window()
