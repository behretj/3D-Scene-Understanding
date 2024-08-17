import os
import argparse
import numpy as np
import open3d as o3d
from drawer_integration import register_drawers, register_light_switches
from camera_transforms import pose_ipad_pointcloud


def preprocess_scan(scan_dir, drawer_detection=False, light_switch_detection=False):
    """ runs the drawer detection on the iPad scan and overwrites detected drawers in the mask3d prediction"""
    with open(scan_dir + "/predictions.txt", 'r') as file:
        lines = file.readlines()

    pcd = o3d.io.read_point_cloud(scan_dir + "/mesh_labeled.ply")
    points = np.asarray(pcd.points)

    if drawer_detection and not os.path.exists(scan_dir + "/predictions_drawers.txt"):
        if os.path.exists(scan_dir + "/predictions_light_switches.txt"):
            with open(scan_dir + "/predictions_light_switches.txt", 'r') as file:
                light_lines = file.readlines()
        
            next_line = len(lines) + len(light_lines)
        else:
            next_line = len(lines)
        
        indices_drawers = register_drawers(scan_dir)
        
        drawer_lines=[]
        for indices_drawer in indices_drawers:
            binary_mask = np.zeros(points.shape[0])
            binary_mask[indices_drawer] = 1
            np.savetxt(scan_dir + f"/pred_mask/{next_line:03}.txt", binary_mask, fmt='%d')
            drawer_lines += [f"pred_mask/{next_line:03}.txt 25 1.0\n",]
            next_line += 1
        
        with open(scan_dir + "/predictions_drawers.txt", 'a') as file:
            file.writelines(drawer_lines)
    
    if light_switch_detection and not os.path.exists(scan_dir + "/predictions_light_switches.txt"):
        if os.path.exists(scan_dir + "/predictions_drawers.txt"):
            with open(scan_dir + "/predictions_drawers.txt", 'r') as file:
                drawer_lines = file.readlines()
            
            next_line = len(lines) + len(drawer_lines)
        else:
            next_line = len(lines)

        indices_lights = register_light_switches(scan_dir)
        
        light_lines = []
        for indices_light in indices_lights:
            binary_mask = np.zeros(points.shape[0])
            binary_mask[indices_light] = 1
            np.savetxt(scan_dir + f"/pred_mask/{next_line:03}.txt", binary_mask, fmt='%d')
            light_lines += [f"pred_mask/{next_line:03}.txt 232 1.0\n",]
            next_line += 1
    
        with open(scan_dir + "/predictions_light_switches.txt", 'a') as file:
            file.writelines(light_lines)
    
    if not os.path.exists(scan_dir + "/aruco_pose.npy"):
        T_ipad = pose_ipad_pointcloud(scan_dir)
        np.save(scan_dir + "/aruco_pose.npy", T_ipad)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Preprocess the iPad Scan.')
   parser.add_argument('--scan_dir', type=str, required=True, help='Path to the "all data" folder from the 3D iPad scan.')
   args = parser.parse_args()
   preprocess_scan(args.scan_dir)