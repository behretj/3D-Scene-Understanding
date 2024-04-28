import open3d as o3d
import numpy as np
import pickle
from scene_graph import SceneGraph
from camera_transforms import pose_aria_pointcloud, pose_ipad_pointcloud, transform_ipad_to_aria_pointcloud

if __name__ == "__main__":
    scene_graph = SceneGraph()
    # Load the pickle file
    # with open("detection_results.pickle", "rb") as f:
    #     detection_results = pickle.load(f)
    # print(detection_results)
    
    # Find the transformation between the Aria and iPad point clouds
    T_aria = pose_aria_pointcloud("/home/tjark/Documents/aria_data/semantic-corner-1", vis_detection=True, vis_poses=True)
    T_ipad = pose_ipad_pointcloud("/home/tjark/Documents/aria_data/first_ipad_scan_semantic_corner", vis_detection=True, pcd_path="/home/tjark/Documents/growing_scene_graphs/semantic_corner_horizontal.ply")
    transformed_pcd_filename = transform_ipad_to_aria_pointcloud("/home/tjark/Documents/growing_scene_graphs/semantic_corner_horizontal.ply", T_ipad, T_aria)
    
    # Load the point clouds to visualize them
    pcd1 = o3d.io.read_point_cloud(transformed_pcd_filename)
    pcd2 = o3d.io.read_point_cloud("/home/tjark/Documents/growing_scene_graphs/aria_pointcloud.ply")
    o3d.visualization.draw_geometries([pcd1, pcd2])

    # Build the scene graph
    file_path = transformed_pcd_filename
    label_path = 'semantic_corner_horizontal-labels.txt'
    scene_graph.build(file_path, label_path, k=1)
    scene_graph.visualize(centroids=True, connections=True, scale=2, exlcude=[0])

