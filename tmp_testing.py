import numpy as np
import pandas as pd
import os
from scene_graph import SceneGraph
from preprocessing import preprocess_scan
from camera_transforms import pose_aria_pointcloud, icp_alignment


if __name__ == "__main__":
    SCAN_DIR = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-All/d-lab-all"
    DATA_FOLDER = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-All"

    # instantiate the label mapping for Mask3D object classes (would change if using different 3D instance segmentation model)
    label_map = pd.read_csv('mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    
    preprocess_scan(SCAN_DIR, drawer_detection=True, light_switch_detection=True)

    # T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")
    
    # scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, unmovable=["armchair", "bookshelf", "end table", "shelf"], pose=T_ipad)
    # scene_graph.build(SCAN_DIR, drawers=False)

    # scene_graph.color_with_ibm_palette()

    # scene_graph.remove_category("curtain")

    folder_names = [
        # "ball",
        # "clock_1",
        # "clock_2",
        # "clock_3",
        # "cow_bottom_left",
        # "cow_drawer_1",
        # "cow_drawer_2",
        "cow_drawer_3",
        # "cow_drawer_4",
        # "picture_1",
        # "plants_1",
        # "plants_2",
    ]

    for name in folder_names:
        if not os.path.exists(DATA_FOLDER + "/" + name + "/mesh_labeled.ply"):
            if os.path.exists(DATA_FOLDER + "/" + name + "/aruco_pose.npy"):
                T_aria = np.load(DATA_FOLDER + "/" + name + "/aruco_pose.npy")
            else:
                T_aria = pose_aria_pointcloud(DATA_FOLDER + "/" + name)
                np.save(DATA_FOLDER + "/" + name + "/aruco_pose.npy", T_aria)
            
            T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")

            icp_alignment(SCAN_DIR, DATA_FOLDER + "/" + name, T_init=np.dot(T_aria, np.linalg.inv(T_ipad)))
        
        T_aria = np.load(DATA_FOLDER + "/" + name + "/aruco_pose.npy")
        T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")
        scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.0, unmovable=["armchair", "bookshelf", "end table", "shelf"], pose=T_aria)
        scene_graph.build(SCAN_DIR, DATA_FOLDER + "/" + name)

        scene_graph.color_with_ibm_palette()

        scene_graph.remove_category("curtain")

        # scene_graph.tracking_video(DATA_FOLDER  + "/" + name, DATA_FOLDER + "/" + name + "/tracking.mp4")
        # scene_graph.track_changes(DATA_FOLDER + "/" + name)

        scene_graph.visualize(labels=True, connections=True, centroids=True)