import open3d as o3d
import numpy as np
import pandas as pd
from scene_graph import SceneGraph
from camera_transforms import pose_aria_pointcloud, pose_ipad_pointcloud, transform_ipad_to_aria_pointcloud, convert_z_up_to_y_up
from object_detection import get_first_detection, get_all_object_detections, get_hand_object_interactions, get_all_aria_hand_poses
from utils import vis_detections, get_all_images, mask3d_labels, create_video, stitch_videos
import copy

if __name__ == "__main__":
    
    folder_names = [
        "Easy_Ball_Both_Hands",
        "Easy_Frame_Both_Hands",
        "Easy_Frame_Left_Hand",
        "Easy_Frame_Right_Hand",
        "Easy_Pillow_Both_Hands",
        "Easy_Pillow_Left_Hand",
        "Easy_Pillow_Right_Hand",
        "Easy_Plant_Both_Hands",
        "Easy_Plant_Left_Hand",
        "Easy_Plant_Right_Hand",
        "Easy_WateringPot_Both_Hands",
        "Easy_WateringPot_Left_Hand",
        "Easy_WateringPot_Right_Hand",
    ]

    df = pd.read_csv('label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(df['category'].values, index=df['id']).to_dict()
    
    for name in folder_names:
        scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, unmovable=["armchair", "bookshelf", "end table", "shelf"])

        T_aria = pose_aria_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/" + name)
        T_ipad = pose_ipad_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1")
        if T_aria is None or T_ipad is None:
            print("Skipping " + name)
            continue
        transformed_pcd_filename = transform_ipad_to_aria_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up.ply", T_ipad, T_aria)
        
        
        # scene_graph.build_mask3d("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/predictions_mask3d_1.txt", "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up_transformed.ply")
        scene_graph.build_mask3d("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/predictions_mask3d_1.txt", "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up_transformed.ply")

        #### Make the scene Graph more beautiful
        scene_graph.remove_category("curtain")
        scene_graph.remove_category("cabinet")
        scene_graph.remove_category("book")
        scene_graph.remove_category("doorframe")
        scene_graph.color_with_ibm_palette()

        scene_graph.visualize()
        scene_graph.track_changes("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/" + name)
        scene_graph.visualize()




