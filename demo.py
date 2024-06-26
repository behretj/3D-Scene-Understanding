import numpy as np
import pandas as pd
from scene_graph import SceneGraph
from camera_transforms import pose_aria_pointcloud, pose_ipad_pointcloud, transform_ipad_to_aria_pointcloud, spot_to_aria_coords

if __name__ == "__main__":

    ### add all the scenes that should be processed here
    folder_names = [
        "Easy_Ball_Both_Hands",
        "Easy_Frame_Both_Hands",
        "Easy_Frame_Left_Hand",
        "Easy_Frame_Right_Hand",
        "Easy_Pillow_Both_Hands",
        "Easy_Plant_Right_Hand",
        "Easy_WateringPot_Both_Hands",
        "Medium_Clock_Plant",
    ]

    # instantiate the label mapping for Mask3D object classes (would change if using different 3D instance segmentation model)
    label_map = pd.read_csv('mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    
    for name in folder_names:
        # Create a scene graph object with the label mapping and a minimum confidence threshold, for now the unmovable object categories are passed manually
        scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, unmovable=["armchair", "bookshelf", "end table", "shelf"])

        # Cmpute the transformation between the Aria and iPad coordinate systems
        T_aria = pose_aria_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/" + name)
        T_ipad = pose_ipad_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1")
        if T_aria is None or T_ipad is None:
            print("Skipping " + name + " due to missing transformation")
            continue
        transformed_pcd_filename = transform_ipad_to_aria_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up.ply", T_ipad, T_aria)
        
        # add instances to the scene graph based on the Mask3D predictions
        scene_graph.build_mask3d("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/predictions_mask3d_1.txt", transformed_pcd_filename)

        # Manual corrections to the scene graph for demonstration purposes
        scene_graph.label_correction()
        scene_graph.remove_category("curtain")
        scene_graph.remove_category("cabinet")
        scene_graph.remove_category("book")
        scene_graph.remove_category("doorframe")

        #### Example for adding drawers to the scene graph, still experimental
        masks = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_masks_drawers.npy")
        points = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_pcd_drawers.npy")

        handle_masks = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_masks_handles.npy")
        handle_points = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_pcd_handles.npy")

        points = spot_to_aria_coords(points, T_aria)
        handle_points = spot_to_aria_coords(handle_points, T_aria)

        for i in range(masks.shape[0]):
            mask = masks[i, :].astype(bool)
            mask_handle = handle_masks[i, :].astype(bool)
            pcd_points = points[mask]
            pcd_handle_points = handle_points[mask_handle]

            scene_graph.add_drawer(np.array(pcd_points), np.array(pcd_handle_points))

        # Change color scheme of the scene graph to IBM palette
        scene_graph.color_with_ibm_palette()

        # Uncomment the follwing line to visualize the tracked changes in a video
        # scene_graph.tracking_video("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/" + name, "/home/tjark/Documents/growing_scene_graphs/tracking_vis/Final_" + name + "_tracking.mp4")
        
        # Apply the tracked changes to the scene graph
        scene_graph.track_changes("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/" + name)        

        scene_graph.visualize()