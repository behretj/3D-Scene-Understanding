import numpy as np
import pandas as pd
import open3d as o3d
from scene_graph import SceneGraph
from camera_transforms import pose_aria_pointcloud, pose_ipad_pointcloud, transform_ipad_to_aria_pointcloud, spot_to_aria_coords
from utils import vis_detections, get_all_images, mask3d_labels, create_video, stitch_videos

if __name__ == "__main__":

    folder_names = [
        "Easy_Ball_Both_Hands",
        # "Easy_Frame_Both_Hands",
        # "Easy_Frame_Left_Hand",
        # "Easy_Frame_Right_Hand",
        # "Easy_Pillow_Both_Hands",
        # "Easy_Plant_Right_Hand",
        # "Easy_WateringPot_Both_Hands",
        # "Medium_Clock_Plant",
        # "Easy_Plant_Both_Hands", ## hand-object-tracker failed
        # "Easy_Pillow_Left_Hand",  ## hand pose generation failed
        # "Easy_Pillow_Right_Hand", ## hand pose generation failed
        # "Easy_Plant_Left_Hand", ## hand pose generation failed
        # "Easy_WateringPot_Left_Hand", ## hand pose generation failed
        # "Easy_WateringPot_Right_Hand", ## hand pose generation failed
        # "MD_Pillow_Plant_Sim",
        # "MD_Frame_WateringPot_Sim",
        # "MD_Ball_double",
        # "drawer_top_right_cow",
        # "drawer_bottom_left_open",
        # "drawer_middle_left_open_close",
        # "drawer_top_left_open",
        # "drawer_top_left_open_2",
        # "Drawer_1",
        # "Drawer_2",
        # "Drawer_3",
    ]

    df = pd.read_csv('mask3d_label_mapping.csv', usecols=['id', 'category'])
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
        scene_graph.label_correction()
        scene_graph.remove_category("curtain")
        scene_graph.remove_category("cabinet")
        scene_graph.remove_category("book")
        scene_graph.remove_category("doorframe")

        scene_graph.color_with_ibm_palette()
        
        # scene_graph.track_changes("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/" + name)
        # scene_graph.tracking_video("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/" + name, "/home/tjark/Documents/growing_scene_graphs/tracking_vis/Final_" + name + "_tracking.mp4")

        # #### Example for adding drawers to the scene graph
        # masks = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_masks_drawers.npy")
        # points = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_pcd_drawers.npy")

        # handle_masks = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_masks_handles.npy")
        # handle_points = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_pcd_handles.npy")

        # points = spot_to_aria_coords(points, T_aria)
        # handle_points = spot_to_aria_coords(handle_points, T_aria)

        # for i in range(masks.shape[0]):
        #     mask = masks[i, :].astype(bool)
        #     mask_handle = handle_masks[i, :].astype(bool)
        #     pcd_points = points[mask]
        #     pcd_handle_points = handle_points[mask_handle]

        #     scene_graph.add_drawer(np.array(pcd_handle_points), np.array(pcd_points))


        # scene_graph.visualize(labels=True)

        

    
    # masks = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_masks_drawers.npy")
    # points = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_pcd_drawers.npy")
    # # handle_masks = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_masks_handles.npy")
    # # handle_points = np.load("SceneGraph-Dataset/iPad-Scan-1/cabinet_pcd_handles.npy")

    # roty270 = np.array([
    #     [np.cos(3*np.pi/2), 0, np.sin(3*np.pi/2), 0],
    #     [0, 1, 0, 0],
    #     [-np.sin(3*np.pi/2), 0, np.cos(3*np.pi/2), 0],
    #     [0, 0, 0, 1]
    # ])

    # rotz90 = np.array([
    #     [np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
    #     [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])

    # for i in range(masks.shape[0]):
    #     mask = masks[i, :].astype(bool)
    #     pcd_points = points[mask]
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pcd_points)
    #     pcd.paint_uniform_color(np.random.rand(3))
    #     plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
    #                                      ransac_n=3,
    #                                      num_iterations=1000)
    #     [a, b, c, d] = plane_model
    #     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    #     inlier_cloud = pcd.select_by_index(inliers)
    #     inlier_cloud.paint_uniform_color([1.0, 0, 0])
    #     o3d.visualization.draw_geometries([pcd, inlier_cloud])


