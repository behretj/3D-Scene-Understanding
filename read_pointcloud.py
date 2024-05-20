import open3d as o3d
import numpy as np
import pandas as pd
from scene_graph import SceneGraph
from camera_transforms import pose_aria_pointcloud, pose_ipad_pointcloud, transform_ipad_to_aria_pointcloud, convert_z_up_to_y_up
from object_detection import get_first_detection, get_all_object_detections, get_hand_object_interactions, get_all_aria_hand_poses
from utils import vis_detections, get_all_images, mask3d_labels
import copy

if __name__ == "__main__":
    # TODO: which is the correct mapping for mask3d?

    df = pd.read_csv('label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(df['category'].values, index=df['id']).to_dict()
    
    scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2)

    # T_aria = pose_aria_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/Easy_Frame_Left_Hand", vis_detection=True, vis_poses=True)
    # T_ipad = pose_ipad_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1")
    # transformed_pcd_filename = transform_ipad_to_aria_pointcloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up.ply", T_ipad, T_aria)


    scene_graph.build_mask3d__priortize_small_objects("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/predictions_mask3d_1.txt", "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up_transformed.ply")
    scene_graph.remove_category("curtain")
    scene_graph.remove_category("cabinet")
    # # scene_graph.get_node_info()
    # print(np.linalg.norm(scene_graph.nodes[1].centroid - scene_graph.nodes[2].centroid))
    # bboxes = scene_graph.draw_bboxes()
    # # bboxes += [o3d.io.read_point_cloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/Easy_Frame_Left_Hand/aria_pointcloud.ply")]  
    # # scene_graph.visualize(centroids=True, connections=True, labels=True, optional_geometry=bboxes)
    scene_graph.visualize(centroids=True, connections=True, labels=True, scale=2.0)

    # all_contacts, T_poses = get_all_aria_hand_poses("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/Easy_Frame_Left_Hand", mode_left_right=False)
    # interactions, T_transforms = get_hand_object_interactions("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/Easy_Frame_Left_Hand")
    
    # ind1, ind2 = scene_graph.get_track_interactions(interactions, all_contacts)

    # # Convert both arrays to tuples to allow comparison along axis 1
    # all_contacts_tuples = [tuple(point) for point in all_contacts]
    # interactions_tuples = [tuple(point) for point in interactions]

    # # Find the indices of points in all_contacts that are not present in interactions
    # indices_to_keep = np.array([point not in interactions_tuples for point in all_contacts_tuples])

    # # Filter out the points from all_contacts using the indices
    # filtered_all_contacts = all_contacts[indices_to_keep]

    # distances_begin = np.array([np.linalg.norm(contact - interactions[0]) for contact in all_contacts])
    # distances_end = np.array([np.linalg.norm(contact - interactions[-1]) for contact in all_contacts])
    # i1 = np.argmax(distances_begin < 0.01)
    # i2 = len(all_contacts) - np.argmax(distances_end[::-1] < 0.01) - 1
    # all_contacts = np.concatenate((all_contacts[:i1, :], all_contacts[i2:, :]), axis=0)
    # spheres = vis_detections(all_contacts, [75.0/255.0,0.0/255.0,146.0/255.0]) # vis_detections(interactions, color=[26.0/255.0,155.0/255.0,26.0/255.0]) + 
    
    # # scene_graph.track_object_interaction_2(interactions, T_transforms)

    # _, idx = scene_graph.tree.query(all_contacts[ind1])

    # node = scene_graph.nodes[idx]
    # first_transform = np.linalg.inv(T_poses[ind1])

    # pcd = o3d.geometry.PointCloud()
    # pcd_points = node.points
    # pcd.points = o3d.utility.Vector3dVector(pcd_points)
    # pcd_color = np.array(node.color, dtype=np.float64)
    # pcd.paint_uniform_color([1,0,0])
    # pcd.transform(first_transform)

    # pcds = []
    # for i in range (ind1, ind2, ind2//20):
    #     pcd_tmp = copy.deepcopy(pcd)
    #     pcd_tmp.transform(T_poses[i])
    #     pcds.append(pcd_tmp)


    # scene_graph.visualize(centroids=True, connections=True, labels=True, optional_geometry=pcds)


