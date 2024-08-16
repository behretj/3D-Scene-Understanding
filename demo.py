import numpy as np
import pandas as pd
from scene_graph import SceneGraph
from preprocessing import preprocess_scan


if __name__ == "__main__":
    SCAN_DIR = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-All/d-lab-all"

    # instantiate the label mapping for Mask3D object classes (would change if using different 3D instance segmentation model)
    label_map = pd.read_csv('mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    
    preprocess_scan(SCAN_DIR, drawer_detection=True, light_switch_detection=True)

    T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")
    
    scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, unmovable=["armchair", "bookshelf", "end table", "shelf"], pose=T_ipad)
    scene_graph.build(SCAN_DIR, drawers=False)

    scene_graph.color_with_ibm_palette()

    scene_graph.remove_category("curtain")

    scene_graph.visualize(labels=True, connections=True, centroids=True)

    # to transform to Spot coordinate system:
    # scene_graph.change_coordinate_system(T_spot) # where T_spot is a 4x4 transformation matrix of the aruco marker in Spot coordinate system

    # e.g. add a lamp to a light switch: 
    # scene_graph.nodes[21].add_lamp(7)  # 7 is id of lamp, 21 is id of light switch
