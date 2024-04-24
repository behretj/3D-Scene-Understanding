import open3d as o3d
import numpy as np
from scene_graph import SceneGraph

if __name__ == "__main__":
    scene_graph = SceneGraph()
    file_path = 'semantic_corner_point_lifted_mesh_ascii.ply'
    label_path = 'semantic_corner_labels.txt'
    # file_path = 'point_lifted_mesh_ascii.ply'
    # label_path = 'labels.txt'
    scene_graph.build(file_path, label_path, k=2)
    scene_graph.visualize(centroids=True, connections=True, scale=2, exlcude=[0])

