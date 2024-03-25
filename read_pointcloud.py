import open3d as o3d
import numpy as np
from scene_graph import SceneGraph

def read_ply(file_path="point_lifted_mesh_ascii.ply", label_file='labels.txt'):
    points = []
    colors = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_start = False
        for line in lines:
            if line.startswith('end_header'):
                data_start = True
                continue
            if data_start:
                parts = line.strip().split()
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                # colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    with open(label_file, 'r') as f:
        labels = [int(label.strip()) for label in f.readlines()]
    # return np.array(points), np.array(colors), np.array(labels)
    return np.array(points), np.array(labels) 

def calculate_mean_pointcloud(points, labels):
    unique_labels = np.unique(labels, axis=0)
    centroids = []
    for label in unique_labels:
        indices = np.where(labels == label)
        mean_point = np.mean(points[indices], axis=0)
        centroids.append((mean_point, label))
    return centroids

if __name__ == "__main__":
    scene_graph = SceneGraph()
    file_path = 'point_lifted_mesh_ascii.ply'
    label_path = 'labels.txt'
    points, labels = read_ply(file_path)
    centroids = calculate_mean_pointcloud(points, labels)
    scene_graph.build_scene_graph(centroids)
    for node in scene_graph.nodes:
        print(f"Object {node.object_id}: Neighbors {[neighbor.object_id for neighbor in node.neighbors]}, label {node.label}")

