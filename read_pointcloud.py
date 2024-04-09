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
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    with open(label_file, 'r') as f:
        labels = [int(label.strip()) for label in f.readlines()]
    return np.array(points), np.array(colors), np.array(labels)

def calculate_mean_pointcloud(points, colors, labels):
    unique_labels = np.unique(labels, axis=0)
    centroids = []
    for label in unique_labels:
        indices = np.where(labels == label)
        mean_point = np.mean(points[indices], axis=0)
        color = np.array(colors[indices])
        centroids.append((mean_point, color[0], label))
    return centroids

def visualize_scene_graph(scene_graph, point_cloud_path):
    # Load your original point cloud
    original_pcd = o3d.io.read_point_cloud(point_cloud_path)

    cluster_mean_pcd = o3d.geometry.PointCloud()
    cluster_means = [node.centroid for node in scene_graph.nodes]
    cluster_colors = [node.color for node in scene_graph.nodes]
    cluster_mean_pcd.points = o3d.utility.Vector3dVector(cluster_means)
    cluster_mean_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

    line_sets = []
    # Add cluster centroids as points to the original point cloud
    for node in scene_graph.nodes:
        for neighbor in node.neighbors:
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([node.centroid, neighbor.centroid])
            lines = [[0, 1]]
            line.lines = o3d.utility.Vector2iVector(lines)
            line_sets.append(line)

    # Visualize the updated point cloud with cluster means and connections
    o3d.visualization.draw_geometries([original_pcd, cluster_mean_pcd] + line_sets)


if __name__ == "__main__":
    scene_graph = SceneGraph()
    file_path = 'point_lifted_mesh_ascii.ply'
    label_path = 'labels.txt'
    points, colors, labels = read_ply(file_path)
    node_objs = calculate_mean_pointcloud(points, colors, labels)
    scene_graph.build_scene_graph(node_objs, k=3)
    visualize_scene_graph(scene_graph, file_path)

