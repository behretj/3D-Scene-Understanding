import open3d as o3d
import numpy as np
from scene_graph import SceneGraph


def visualize_scene_graph(scene_graph, point_cloud_path):
    # Load your original point cloud
    original_pcd = o3d.io.read_point_cloud(point_cloud_path)

    cluster_mean_pcd = o3d.geometry.PointCloud()
    cluster_means = np.array([node.centroid for node in scene_graph.nodes])
    cluster_colors = np.array([node.color for node in scene_graph.nodes], dtype=np.float64) / 255.0
    # cluster_labels = [node.sem_label for node in scene_graph.nodes]
    cluster_mean_pcd.points = o3d.utility.Vector3dVector(cluster_means)
    cluster_mean_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

    # cluster_mean_pcd.scale(1.5, center=cluster_mean_pcd.get_center())  # Increase size by a factor of 1.5

    # # Create text annotations for cluster labels
    # text_annotations = []
    # for i, cluster_label in enumerate(cluster_labels):
    #     text_annotation = o3d.geometry.TriangleMesh.create_text(str(cluster_label), font_size=10, font_name="Open Sans")
    #     text_annotation.compute_vertex_normals()
    #     text_annotation.paint_uniform_color(cluster_colors[i])
    #     text_annotation.translate(cluster_means[i] + [0, 0, 0.1])
    #     text_annotations.append(text_annotation)
    
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
    scene_graph.build(file_path, label_path, k=2)
    # visualize_scene_graph(scene_graph, file_path)
    scene_graph.visualize(centroids=True, connections=True, scale=1.0, exlcude=[0, 1])

