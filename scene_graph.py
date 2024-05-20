import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import os
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class ObjectNode:
    def __init__(self, object_id, centroid, color, sem_label, points, confidence=None):
        self.object_id = object_id
        self.centroid = centroid
        self.points = points
        self.sem_label = sem_label
        self.color = color
        self.confidence = confidence

class SceneGraph:
    def __init__(self, label_mapping = dict(), min_confidence = 0.0,  k=2):
        self.index = 0
        self.nodes = []
        self.tree = None
        self.k = k
        self.label_mapping = label_mapping
        self.min_confidence = min_confidence

    def add_node(self, centroid, color, sem_label, points, confidence=None):
        self.nodes.append(ObjectNode(self.index, centroid, color, sem_label, points, confidence))
        self.index += 1

    def get_node_info(self):
        for node in self.nodes:
            print(f"Object ID: {node.object_id}")
            print(f"Centroid: {node.centroid}")
            print("Semantic Label: " + self.label_mapping.get(node.sem_label, "ID not found"))
            print(f"Confidence: {node.confidence}")
    
    def read_ply(self, file_path="point_lifted_mesh_ascii.ply", label_file='labels.txt'):
        pcd = o3d.io.read_point_cloud(file_path)
        with open(label_file, 'r') as f:
            labels = [int(label.strip()) for label in f.readlines()]
        return np.array(pcd.points), np.array(pcd.colors), np.array(labels)

    def build(self, file_path, label_file):
        points, colors, labels = self.read_ply(file_path, label_file)

        unique_labels = np.unique(labels, axis=0)
        for label in unique_labels:
            indices = np.where(labels == label)
            mean_point = np.mean(points[indices], axis=0)
            color = np.array(colors[indices])
            self.add_node(mean_point, color[0], label, points[indices])

        sorted_nodes = sorted(self.nodes, key=lambda node: node.object_id)

        self.tree = KDTree(np.array([node.centroid for node in sorted_nodes]))

    def build_mask3d__priortize_small_objects(self, label_path, pcd_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        file_paths = []
        values = []
        confidences = []

        for line in lines:
            parts = line.split()
            file_paths.append(parts[0])
            values.append(int(parts[1]))
            confidences.append(float(parts[2]))
        
        base_dir = os.path.dirname(os.path.abspath(label_path))
        first_pred_mask_path = os.path.join(base_dir, file_paths[0])

        with open(first_pred_mask_path, 'r') as file:
            num_lines = len(file.readlines())

        mask3d_labels = np.zeros(num_lines, dtype=np.int64)

        pcd = o3d.io.read_point_cloud(pcd_path)

        np_points = np.array(pcd.points)
        np_colors = np.array(pcd.colors)

        for i, relative_path in enumerate(file_paths):
            if confidences[i] < self.min_confidence:
                continue
            file_path = os.path.join(base_dir, relative_path)
            labels = np.loadtxt(file_path, dtype=np.int64)
            index, counts = np.unique(mask3d_labels[labels==1], return_counts=True)
            if index.shape[0] == 1:
                if index[0] == 0 or counts[0] < 10000:
                    mask3d_labels[labels==1] = values[i]
            else:
                if index[np.argmax(counts)] == 0:
                    mask3d_labels[np.logical_and(labels == 1 , mask3d_labels == 0)] = values[i]
                elif np.max(counts) < 10000 and np.max(counts)/np.sum(counts) > 0.75:
                    mask3d_labels[labels==1] = values[i]
        
        for i, relative_path in enumerate(file_paths):
            file_path = os.path.join(base_dir, relative_path)
            labels = np.loadtxt(file_path, dtype=np.int64)
            
            node_points = np_points[np.logical_and(labels == 1 , mask3d_labels == values[i])]
            colors = np_colors[np.logical_and(labels == 1 , mask3d_labels == values[i])]
            if confidences[i] > self.min_confidence and node_points.shape[0] > 0:
                mean_point = np.mean(node_points, axis=0)
                self.add_node(mean_point, colors[0], values[i], node_points, confidences[i])
        
        sorted_nodes = sorted(self.nodes, key=lambda node: node.object_id)
        
        self.tree = KDTree(np.array([node.centroid for node in sorted_nodes]))
    
    def build_mask3d(self, label_path, pcd_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        file_paths = []
        values = []
        confidences = []

        for line in lines:
            parts = line.split()
            file_paths.append(parts[0])
            values.append(int(parts[1]))
            confidences.append(float(parts[2]))
        
        base_dir = os.path.dirname(os.path.abspath(label_path))
        first_pred_mask_path = os.path.join(base_dir, file_paths[0])

        with open(first_pred_mask_path, 'r') as file:
            num_lines = len(file.readlines())

        mask3d_labels = np.zeros(num_lines, dtype=np.int64)

        pcd = o3d.io.read_point_cloud(pcd_path)

        np_points = np.array(pcd.points)
        np_colors = np.array(pcd.colors)

        for i, relative_path in enumerate(file_paths):
            file_path = os.path.join(base_dir, relative_path)
            labels = np.loadtxt(file_path, dtype=np.int64)
            node_points = np_points[labels==1] # np_points[np.logical_and(labels == 1 , mask3d_labels == 0)]
            colors = np_colors[labels==1] #  np_colors[np.logical_and(labels == 1 , mask3d_labels == 0)]
            # TODO: is this correct?
            mask3d_labels[np.logical_and(labels == 1 , mask3d_labels == 0)] = values[i]
            if confidences[i] > self.min_confidence and node_points.shape[0] > 0:
                mean_point = np.mean(node_points, axis=0)
                self.add_node(mean_point, colors[0], values[i], node_points, confidences[i])
        
        sorted_nodes = sorted(self.nodes, key=lambda node: node.object_id)
        
        self.tree = KDTree(np.array([node.centroid for node in sorted_nodes]))
    
    def get_distance(self, point):
        _, idx = self.tree.query(point)
        node = self.nodes[idx]
        return np.linalg.norm(point - node.centroid)
    
    def draw_bboxes(self):
        bboxes = []
        for node in self.nodes:
            bboxes += [self.nearest_neighbor_bbox(node.centroid)]
        return bboxes
    
    def nearest_neighbor_bbox(self, points):
        if len(points.shape) == 1:
            points = np.array([points])

        distances = np.array([self.get_distance(point) for point in points])
        index = np.argmin(distances)
        _, idx = self.tree.query(points[index])
        node = self.nodes[idx]
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(node.points))
        bbox.color = [1,0,0]
        return bbox
    
    def remove_node(self, remove_index):
        try:
            node_index = 0
            for node in self.nodes:
                if node.object_id == remove_index:
                    break
                node_index += 1
            self.nodes.pop(node_index)
            self.tree = KDTree(np.array([node.centroid for node in self.nodes]))
        except IndexError:
            print("Node not found.")

    def remove_category(self, category):
        for node in self.nodes:
            if self.label_mapping.get(node.sem_label, "ID not found") == category:
                self.remove_node(node.object_id)
    
    def track_object_interaction(self, interactions, all_contacts):
        first_interaction = interactions[0]
        last_interaction = interactions[-1]
        _, idx = self.tree.query(first_interaction)
        node = self.nodes[idx]
        distances = np.array([np.linalg.norm(point - node.centroid) for point in all_contacts])
        index = np.argmin(distances)
        self.transform(idx, last_interaction-all_contacts[index])
    
    def get_track_interactions(self, interactions, all_contacts):
        first_interaction = interactions[0]
        last_interaction = interactions[-1]
        _, idx = self.tree.query(first_interaction)
        node = self.nodes[idx]
        distances = np.array([np.linalg.norm(point - node.centroid) for point in all_contacts])
        index1 = np.argmin(distances)
        distances = np.array([np.linalg.norm(point - last_interaction) for point in all_contacts])
        index2 = np.argmin(distances)
        # self.transform(idx, last_interaction-all_contacts[index])
        return index1, index2
    
    def track_object_interaction_2(self, interactions, transforms):
        first_interaction = interactions[0]
        last_interaction = interactions[-1]
        _, idx = self.tree.query(first_interaction)
        first_transform = transforms[0]
        first_transform = np.linalg.inv(first_transform)
        last_transform = transforms[-1]
        self.transform(idx, first_transform)
        self.transform(idx, last_transform)

    
    def transform(self, idx, *args):
        """ Transform the points of a node (identified by idx) using a translation, rotation, or homogeneous transformation matrix."""
        for arg in args:
            if isinstance(arg, np.ndarray):
                if arg.shape == (3,):
                    # Apply the translation
                    self.nodes[idx].centroid += arg
                    self.nodes[idx].points += arg
                elif arg.shape == (3, 3):
                    # Apply the rotation
                    self.nodes[idx].points = np.dot(arg, self.nodes[idx].points.T).T
                    self.nodes[idx].centroid = np.dot(arg, self.nodes[idx].centroid)
                elif arg.shape == (4, 4):
                    # Apply the homogeneous transform matrix
                    self.nodes[idx].points = np.dot(arg, np.vstack((self.nodes[idx].points.T, np.ones(self.nodes[idx].points.shape[0])))).T[:, :3]
                    self.nodes[idx].centroid = np.dot(arg, np.append(self.nodes[idx].centroid, 1))[:3]
                else:
                    raise ValueError("Invalid argument shape. Expected (3,) for rotation, (3,3) for rotation, or (4,4) for homogeneous transformation.")
            else:
                raise TypeError("Invalid argument type. Expected numpy.ndarray.")

        self.tree = KDTree(np.array([node.centroid for node in self.nodes]))

    def instance_segmentation(self):
        for node in self.nodes:
            print(self.label_mapping.get(node.sem_label, "ID not found"))
            db = DBSCAN(eps=0.06, min_samples=250).fit(node.points)
            labels = db.labels_

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f'Number of clusters: {num_clusters}')

            colors = plt.get_cmap("tab20")(labels / (num_clusters if num_clusters > 0 else 1))
            colors[labels == -1] = 0  # set noise points to black

            clustered_pcd = o3d.geometry.PointCloud()
            clustered_pcd.points = o3d.utility.Vector3dVector(node.points)
            clustered_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

            o3d.visualization.draw_geometries([clustered_pcd])

    

    def visualize(self, centroids=True, connections=True, scale=0.0, labels = False, optional_geometry=None):
        geometries = []
        for node in self.nodes:
            pcd = o3d.geometry.PointCloud()
            pcd_points = node.points + scale*node.centroid
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd_color = np.array(node.color, dtype=np.float64)
            pcd.paint_uniform_color(pcd_color)
            geometries.append((pcd, "node_" + str(node.object_id)))

        if centroids:
            centroid_pcd = o3d.geometry.PointCloud()
            centroids_xyz = np.array([node.centroid + scale*node.centroid for node in self.nodes])
            centroids_colors = np.array([node.color for node in self.nodes], dtype=np.float64) / 255.0
            centroid_pcd.points = o3d.utility.Vector3dVector(centroids_xyz)
            centroid_pcd.colors = o3d.utility.Vector3dVector(centroids_colors)
            geometries.append((centroid_pcd, "centroids"))

        if connections:
            line_points = []
            line_indices = []
            idx = 0
            for node in self.nodes:
                _, indices = self.tree.query(node.centroid, k=self.k)
                for idx_neighbor in indices[1:]:
                    neighbor = self.nodes[idx_neighbor]
                    line_points.append(node.centroid + scale * node.centroid)
                    line_points.append(neighbor.centroid + scale * neighbor.centroid)
                    line_indices.append([idx, idx + 1])
                    idx += 2
            if line_points:
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(line_points),
                    lines=o3d.utility.Vector2iVector(line_indices)
                )
                line_set.paint_uniform_color([0, 0, 0])
                geometries.append((line_set, "connections"))
        
        if optional_geometry and scale == 0.0:
            if isinstance(optional_geometry, list):
                for i, geom in enumerate(optional_geometry):
                    geometries.append((geom, f"optional_geometry_{i}"))
            else:
                geometries.append((optional_geometry, "optional_geometry"))

        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("3D Visualization", 1024, 768)

        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(scene)

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"

        for geometry, name in geometries:
            scene.scene.add_geometry(name, geometry, material)

        if geometries:
            bounds = geometries[0][0].get_axis_aligned_bounding_box()
            for geometry, _ in geometries[1:]:
                bounds += geometry.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

        if labels:
            for node in self.nodes:
                label = self.label_mapping.get(node.sem_label, "ID not found")
                point = node.centroid + scale*node.centroid
                offset = np.array([0, 0, 0.1])
                scene.add_3d_label(point+offset, label)

        gui.Application.instance.run()
    
    # def visualize(self, centroids=True, connections=True, scale=0.0, exlcude=[], threshold=0, labels=False, optional_geometry=None):
    #     if threshold:
    #         nodes = [node for node in self.nodes if (node.object_id not in exlcude) and (node.points.shape[0] < threshold )]
    #     else:
    #         nodes = [node for node in self.nodes if node.object_id not in exlcude]
            
    #     geometries = []
    #     for node in nodes:
    #         pcd = o3d.geometry.PointCloud()
    #         pcd_points = node.points + scale*node.centroid
    #         pcd.points = o3d.utility.Vector3dVector(pcd_points)
    #         pcd_color = np.array(node.color, dtype=np.float64)
    #         pcd.paint_uniform_color(pcd_color)
    #         geometries.append(pcd)
        
    #     if centroids:        
    #         centroid_pcd = o3d.geometry.PointCloud()
    #         centroids_xyz = np.array([node.centroid + scale*node.centroid for node in nodes])
    #         centroids_colors = np.array([node.color for node in nodes], dtype=np.float64) / 255.0
    #         centroid_pcd.points = o3d.utility.Vector3dVector(centroids_xyz)
    #         centroid_pcd.colors = o3d.utility.Vector3dVector(centroids_colors)
    #         geometries.append(centroid_pcd)
        
    #     if connections:
    #         # Add cluster centroids as points to the original point cloud
    #         for node in nodes:
    #             _, indices = self.tree.query(node.centroid, k=self.k)
    #             for idx in indices[1:]:
    #                 neighbor = self.nodes[idx]
    #                 line = o3d.geometry.LineSet()
    #                 line.points = o3d.utility.Vector3dVector([node.centroid + scale*node.centroid, neighbor.centroid + scale*neighbor.centroid])
    #                 lines = [[0, 1]]
    #                 line.lines = o3d.utility.Vector2iVector(lines)
    #                 geometries.append(line)
        
    #     if labels:
    #         for node in nodes:
    #             label = str(node.sem_label)
    #             point = node.centroid
    #             offset = np.array([0, 0, 0.1])
    #             text_3d = o3d.visualization.gui.Label3D(label, point + offset)
    #             geometries.append(text_3d)
        
    #     if optional_geometry:
    #         o3d.visualization.draw_geometries(geometries + optional_geometry)
    #     else:
    #         o3d.visualization.draw_geometries(geometries)



