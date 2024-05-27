import numpy as np
from scipy.spatial import KDTree, ConvexHull
import open3d as o3d
import os, glob, pickle
from collections import deque
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose, get_nearest_pose
from utils import capture_image
import imageio

class ObjectNode:
    def __init__(self, object_id, centroid, color, sem_label, points, confidence=None):
        self.object_id = object_id
        self.centroid = centroid
        self.points = points
        self.sem_label = sem_label
        self.color = color
        self.confidence = confidence
        self.hull_tree = KDTree(points[ConvexHull(points).vertices])

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
    
    def track(self, scan_dir, vis=False):
        vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
        assert vrs_files is not None, "No vrs files found in directory"
        vrs_file = vrs_files[0]
        filename = os.path.splitext(os.path.basename(vrs_file))[0]

        provider = data_provider.create_vrs_data_provider(vrs_file)
        assert provider is not None, "Cannot open file"

        camera_label = "camera-rgb"
        stream_id = provider.get_stream_id_from_label(camera_label)

        detection_files = glob.glob(os.path.join(scan_dir, '*.pickle'))
        assert detection_files is not None, "No detection files found in directory"
        detection_path = detection_files[0]
        with open(detection_path, "rb") as f:
            detection_results = pickle.load(f)
        
        wrist_and_palm_poses_path = scan_dir + "/mps_" + filename + "_vrs/hand_tracking/wrist_and_palm_poses.csv"
        wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)

        closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

        if len(detection_results) == 0 or len(wrist_and_palm_poses) == 0 or len(closed_loop_traj) == 0:
            print("One of the provided files (detections, hand poses, camera poses) is empty.")
            return
        
        left_hand_object = None
        poses_left = deque(maxlen=15)
        touches_left = deque(maxlen=9)

        right_hand_object = None
        poses_right = deque(maxlen=15)
        touches_right = deque(maxlen=9)

        
        for index in range(provider.get_num_data(stream_id)):
            name_curr = f"frame_{index:05}.jpg"
            image_info = detection_results[name_curr]
            hand_dets = image_info['hand_dets']
            obj_dets = image_info['obj_dets']

            query_timestamp = provider.get_image_data_by_index(stream_id, index)[1].capture_timestamp_ns

            device_pose = get_nearest_pose(closed_loop_traj, query_timestamp)
            wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, query_timestamp)

            # No pose found
            if device_pose is None or wrist_and_palm_pose is None:
                if vis:
                    yield index, None, None
                continue
            
            # Time difference between query timestamp and found pose timestamps is too large (> 0.1s)
            if abs(device_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) > 1e8 or \
                abs(wrist_and_palm_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) > 1e8:
                if vis:
                    yield index, None, None
                continue
            
            T_world_device = device_pose.transform_world_device.to_matrix()

            if wrist_and_palm_pose.left_hand.confidence > 0.5:
                left_palm_position_device = wrist_and_palm_pose.left_hand.palm_position_device
                left_palm_position_world = np.dot(T_world_device, np.append(left_palm_position_device, 1))[:3]


                T_world_device[:3, 3] = left_palm_position_world
                if vis:
                    if left_hand_object is not None:
                        yield index, left_hand_object[0], T_world_device
                    else:
                        yield index, None, None
                
                if (obj_dets is not None) and (hand_dets is not None):
                    poses_left.append(T_world_device)
                    touches_left.append(True)
                    
                    if not any(hand_dets[i, 4] > 0.8 and hand_dets[i, 5] == 3 for i in range(hand_dets.shape[0])):
                        continue

                    if not any(obj_dets[i, 4] > 0.9 for i in range(obj_dets.shape[0])):
                        continue

                    # find nearest three objects in the scene graph based on distance to centroid
                    _, neighbor_indices = self.tree.query(left_palm_position_world, k=3)
                    # find distance to nearest neighbors by querying their convex hull
                    nearest_neighbor = np.array([self.nodes[neighbor_idx].hull_tree.query(left_palm_position_world, k=1)[0] for neighbor_idx in neighbor_indices])                                
                    
                    # for touching it has to be closer than 0.25m
                    if left_hand_object is None:
                        if np.min(nearest_neighbor) < 0.25: 
                            left_hand_object = neighbor_indices[np.argmin(nearest_neighbor)], np.min(nearest_neighbor), T_world_device, query_timestamp, left_palm_position_world
                    elif neighbor_indices[np.argmin(nearest_neighbor)] == left_hand_object[0] and np.min(nearest_neighbor) < left_hand_object[1] and (query_timestamp - left_hand_object[3] < 5*1e8):
                        left_hand_object = neighbor_indices[np.argmin(nearest_neighbor)], np.min(nearest_neighbor), T_world_device, query_timestamp, left_palm_position_world
                elif left_hand_object is not None:
                    poses_left.append(T_world_device)
                    touches_left.append(False)
                
                if len(poses_left) == 15 and not any(touches_left):
                    if right_hand_object is None or  right_hand_object[0] != left_hand_object[0]:
                        print("Left hand object let go")
                        print(self.label_mapping.get(self.nodes[left_hand_object[0]].sem_label, "ID not found"))
                        last_position = poses_left[0]
                        inv = np.linalg.inv(left_hand_object[2])
                        self.transform(left_hand_object[0], inv)
                        self.transform(left_hand_object[0], last_position)
                    poses_left.clear()
                    touches_left.clear()
                    left_hand_object = None
            else:
                if vis:
                    yield index, None, None
            
            if wrist_and_palm_pose.right_hand.confidence > 0.5:
                right_palm_position_device = wrist_and_palm_pose.right_hand.palm_position_device
                right_palm_position_world = np.dot(T_world_device, np.append(right_palm_position_device, 1))[:3]


                T_world_device[:3, 3] = right_palm_position_world
                
                if (obj_dets is not None) and (hand_dets is not None):
                    poses_right.append(T_world_device)
                    touches_right.append(True)

                    if not any(hand_dets[i, 4] > 0.8 and hand_dets[i, 5] == 3 for i in range(hand_dets.shape[0])):
                        continue

                    if not any(obj_dets[i, 4] > 0.9 for i in range(obj_dets.shape[0])):
                        continue

                    # find nearest three objects in the scene graph based on distance to centroid
                    _, neighbor_indices = self.tree.query(right_palm_position_world, k=3)
                    # find distance to nearest neighbors by querying their convex hull
                    nearest_neighbor = np.array([self.nodes[neighbor_idx].hull_tree.query(right_palm_position_world, k=1)[0] for neighbor_idx in neighbor_indices])                
                    
                    # for touching it has to be closer than 0.25m
                    if right_hand_object is None:
                        if np.min(nearest_neighbor) < 0.25: 
                            right_hand_object = neighbor_indices[np.argmin(nearest_neighbor)], np.min(nearest_neighbor), T_world_device, query_timestamp, right_palm_position_world
                    elif neighbor_indices[np.argmin(nearest_neighbor)] == right_hand_object[0] and np.min(nearest_neighbor) < right_hand_object[1] and (query_timestamp - right_hand_object[3] < 5*1e8):
                        right_hand_object = neighbor_indices[np.argmin(nearest_neighbor)], np.min(nearest_neighbor), T_world_device, query_timestamp, right_palm_position_world
                elif right_hand_object is not None:
                    poses_right.append(T_world_device)
                    touches_right.append(False)
                
                if len(poses_right) == 15 and not any(touches_right):
                    # TODO: what happens if left_hand_object is None?
                    if left_hand_object is None or right_hand_object[0] != left_hand_object[0]:
                        print("Right hand object let go")
                        print(self.label_mapping.get(self.nodes[right_hand_object[0]].sem_label, "ID not found"))
                        last_position = poses_right[0]
                        inv = np.linalg.inv(right_hand_object[2])
                        self.transform(right_hand_object[0], inv)
                        self.transform(right_hand_object[0], last_position)
                    poses_right.clear()
                    touches_right.clear()
                    right_hand_object = None
        
        if left_hand_object is not None and len(poses_left):
            if not vis and (right_hand_object is None or right_hand_object[0] != left_hand_object[0]):
                print("Following object was updated to the last hand pose of the scene as no release was detected:")
                print(self.label_mapping.get(self.nodes[left_hand_object[0]].sem_label, "ID not found"))
                last_position = poses_left[0]
                inv = np.linalg.inv(right_hand_object[2])
                self.transform(right_hand_object[0], inv)
                self.transform(right_hand_object[0], last_position)
            poses_right.clear()
            touches_right.clear()
            right_hand_object = None
        
        if right_hand_object is not None and len(poses_right):
            if left_hand_object is None or right_hand_object[0] != left_hand_object[0]:
                print("Following object was updated to the last hand pose of the scene as no release was detected:")
                print(self.label_mapping.get(self.nodes[right_hand_object[0]].sem_label, "ID not found"))
                last_position = poses_right[0]
                inv = np.linalg.inv(right_hand_object[2])
                self.transform(right_hand_object[0], inv)
                self.transform(right_hand_object[0], last_position)
            poses_right.clear()
            touches_right.clear()
            right_hand_object = None

    

    def track_object_interaction(self, interactions, all_contacts):
        first_interaction = interactions[0]
        last_interaction = interactions[-1]
        _, idx = self.tree.query(first_interaction)
        node = self.nodes[idx]
        distances = np.array([np.linalg.norm(point - node.centroid) for point in all_contacts])
        index = np.argmin(distances)
        self.transform(idx, last_interaction-all_contacts[index])
    
    
    def track_object_interaction_2(self, interactions, transforms):
        first_interaction = interactions[0]
        last_interaction = interactions[-1]
        _, idx = self.tree.query(first_interaction)
        first_transform = transforms[0]
        first_transform = np.linalg.inv(first_transform)
        last_transform = transforms[-1]
        self.transform(idx, first_transform)
        self.transform(idx, last_transform)
    
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

    
    def transform(self, idx, *args):
        """ Transform the points of a node (identified by idx) using a translation, rotation, or homogeneous transformation matrix."""
        for arg in args:
            if isinstance(arg, np.ndarray):
                if arg.shape == (3,):
                    # Apply the translation
                    self.nodes[idx].centroid += arg
                    self.nodes[idx].points += arg
                    self.nodes[idx].hull_tree = KDTree(self.nodes[idx].points[ConvexHull(self.nodes[idx].points).vertices])
                elif arg.shape == (3, 3):
                    # Apply the rotation
                    self.nodes[idx].points = np.dot(arg, self.nodes[idx].points.T).T
                    self.nodes[idx].centroid = np.dot(arg, self.nodes[idx].centroid)
                    self.nodes[idx].hull_tree = KDTree(self.nodes[idx].points[ConvexHull(self.nodes[idx].points).vertices])
                elif arg.shape == (4, 4):
                    # Apply the homogeneous transform matrix
                    self.nodes[idx].points = np.dot(arg, np.vstack((self.nodes[idx].points.T, np.ones(self.nodes[idx].points.shape[0])))).T[:, :3]
                    self.nodes[idx].centroid = np.dot(arg, np.append(self.nodes[idx].centroid, 1))[:3]
                    self.nodes[idx].hull_tree = KDTree(self.nodes[idx].points[ConvexHull(self.nodes[idx].points).vertices])
                else:
                    raise ValueError("Invalid argument shape. Expected (3,) for rotation, (3,3) for rotation, or (4,4) for homogeneous transformation.")
            else:
                raise TypeError("Invalid argument type. Expected numpy.ndarray.")

        self.tree = KDTree(np.array([node.centroid for node in self.nodes]))

    def instance_segmentation(self):
        """Still experimental. Enhance."""
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

    def visualize_tracking(self, scan_dir, output_path="output.mp4", fps=30, scale=0.0, labels=False):
        geometries = []
        for node in self.nodes:
            pcd = o3d.geometry.PointCloud()
            pcd_points = node.points + scale * node.centroid
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd_color = np.array(node.color, dtype=np.float64)
            pcd.paint_uniform_color(pcd_color)
            geometries.append((pcd, "node_" + str(node.object_id)))
        
        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Press <S> to save the pose", 1440, 1440)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.camera.look_at(np.array([0, 0, 0]), [0, 5, 0], [0, 1, 0])
        window.add_child(scene)

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        
        for geometry, name in geometries:
            scene.scene.add_geometry(name, geometry, material)

        if labels:
            for node in self.nodes:
                label = self.label_mapping.get(node.sem_label, "ID not found")
                point = node.centroid + scale * node.centroid
                offset = np.array([0, 0, 0.1])
                scene.add_3d_label(point + offset, label)

        camera_params = [None, None, None]
        
        # Set a key event callback to capture the screen
        def on_key_event(event):
            if event.key == gui.KeyName.S:  # Capture screen when 'S' key is pressed
                model = scene.scene.camera.get_model_matrix()
                fov = scene.scene.camera.get_field_of_view()
                up_direction = np.dot(model, np.array([0, 1, 0, 1]))[:3]
                eye = np.dot(model, np.array([0, 0, 0, 1]))[:3]
                camera_params[0] = up_direction
                camera_params[1] = eye
                camera_params[2] = fov
                gui.Application.instance.quit()
                return True
            return False

        window.set_on_key(on_key_event)
        
        # Run the application
        gui.Application.instance.run()
        if camera_params[0] is None:
            print("camera parameters not set correctly")
            return
        
        up_direction = camera_params[0]
        eye = camera_params[1]
        fov = camera_params[2]
        center = np.array([0, 0, 0])
        
        #### image creation:
        render = o3d.visualization.rendering.OffscreenRenderer(1440, 1440)

        for geometry, name in geometries:
            render.scene.add_geometry(name, geometry, material)

        render.setup_camera(fov, center, eye, up_direction)
        
        last_img = render.render_to_image()

        render.scene.clear_geometry()

        images = []
        initial_pose = None
        for index, identifier, pose in self.track(scan_dir=scan_dir, vis=True):
            print(index)
            if identifier is None:
                images += [np.asarray(last_img)]
                continue
            
            geometries = []
            if initial_pose is None:
                for node in self.nodes:
                    pcd = o3d.geometry.PointCloud()
                    pcd_points = node.points + scale * node.centroid
                    pcd.points = o3d.utility.Vector3dVector(pcd_points)
                    pcd_color = np.array(node.color, dtype=np.float64)
                    pcd.paint_uniform_color(pcd_color)
                    geometries.append((pcd, "node_" + str(node.object_id)))            
                inv = np.linalg.inv(pose)
                initial_pose = inv
                self.transform(identifier, inv)
            else:
                inv = np.linalg.inv(pose)
                self.transform(identifier, pose)
                for node in self.nodes:
                    pcd = o3d.geometry.PointCloud()
                    pcd_points = node.points + scale * node.centroid
                    pcd.points = o3d.utility.Vector3dVector(pcd_points)
                    pcd_color = np.array(node.color, dtype=np.float64)
                    pcd.paint_uniform_color(pcd_color)
                    geometries.append((pcd, "node_" + str(node.object_id))) 
                self.transform(identifier, inv)

            

            for geometry, name in geometries:
                render.scene.add_geometry(name, geometry, material)

            render.setup_camera(fov, center, eye, up_direction)
            
            img_o3d = render.render_to_image()
            last_img = img_o3d
            images += [np.asarray(img_o3d)]

            render.scene.clear_geometry()
        
        with imageio.get_writer(output_path, fps=fps) as writer:
            for image_file in images:
                # image = imageio.imread(image_file)
                writer.append_data(image_file)


    
    def visualize(self, centroids=True, connections=True, scale=0.0, labels=False, optional_geometry=None):
        geometries = []
        for node in self.nodes:
            pcd = o3d.geometry.PointCloud()
            pcd_points = node.points + scale * node.centroid
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd_color = np.array(node.color, dtype=np.float64)
            pcd.paint_uniform_color(pcd_color)
            geometries.append((pcd, "node_" + str(node.object_id)))

        if centroids:
            centroid_pcd = o3d.geometry.PointCloud()
            centroids_xyz = np.array([node.centroid + scale * node.centroid for node in self.nodes])
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

        # Initialize the GUI application
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
                point = node.centroid + scale * node.centroid
                offset = np.array([0, 0, 0.1])
                scene.add_3d_label(point + offset, label)

        # Function to capture the screen
        def capture_screen():
            # Wait for a moment to ensure the scene is rendered
            # gui.Application.instance.post_to_main_thread(window, lambda: None)
            # Render the scene to an image
            image = gui.Application.instance.render_to_image(scene.scene, 1024, 768)
            o3d.io.write_image("screenshot.png", image)

        # Set a key event callback to capture the screen
        def on_key_event(event):
            if event.key == gui.KeyName.S:  # Capture screen when 'S' key is pressed
                capture_screen()
                return True
            return False

        window.set_on_key(on_key_event)
        
        # Run the application
        gui.Application.instance.run()

    # def visualize(self, centroids=True, connections=True, scale=0.0, labels = False, optional_geometry=None):
    #     geometries = []
    #     for node in self.nodes:
    #         pcd = o3d.geometry.PointCloud()
    #         pcd_points = node.points + scale*node.centroid
    #         pcd.points = o3d.utility.Vector3dVector(pcd_points)
    #         pcd_color = np.array(node.color, dtype=np.float64)
    #         pcd.paint_uniform_color(pcd_color)
    #         geometries.append((pcd, "node_" + str(node.object_id)))

    #     if centroids:
    #         centroid_pcd = o3d.geometry.PointCloud()
    #         centroids_xyz = np.array([node.centroid + scale*node.centroid for node in self.nodes])
    #         centroids_colors = np.array([node.color for node in self.nodes], dtype=np.float64) / 255.0
    #         centroid_pcd.points = o3d.utility.Vector3dVector(centroids_xyz)
    #         centroid_pcd.colors = o3d.utility.Vector3dVector(centroids_colors)
    #         geometries.append((centroid_pcd, "centroids"))

    #     if connections:
    #         line_points = []
    #         line_indices = []
    #         idx = 0
    #         for node in self.nodes:
    #             _, indices = self.tree.query(node.centroid, k=self.k)
    #             for idx_neighbor in indices[1:]:
    #                 neighbor = self.nodes[idx_neighbor]
    #                 line_points.append(node.centroid + scale * node.centroid)
    #                 line_points.append(neighbor.centroid + scale * neighbor.centroid)
    #                 line_indices.append([idx, idx + 1])
    #                 idx += 2
    #         if line_points:
    #             line_set = o3d.geometry.LineSet(
    #                 points=o3d.utility.Vector3dVector(line_points),
    #                 lines=o3d.utility.Vector2iVector(line_indices)
    #             )
    #             line_set.paint_uniform_color([0, 0, 0])
    #             geometries.append((line_set, "connections"))
        
    #     if optional_geometry and scale == 0.0:
    #         if isinstance(optional_geometry, list):
    #             for i, geom in enumerate(optional_geometry):
    #                 geometries.append((geom, f"optional_geometry_{i}"))
    #         else:
    #             geometries.append((optional_geometry, "optional_geometry"))
        
    #     # Create a new Visualizer
    #     vis = o3d.visualization.Visualizer()

    #     gui.Application.instance.initialize()
    #     window = gui.Application.instance.create_window("3D Visualization", 1024, 768)

    #     scene = gui.SceneWidget()
    #     scene.scene = rendering.Open3DScene(window.renderer)
    #     # scene.scene = rendering.Scene(window.renderer)
    #     window.add_child(scene)

    #     material = rendering.MaterialRecord()
    #     material.shader = "defaultLit"

    #     for geometry, name in geometries:
    #         scene.scene.add_geometry(name, geometry, material)

    #     if geometries:
    #         bounds = geometries[0][0].get_axis_aligned_bounding_box()
    #         for geometry, _ in geometries[1:]:
    #             bounds += geometry.get_axis_aligned_bounding_box()
    #         scene.setup_camera(60, bounds, bounds.get_center())

    #     if labels:
    #         for node in self.nodes:
    #             label = self.label_mapping.get(node.sem_label, "ID not found")
    #             point = node.centroid + scale*node.centroid
    #             offset = np.array([0, 0, 0.1])
    #             scene.add_3d_label(point+offset, label)

    #     gui.Application.instance.run()

    #     vis.destroy_window()

    # def build_mask3d(self, label_path, pcd_path):
    #     with open(label_path, 'r') as file:
    #         lines = file.readlines()
        
    #     file_paths = []
    #     values = []
    #     confidences = []

    #     for line in lines:
    #         parts = line.split()
    #         file_paths.append(parts[0])
    #         values.append(int(parts[1]))
    #         confidences.append(float(parts[2]))
        
    #     base_dir = os.path.dirname(os.path.abspath(label_path))
    #     first_pred_mask_path = os.path.join(base_dir, file_paths[0])

    #     with open(first_pred_mask_path, 'r') as file:
    #         num_lines = len(file.readlines())

    #     mask3d_labels = np.zeros(num_lines, dtype=np.int64)

    #     pcd = o3d.io.read_point_cloud(pcd_path)

    #     np_points = np.array(pcd.points)
    #     np_colors = np.array(pcd.colors)

    #     for i, relative_path in enumerate(file_paths):
    #         file_path = os.path.join(base_dir, relative_path)
    #         labels = np.loadtxt(file_path, dtype=np.int64)
    #         node_points = np_points[labels==1] # np_points[np.logical_and(labels == 1 , mask3d_labels == 0)]
    #         colors = np_colors[labels==1] #  np_colors[np.logical_and(labels == 1 , mask3d_labels == 0)]
    #         # TODO: is this correct?
    #         mask3d_labels[np.logical_and(labels == 1 , mask3d_labels == 0)] = values[i]
    #         if confidences[i] > self.min_confidence and node_points.shape[0] > 0:
    #             mean_point = np.mean(node_points, axis=0)
    #             self.add_node(mean_point, colors[0], values[i], node_points, confidences[i])
        
    #     sorted_nodes = sorted(self.nodes, key=lambda node: node.object_id)
        
    #     self.tree = KDTree(np.array([node.centroid for node in sorted_nodes]))
    
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



