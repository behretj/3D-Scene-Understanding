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
import imageio, datetime, time
from tqdm import tqdm

class ObjectNode:
    def __init__(self, object_id, centroid, color, sem_label, points, confidence=None, movable=True):
        self.object_id = object_id
        self.centroid = centroid
        self.points = points
        self.sem_label = sem_label
        self.color = color
        self.movable = movable
        self.confidence = confidence
        self.update_hull_tree()
    
    def update_hull_tree(self):
        self.hull_tree = KDTree(self.points[ConvexHull(self.points).vertices])

class SceneGraph:
    def __init__(self, label_mapping = dict(), min_confidence = 0.0,  k=2, unmovable=[]):
        self.index = 0
        self.nodes = []
        self.tree = None
        self.k = k
        self.label_mapping = label_mapping
        self.min_confidence = min_confidence
        self.unmovable = unmovable

    def add_node(self, centroid, color, sem_label, points, confidence=None):
        if self.label_mapping.get(sem_label, "ID not found") in self.unmovable:
            # mark objects as unmovable if a list was given
            # TODO: could in the future either be a complete list or LLM api request for open vocabulary
            self.nodes.append(ObjectNode(self.index, centroid, np.array([0.5, 0.5, 0.5]), sem_label, points, confidence, movable=False))
        else:
            self.nodes.append(ObjectNode(self.index, centroid, color, sem_label, points, confidence))
        self.index += 1

    def get_node_info(self):
        for node in self.nodes:
            print(f"Object ID: {node.object_id}")
            print(f"Centroid: {node.centroid}")
            print("Semantic Label: " + self.label_mapping.get(node.sem_label, "ID not found"))
            print(f"Confidence: {node.confidence}")
    
    def save(self, file_path):
        """ Saves the scene graph to a pickle file. """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
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
    
    def add_drawer(self, drawer_points, handle_points):
        all_points = np.vstack((drawer_points, handle_points))
        mean_point = np.mean(all_points, axis=0)
        # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(all_points))
        # bbox_min = bbox.get_minimal_oriented_bounding_box()
        # for node in self.nodes:
        #     ind_list = bbox_min.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(node.points))
        #     mask = np.isin(np.arange(len(node.points)), ind_list, invert=True)
        #     node.points = node.points[mask]

        # 25 corresponds to label of door
        self.add_node(mean_point, np.random.rand(3), 25, all_points)
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
                    # TODO: could be replaced by self.nodes[idx].update_hull_tree()?
                else:
                    raise ValueError("Invalid argument shape. Expected (3,) for rotation, (3,3) for rotation, or (4,4) for homogeneous transformation.")
            else:
                raise TypeError("Invalid argument type. Expected numpy.ndarray.")

        self.tree = KDTree(np.array([node.centroid for node in self.nodes]))

    def instance_segmentation(self):
        """Still experimental. TODO: Enhance."""
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

    def visualize_drawers(self, points, masks):
        pcds = []
        for i in range(masks.shape[0]):
            mask = masks[i, :].astype(bool)
            pcd_points = points[mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.paint_uniform_color(np.random.rand(3))
            pcds += [pcd]
        self.visualize(optional_geometry=pcds)
    
    def color_with_ibm_palette(self):
        """ manual definition of the IBM palette including 10 colors """
        colors = np.array([[0.39215686, 0.56078431, 1.], [0.47058824, 0.36862745, 0.94117647], [0.8627451 , 0.14901961, 0.49803922],
                [0.99607843, 0.38039216, 0], [1., 0.69019608, 0.], [0.29803922, 0.68627451, 0.31372549], [0., 0.6, 0.8],
                [0.70196078, 0.53333333, 1.], [0.89803922, 0.22352941, 0.20784314], [1., 0.25098039, 0.50588235]])

        index = 0
        for node in self.nodes:
            if node.movable:
                node.color = colors[index]
                index += 1
            if index >= len(colors):
                index = 0
    
    def label_correction(self):
        """ manual correction of the semantic labels for my end presentation. """

        object_id_to_sem_label = {
            18: 572,
            16: 1163,
            10: 1189,
            3: 1276,
            15: 8,
            12: 1145,
            6: 1276,
            13: 15,
            14: 168,
            17: 87,
            8: 8,
            0: 749
        }

        for node in self.nodes:
            if node.object_id in object_id_to_sem_label:
                node.sem_label = object_id_to_sem_label[node.object_id]
            

    def track_hand(self, scan_dir, left=True):
        """Track the hand in the given scene. Returns, if posible a list of tuples containing the object id,
        the pose of the camera, the position of the hand, and the offset between the hand and the object."""


        ### Load the necessary files
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
        
        tracking = []
        
        hand_object, offset = None, None
        object_detected = deque(maxlen=7)
        object_positions = deque(maxlen=5)
        average_speed = dict()
        min_distance = 0.25
        
        # iterate through the frames
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
                tracking += [None]
                continue
            
            # Time difference between query timestamp and found pose timestamps is too large (> 0.1s)
            if abs(device_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) > 1e8 or \
                abs(wrist_and_palm_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) > 1e8:
                tracking += [None]
                continue
            
            # Pose of aria glasses in world coordinates
            T_world_device = device_pose.transform_world_device.to_matrix()

            # Get left or right hand pose, depending on passed argument
            if left:
                palm_position_device = wrist_and_palm_pose.left_hand.palm_position_device
                if wrist_and_palm_pose.left_hand.confidence < 0.0:
                    tracking += [(None, T_world_device, None, None)]
                    object_detected.append(False)
                    continue
            else:
                palm_position_device = wrist_and_palm_pose.right_hand.palm_position_device
                if wrist_and_palm_pose.right_hand.confidence < 0.0:
                    tracking += [(None, T_world_device, None, None)]
                    object_detected.append(False)
                    continue
            
            # Calculate the palm position in world coordinates
            palm_position_world = np.dot(T_world_device, np.append(palm_position_device, 1))[:3]

            object_positions.append(palm_position_world)
            if len(object_positions) > 2:
                displacements = [np.linalg.norm(object_positions[i+1] - object_positions[i]) for i in range(len(object_positions)-1)]
                average_displacement = sum(displacements) / len(displacements)
                average_speed[index] = average_displacement

            # Hand-object-tracker found both, a hands and an object
            if (obj_dets is not None) and (hand_dets is not None):
                #  Confidence of the hand detection is too low
                if not any(hand_dets[i, 4] > 0.7 and hand_dets[i, 5] == 3 for i in range(hand_dets.shape[0])) or \
                    not any(obj_dets[i, 4] > 0.7 for i in range(obj_dets.shape[0])):
                    object_detected.append(False)
                else:
                    # Get the nearest 4 objects to the palm position
                    _, neighbor_indices = self.tree.query(palm_position_world, k=4)
                    neighbor_indices = [index for index in neighbor_indices if self.nodes[index].movable]
                    
                    # No object is close by
                    if len(neighbor_indices)==0:
                        # TODO: shouldn't this be True?
                        object_detected.append(False)
                    else:
                        # Query the convex hull of the nearest neighbors to determine the actual nearest object
                        nearest_neighbor = np.array([self.nodes[neighbor_idx].hull_tree.query(palm_position_world, k=1)[0] for neighbor_idx in neighbor_indices])                                
                        # If the nearest neighbor is closer than 25cm, we assume that it was an actual object detection
                        if np.min(nearest_neighbor) < 0.25 and hand_object is None:
                            object_detected.append(True)
                            
                            # hand velocity has to be low enough to assume that the object was taken
                            # TODO: overwrite previous object detections, if we are even closer than the previous iteration
                            if average_speed.get(index, 1) < 0.015:
                                hand_object = neighbor_indices[np.argmin(nearest_neighbor)]
                                offset = self.nodes[hand_object].centroid - palm_position_world
                                min_distance = np.min(nearest_neighbor)
                                print("object %d was taken at frame %d" %(hand_object, index))
                        elif np.min(nearest_neighbor) < min_distance:
                            object_detected.append(True)
                            tmp_object = neighbor_indices[np.argmin(nearest_neighbor)]
                            if hand_object == tmp_object and index > 0:
                                offset = self.nodes[hand_object].centroid - palm_position_world
                                # remove the previous object detection, if the hand is now closer to the object
                                tmp_tracking = tracking[index-1]
                                min_distance = np.min(nearest_neighbor)
                                tracking[index-1] = (None, tmp_tracking[1], tmp_tracking[2], None)
                        elif hand_object is not None:
                            object_detected.append(True)
            else:
                # we don't have an object detection, except if we have something in hand (hand_object is not None) and the average hand speed is high
                # print(average_speed.get(index, 0), hand_object)
                # if not (average_speed.get(index, 0) > 0.02 and hand_object is not None):
                object_detected.append(False)

            if not any(object_detected) and hand_object is not None:
                ### clean up for noise in the hand-object-tracker, TODO: make this "more" online
                # print("Before cleaning", index)
                momentum_index = index
                current = average_speed.get(momentum_index, 1)
                while current > 0.015 and momentum_index > 0:
                    if tracking[momentum_index-1] is not None:
                        _, tmp_pose, tmp_position, tmp_offset = tracking[momentum_index-1]
                        tracking[momentum_index-1] = (None, tmp_pose, tmp_position, tmp_offset)
                    momentum_index -= 1
                    current = average_speed.get(momentum_index, 1)
                hand_object = None
                min_distance = 0.25
                print("object was let go at frame", momentum_index)
            
            if hand_object is None:
                tracking += [(None, T_world_device, palm_position_world, None)]
            else:
                tracking += [(hand_object, T_world_device, palm_position_world, offset)]
        
        return tracking
        
    def merge_tracking(self, scan_dir):
        """ Merges the results of the tracking of left and right hand into a common list. """
        left = self.track_hand(scan_dir, left=True)
        right = self.track_hand(scan_dir, left=False)

        assert len(left) == len(right)

        tracking = []
        
        for left_info, right_info in zip(left, right):
            pose, left_id, left_pos, left_offset, right_id, right_pos, right_offset = None, None, None, None, None, None, None
            if left_info is not None:
                pose = left_info[1]
                left_id = left_info[0]
                left_pos = left_info[2]
                left_offset = left_info[3]
            if right_info is not None:
                pose = right_info[1]
                right_id = right_info[0]
                right_pos = right_info[2]
                right_offset = right_info[3]
            tracking += [(pose, left_id, left_pos, left_offset, right_id, right_pos, right_offset)]
        
        return tracking
    
    def track_changes(self, scan_dir):
        tracking = self.merge_tracking(scan_dir)

        initial_left, initial_right = None, None
        for (pose, left_id, left_pos, left_offset, right_id, right_pos, right_offset) in tracking:
            if pose is None:
                initial_left, initial_right = None, None
                continue            
            
            # there is a detection for the left hand
            if left_id is not None:
                left_correction = np.dot(pose[:3,:3], left_offset)
                pose[:3, 3] = left_pos + left_correction
                # object was not moved in previous iteration
                if initial_left is None:           
                    inv = np.linalg.inv(pose)
                    initial_left = inv
                else:
                    self.transform(left_id, np.dot(pose, initial_left))
                    initial_left = np.linalg.inv(pose)
            else:
                initial_left = None
            
            # there is a detection for the right hand and the same object hasn't been moved with the left hand
            if right_id is not None and right_id != left_id:
                right_correction = np.dot(pose[:3,:3], right_offset)
                pose[:3, 3] = right_pos + right_correction
                # object was not moved in previous iteration
                if initial_right is None:
                    inv = np.linalg.inv(pose)
                    initial_right = inv
                else:
                    self.transform(right_id, np.dot(pose, initial_right))
                    initial_right = np.linalg.inv(pose)
            else:
                initial_right = None


    def tracking_video(self, scan_dir, output_path="output.mp4", fps=30, scale=0.0, labels=False, fpv=False):
        """ create a video of the tracking process from the given scan directory. It does NOT
        modify the scene graph, but only visualizes the tracking process. To accomodate the tracking,
        use the 'track_changes' method. The fpv option offers a first-person view of the scene without actually
        tracking the changes. """

        def get_look_at_parameters(extrinsics):
            """ transfer the extrinsics matrix to compatible parameters for the Open3D rendering camera"""
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]
            
            eye = t
            
            camera_direction = np.array([0, 0, 1])
            center = R @ camera_direction + t
            
            camera_up = np.array([-1, 0, 0])
            up = R @ camera_up
            
            return center, eye, up

        # get the tracking results
        tracking = self.merge_tracking(scan_dir)
        
        geometries = []
        for node in self.nodes:
            pcd = o3d.geometry.PointCloud()
            pcd_points = node.points + scale * node.centroid
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd_color = np.array(node.color, dtype=np.float64)
            pcd.paint_uniform_color(pcd_color)
            geometries.append((pcd, "node_" + str(node.object_id)))
        

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"

        if not fpv:
            # Logic to capture the camera parameters for the resultsing video
            gui.Application.instance.initialize()
            window = gui.Application.instance.create_window("Press <S> to save the pose", 1408, 1408)
            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(window.renderer)
            scene.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
            intrinsics = np.array([
                [611.428, 0, 703.5],
                [0, 611.428, 703.5],
                [0, 0, 1]
            ])
            scene.scene.camera.set_projection(intrinsics, 1, 5, 1408, 1408)
            scene.scene.camera.look_at(np.array([0, 0, 0]), [0, 3, 0], [0, 1, 0])
            window.add_child(scene)
            
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
                if event.type == gui.KeyEvent.Type.DOWN:
                    if event.key == gui.KeyName.S:  # Capture screen when 'S' key is pressed
                        model = scene.scene.camera.get_model_matrix()
                        center = scene.scene.camera.unproject(704, 704, 0.5, 1408, 1408)
                        up_direction = np.dot(model, np.array([0, 1, 0, 1]))[:3]
                        eye = np.dot(model, np.array([0, 0, 0, 1]))[:3]
                        camera_params[0] = up_direction
                        camera_params[1] = eye
                        camera_params[2] = center
                        gui.Application.instance.quit()
                        return True
                return False

            window.set_on_key(on_key_event)
            
            # Run the application
            gui.Application.instance.run()
            
            if camera_params[0] is None:
                print("camera parameters not set correctly")
                return
            
            up = camera_params[0]
            eye = camera_params[1]
            center = camera_params[2]
        else:
            T_world_device = np.eye(4)
            ## get first pose from video
            for pose, left_id, left_pos, left_offset, right_id, right_pos, right_offset in tracking:
                if pose is not None:
                    T_world_device = pose
                    break
            ### Aria specific camera parameters, TODO: make these two variables generic?
            T_device_camera = np.array([
                [0.99205683, -0.05140061, 0.11480955, -0.00404916],
                [0.11182453, 0.77834746, -0.61779488, -0.01218969],
                [-0.05760668, 0.62572615, 0.77791276, -0.0051337],
                [0, 0, 0, 1]
            ])
            intrinsics = np.array([
                [611.428, 0, 703.5],
                [0, 611.428, 703.5],
                [0, 0, 1]
            ])
            extrinsics = np.dot(T_world_device, T_device_camera)
            center, eye, up = get_look_at_parameters(extrinsics)
        
        #### image creation:
        render = o3d.visualization.rendering.OffscreenRenderer(1408, 1408)
        render.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))

        # create initial scene with the geometries
        for geometry, name in geometries:
            render.scene.add_geometry(name, geometry, material)

        render.scene.camera.look_at(center, eye, up)
        
        last_img = render.render_to_image()

        images = []

        initial_left, initial_right = None, None

        for idx, (pose, left_id, left_pos, left_offset, right_id, right_pos, right_offset) in enumerate(tracking):       
            # for non-fpv view, we generate the images with the hand positions
            if not fpv:
                render.scene.remove_geometry("left"+str(idx-10))
                render.scene.remove_geometry("right"+str(idx-10))
                
                if left_pos is not None or right_pos is not None:
                    if left_pos is not None:
                        lefty = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                        lefty.translate(left_pos)
                        if left_id is None: lefty.paint_uniform_color([0.89803922, 0.22352941, 0.20784314])
                        else: lefty.paint_uniform_color([0.29803922, 0.68627451, 0.31372549])
                        lefty.compute_vertex_normals()
                        render.scene.add_geometry("left"+str(idx), lefty, material)
                    if right_pos is not None:
                        righty = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                        righty.translate(right_pos)
                        if right_id is None: righty.paint_uniform_color([0.47058824, 0.36862745, 0.94117647])
                        else: righty.paint_uniform_color([0.99607843, 0.38039216, 0])
                        righty.compute_vertex_normals()
                        render.scene.add_geometry("right"+str(idx), righty, material)
                    if pose is None:
                        last_img = render.render_to_image()

            if pose is None:
                images += [np.asarray(last_img)]
                initial_left, initial_right = None, None
                continue

            if not fpv:
                # change the scene according to tracking results
                # there is a detection for the left hand
                if left_id is not None:
                    left_correction = np.dot(pose[:3,:3], left_offset)
                    pose[:3, 3] = left_pos + left_correction
                    if initial_left is None:           
                        inv = np.linalg.inv(pose)
                        initial_left = inv
                    else:
                        current_pose = render.scene.get_geometry_transform("node_" + str(self.nodes[left_id].object_id))
                        current_pose = np.dot(pose, np.dot(initial_left, current_pose)) # transform
                        render.scene.set_geometry_transform("node_" + str(self.nodes[left_id].object_id), current_pose)
                        initial_left = np.linalg.inv(pose)
                else:
                    initial_left = None
                
                # there is a detection for the right hand and the same object hasn't been moved with the left hand
                if right_id is not None and right_id != left_id:
                    right_correction = np.dot(pose[:3,:3], right_offset)
                    pose[:3, 3] = right_pos + right_correction
                    if initial_right is None:
                        inv = np.linalg.inv(pose)
                        initial_right = inv
                    else:
                        current_pose = render.scene.get_geometry_transform("node_" + str(self.nodes[right_id].object_id))
                        current_pose = np.dot(pose, np.dot(initial_right, current_pose))
                        render.scene.set_geometry_transform("node_" + str(self.nodes[right_id].object_id), current_pose)
                        initial_right = np.linalg.inv(pose)
                else:
                    initial_right = None
            
            # in first-person view, we need to update to our current view each iteration
            if fpv:
                extrinsics = np.dot(pose, T_device_camera)
                center, eye, up = get_look_at_parameters(extrinsics)
                render.scene.camera.look_at(center, eye, up)
            
            
            last_img = render.render_to_image()
            images += [np.asarray(last_img)]
        
        # create video from images
        with imageio.get_writer(output_path, fps=fps) as writer:
            for image_file in tqdm(images):
                writer.append_data(image_file)

        render.scene.clear_geometry()
    
    def visualize(self, centroids=True, connections=True, scale=0.0, labels=False, optional_geometry=None):
        """ Visualizes the scene graph in its current state, different visualization options can be set. """
        
        # add the geometries to the scene
        geometries = []

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 2

        for node in self.nodes:
            pcd = o3d.geometry.PointCloud()
            pcd_points = node.points + scale * node.centroid
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd_color = np.array(node.color, dtype=np.float64)
            pcd.paint_uniform_color(pcd_color)
            geometries.append((pcd, "node_" + str(node.object_id), material))

        if centroids:
            centroid_pcd = o3d.geometry.PointCloud()
            centroids_xyz = np.array([node.centroid + scale * node.centroid for node in self.nodes])
            centroids_colors = np.array([node.color for node in self.nodes], dtype=np.float64) / 255.0
            centroid_pcd.points = o3d.utility.Vector3dVector(centroids_xyz)
            centroid_pcd.colors = o3d.utility.Vector3dVector(centroids_colors)
            geometries.append((centroid_pcd, "centroids", material))

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
                geometries.append((line_set, "connections", line_mat))

        if optional_geometry and scale == 0.0:
            if isinstance(optional_geometry, list):
                for i, geom in enumerate(optional_geometry):
                    geometries.append((geom, f"optional_geometry_{i}", material))
            else:
                geometries.append((optional_geometry, "optional_geometry", material))

        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Press <S> to capture a screenshot or <ESC> to quit the application.", 1024, 1024)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
        window.add_child(scene)

        for geometry, name, mat in geometries:
            scene.scene.add_geometry(name, geometry, mat)

        if geometries:
            bounds = geometries[0][0].get_axis_aligned_bounding_box()
            for geometry, _, _ in geometries[1:]:
                bounds += geometry.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

        if labels:
            for node in self.nodes:
                label = self.label_mapping.get(node.sem_label, "ID not found")
                point = node.centroid + scale * node.centroid
                offset = np.array([0, 0, 0.01])
                scene.add_3d_label(point + offset, label)
        
        # Set a key event callback to capture the screen
        def on_key_event(event):
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == gui.KeyName.S:  # Capture screen when 'S' key is pressed
                    image = gui.Application.instance.render_to_image(scene.scene, 1024, 1024)
                    current_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
                    filename = f"screenshot_{current_time}.png"
                    image = gui.Application.instance.render_to_image(scene.scene, 1024, 1024)
                    o3d.io.write_image(filename, image)
                    time.sleep(0.5)
                    return True
                if event.key == gui.KeyName.ESCAPE:  # Quit application when 'ESC' key is pressed
                    gui.Application.instance.quit()
                    return True
            return False

        window.set_on_key(on_key_event)
        
        # Run the application
        gui.Application.instance.run()



