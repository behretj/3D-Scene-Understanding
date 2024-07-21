import open3d as o3d
import numpy as np
import cv2, os, glob, pickle
from sklearn.cluster import MeanShift, KMeans, DBSCAN
from math import ceil
from projecting import detections_to_bboxes
from drawer_detection import predict_yolodrawer

def compute_iou(array1, array2):
    intersection = np.intersect1d(array1, array2)
    union = np.union1d(array1, array2)
    iou = len(intersection) / len(union)
    return iou

def dynamic_threshold(detection_counts, n_clusters=2):
    differences = np.array([abs(j - i) for i, j in zip(detection_counts[:-1], detection_counts[1:])]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(differences)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    
    if len(cluster_centers) > 1:
        threshold = (cluster_centers[0] + cluster_centers[1]) / 2
    else:
        threshold = cluster_centers[0]
    
    return threshold

def cluster_images(detections):
    if not detections:
        return []
    
    detection_counts = [n for (_, n) in detections]
    
    threshold = ceil(dynamic_threshold(detection_counts))
    clusters = []
    current_cluster = []

    for index, count in enumerate(detection_counts):
        if not current_cluster or (index > 0 and abs(detection_counts[index - 1] - count) <= threshold):
            current_cluster.append((index, count))
        else:
            if current_cluster[-1][1] > 0: 
                clusters.append(current_cluster)
            current_cluster = [(index, count)]
    
    if current_cluster:
        clusters.append(current_cluster)

    return clusters

def select_optimal_images(clusters):
    optimal_images = []
    for cluster in clusters:
        if cluster:
            optimal_images.append(max(cluster, key=lambda x: x[1])[0])
    return optimal_images

def register_drawers(dir_path):
    # stores tuples containing the detected box(es) and its/their confidence(s)
    detections = []
    if os.path.exists(os.path.join(dir_path, 'detections.pkl')):
        with open(os.path.join(dir_path, 'detections.pkl'), 'rb') as f:
            detections = pickle.load(f)
    else:
        for image_name in sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg'))):
            img_path = os.path.join(dir_path, image_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections += [predict_yolodrawer(image, image_name[:-4], vis_block=False)]
        with open(os.path.join(dir_path, 'detections.pkl'), 'wb') as f:
            pickle.dump(detections, f)
    
    clusters = cluster_images(detections)
    
    optimal_images = select_optimal_images(clusters)
    
    detections = [det for subdets in [detections[opt][0] for opt in optimal_images] for det in subdets]
    
    # TODO: choose a different standard name for the pcd file
    pcd_original = o3d.io.read_point_cloud(os.path.join(dir_path, 'mesh_labelled_mask3d_dataset_1_y_up.ply'))
    bboxes_3d = detections_to_bboxes(np.asarray(pcd_original.points), detections)

    all_bbox_indices = [(np.array(bbox.get_point_indices_within_bounding_box(pcd_original.points)), conf) for bbox, conf in bboxes_3d]

    registered_indices = []
    for indcs, conf in all_bbox_indices:     
        for idx, (reg_indcs, confidence) in enumerate(registered_indices):
            iou = compute_iou(reg_indcs, indcs)
            if iou > 0.1:  # Check if the overlap is greater than 10%
                if conf > confidence:
                    registered_indices[idx] = (indcs, conf)
                break
        else:
            registered_indices.append((indcs, conf))
    
    return [indcs for (indcs, _) in sorted(registered_indices, key=lambda x: x[1])]

def dbscan_clustering(detections):

    features = [{'image_id': id, 'num_drawers': n} for (id, n) in detections]

    # Convert detection counts to numpy array for clustering
    num_detections = np.array([dc['num_drawers'] for dc in features]).reshape(-1, 1)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=1, min_samples=5)  # eps and min_samples can be tuned based on your data
    labels = dbscan.fit_predict(num_detections)

    # Identify the core cluster with the most images
    unique_labels, counts = np.unique(labels, return_counts=True)
    core_cluster = unique_labels[np.argmax(counts[unique_labels != -1])]  # Exclude noise label (-1)

    # Filter images based on the core cluster
    selected_indices = np.where(labels == core_cluster)[0]
    refined_detections = [detections[i] for i in selected_indices]

    # print(f"Selected {len(refined_detections)} images from the core cluster with the most detections.")

def mean_shift_clustering(detections):
    # features are only the number of detections per image
    features = np.array([np.array([i, n]) for i, (_, n) in enumerate(detections)])

    mean_shift = MeanShift()
    mean_shift.fit(features)
    labels = mean_shift.labels_

    cluster_counts = np.bincount(labels)
    best_cluster = np.argmax(cluster_counts)
    selected_indices = np.where(labels == best_cluster)[0]


    refined_detections = [detections[i][0] for i in selected_indices]

    # print(f"Selected {len(refined_detections)} images from cluster with the most detections.")

    return refined_detections

if __name__ == "__main__":
    _ = register_drawers("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1")
