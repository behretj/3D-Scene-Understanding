import open3d as o3d
import numpy as np
import cv2, os, glob
from projecting import detections_to_bboxes
from drawer_detection import predict_yolodrawer

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for 3D bounding boxes."""
    # Convert Open3D bounding boxes to numpy arrays
    box1_points = np.asarray(box1.get_box_points())
    box2_points = np.asarray(box2.get_box_points())
    
    # Calculate intersection volume
    inter_min = np.maximum(np.min(box1_points, axis=0), np.min(box2_points, axis=0))
    inter_max = np.minimum(np.max(box1_points, axis=0), np.max(box2_points, axis=0))
    inter_vol = np.prod(np.maximum(0, inter_max - inter_min))
    
    # Calculate volume of each bounding box
    box1_vol = np.prod(np.max(box1_points, axis=0) - np.min(box1_points, axis=0))
    box2_vol = np.prod(np.max(box2_points, axis=0) - np.min(box2_points, axis=0))
    
    # Calculate IoU
    iou = inter_vol / (box1_vol + box2_vol - inter_vol)
    return iou

def register_drawers(dir_path):
    # stores tuples containing the detected box(es) and its/their confidence(s)
    detections = []
    # TODO: is it jpg or png?
    for image_name in sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg'))):
        img_path = os.path.join(dir_path, image_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Detection = namedtuple("Detection", ["name", "conf", "bbox"])
        # TODO: integrate from Oliver's repo, adapted to fit nicely here
        detections += predict_yolodrawer(image, image_name[:-4], vis_block=False)

    
    
    bboxes_3d = detections_to_bboxes(dir_path, "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up.ply", detections) #  TODO: get the bounding boxes from the detections


    # this will contain all the detected non-overlapping boxes
    registered_boxes = []
    for bbox, conf in bboxes_3d:     
        for idx, (reg_box, confidence) in enumerate(registered_boxes):
            iou = calculate_iou(reg_box, bbox)
            if iou > 0.2:  # Check if the overlap is greater than 20%, TODO: is this value reasonable?
                if conf > confidence:
                    # Replace the existing box with the new one
                    registered_boxes[idx] = (bbox, conf)
                break
        else:
            registered_boxes.append((bbox, conf))
    
    final_bboxes = [bbox for (bbox, _) in registered_boxes]
    pcd_original = o3d.io.read_point_cloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up.ply")
    o3d.visualization.draw_geometries([pcd_original] + final_bboxes)


if __name__ == "__main__":
    register_drawers("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1")
