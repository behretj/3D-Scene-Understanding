import open3d as o3d
import glob, os, cv2
import numpy as np
from projectaria_tools.core import data_provider, calibration

def get_all_images(scan_dir):
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    for vrs_file in vrs_files:
        provider = data_provider.create_vrs_data_provider(vrs_file)
        assert provider is not None, "Cannot open file"

        deliver_option = provider.get_default_deliver_queued_options()

        deliver_option.deactivate_stream_all()
        camera_label = "camera-rgb"
        stream_id = provider.get_stream_id_from_label(camera_label)
        calib = provider.get_device_calibration().get_camera_calib(camera_label)
        w, h = calib.get_image_size()
        pinhole = calibration.get_linear_camera_calibration(w, h, calib.get_focal_lengths()[0])

        image_dir = os.path.join(scan_dir, vrs_file[:-4] + "_images")
        os.makedirs(image_dir, exist_ok=True)

        for i in range(provider.get_num_data(stream_id)):
            image_data = provider.get_image_data_by_index(stream_id, i)
            img = image_data[0].to_numpy_array()
            undistorted_image = calibration.distort_by_calibration(img, pinhole, calib)
            aruco_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)
            aruco_image = cv2.rotate(aruco_image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(image_dir, f"frame_{i:05}.jpg"), aruco_image)


def mask3d_labels(label_path):
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

    mask3d_labels = np.zeros(num_lines, dtype=int)

    for i, relative_path in enumerate(file_paths):
        value = values[i]
        file_path = os.path.join(base_dir, relative_path)
        with open(file_path, 'r') as file:
            for j, line in enumerate(file):
                if line.strip() == '1':
                    mask3d_labels[j] = value

    output_path = os.path.join(base_dir, 'mask3d_labels.txt')
    with open(output_path, 'w') as file:
        for label in mask3d_labels:
            file.write(f"{label}\n")

    

def vis_detections(coords, color=[0,0,1]):
    spheres = []

    if len(coords) == 0:
        return spheres
    
    if len(coords.shape) == 1:
        coords = np.array([coords])
    
    for coord in coords:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        sphere.translate(coord)
        spheres += [sphere]
    return spheres


def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]

def filter_object(obj_dets, hand_dets):
    object_cc_list = [] # object center list
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)

    img_obj_id = [] # matching list
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0: # if hand is non-contact
            img_obj_id.append(-1)
            continue
        else: # hand is in-contact
            hand_cc = np.array(calculate_center(hand_dets[i,:4])) # hand center points
            # caculates, using the hand offset vector, which object is the closest to this object
            point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])]) # extended points (hand center + offset)
            dist = np.sum((object_cc_list - point_cc)**2,axis=1)
            dist_min = np.argmin(dist) # find the nearest 
            img_obj_id.append(dist_min)
        
    return img_obj_id
