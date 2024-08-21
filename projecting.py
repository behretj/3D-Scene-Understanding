import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import json, cv2, os, glob, pickle
import matplotlib.pyplot as plt
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from PIL import Image
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose, get_nearest_pose
from utils import crop_image
from scipy.spatial import cKDTree

def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    intrinsics = np.array(data["intrinsics"]).reshape(3, 3)
    # projection_matrix = np.array(data["projectionMatrix"]).reshape(4, 4)
    camera_pose = np.array(data["cameraPoseARFrame"]).reshape(4, 4)
    return intrinsics, camera_pose

def draw_box(image, box, width, color=(0, 255, 0), thickness=2):
    for i in range(4):
        p1 = (width - int(box[i][0]), int(box[i][1]))
        p2 = (width - int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1]))
        image = cv2.line(image, p1, p2, color, thickness)
    
    scale_percent = 20  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # transformed_bbox_2d_resized_ = pixels_bbox * (scale_percent / 100)
    cv2.imshow('Transformed Bounding Box', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_3d_points(points, color, name):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def create_line(p1, p2, color=[1, 0, 0]):
    """ Create a line from point p1 to point p2 for debugging purposes. """
    
    points = [p1, p2]
    lines = [[0, 1]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def visualize_points_on_image(image_path, points_2d):
    ''' For debugging purposes. '''
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='r', marker='o')
    plt.savefig("points_on_image.png")

def compute_rotation_matrix(normal_dst):
    normal_src = np.array([0, 0, 1])
    normal_dst = normal_dst / np.linalg.norm(normal_dst)
    
    v = np.cross(normal_src, normal_dst)
    c = np.dot(normal_src, normal_dst)
    s = np.linalg.norm(v)
    
    vx = np.array([[0, -v[2], v[1]], 
                   [v[2], 0, -v[0]], 
                   [-v[1], v[0], 0]])
    
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    return R

def project_points_bbox(points_3d, extrinsics, intrinsics, width, height, bbox, grid=15):
    extrinsics = np.linalg.inv(extrinsics)
    
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    extrinsics =  np.zeros((3,4))
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t

    points_cam = extrinsics @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    points_cam = points_cam.T
    
    points = intrinsics @ extrinsics @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T

    image_points = points[:2, :] / points[2, :]
    image_points = image_points.T

    
    depth_buffer = np.full((height//grid, width//grid), -np.inf)
    best_points = np.zeros((height//grid, width//grid, 3))
    best_cam_points = np.zeros((height//grid, width//grid, 3))
    
    # full_bbox = bbox.copy()
    # correct x-coordinate of bbox:
    bbox[0], bbox[2] = width - bbox[2], width - bbox[0]
    bbox = bbox // grid  
    
    for point, img_pt, cam_pt in zip(points_3d, image_points, points_cam):
        x, y = int(img_pt[0]//grid), int(img_pt[1]//grid)
        # TODO: < vs. <= ?
        if int(bbox[0]) <= x < int(bbox[2]) and int(bbox[1]) <= y < bbox[3]:
            if cam_pt[2] > depth_buffer[y, x]:  # Since z is negative, a smaller value means closer
                depth_buffer[y, x] = cam_pt[2]
                best_points[y, x] = point
                best_cam_points[y, x] = cam_pt
    
    # Filter valid points and their 2D projections
    valid = (depth_buffer != -np.inf)
    valid_points_3d = best_points[valid]
    valid_cam_points = best_cam_points[valid]
    y_indices, x_indices = np.where(valid)
    valid_image_points = np.vstack((x_indices*grid, y_indices*grid)).T

    # ### Segment the plane in camera space
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(valid_cam_points)
    # pcd.paint_uniform_color([0,1,0])
    # eq, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    
    # bbox_center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
    # bbox_depth = depth_buffer[int(bbox_center[1]), int(bbox_center[0])]
    
    # bbox_2d_pixels = np.array([[width-full_bbox[0], full_bbox[1]], [width-full_bbox[0], full_bbox[3]], [width-full_bbox[2], full_bbox[3]], [width-full_bbox[2], full_bbox[1]]])
    # bbox_2d_center = np.array([(2*width-full_bbox[0]-full_bbox[2])//2, (full_bbox[1]+full_bbox[3])//2, 1])
    
    # pixels_bbox = np.hstack((bbox_2d_pixels, np.ones((bbox_2d_pixels.shape[0], 1))))
    
    # unproject_bbox = (np.linalg.inv(intrinsics) @ pixels_bbox.T).T * bbox_depth
    # center_bbox = (np.linalg.inv(intrinsics) @ bbox_2d_center.T).T * bbox_depth
    
    # normal_dst = np.asarray(eq[:3])
    # R = compute_rotation_matrix(normal_dst)

    # unproject_bbox -= center_bbox
    # transformed_bbox = (R @ unproject_bbox.T).T
    # transformed_bbox += center_bbox

    # bbox_2d_pixels_transformed = intrinsics @ transformed_bbox.T

    # bbox_2d_pixels_transformed = bbox_2d_pixels_transformed[:2, :] / bbox_2d_pixels_transformed[2, :]
    # bbox_2d_pixels_transformed = bbox_2d_pixels_transformed.T
    # # TODO: next steps with the transformed 2d pixels??
    
    return valid_image_points, valid_points_3d

def detections_to_bboxes(points, detections, threshold=0.7):
    bboxes_3d = []
    for file, _, confidence, bbox in detections:
        intrinsics, extrinsics = parse_json(file + ".json")
        image = cv2.imread(file + ".jpg")
        width, height = image.shape[1], image.shape[0]

        if confidence > threshold:
            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            _, points_3d = project_points_bbox(points, extrinsics, intrinsics, width, height, bbox)
            # TODO: where should I put this sanity check?
            if points_3d.shape[0] < 15:
                continue
            pcd_bbox = o3d.geometry.PointCloud()
            pcd_bbox.points = o3d.utility.Vector3dVector(points_3d)
            _, inliers = pcd_bbox.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            pcd_bbox = pcd_bbox.select_by_index(inliers) 
            bbox_3d = pcd_bbox.get_minimal_oriented_bounding_box()
            bboxes_3d += [(bbox_3d, confidence)]

       
    return bboxes_3d


def get_mask_points(scan_dir, img_id):
    ### Load the necessary files
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    vrs_file = vrs_files[0]
    filename = os.path.splitext(os.path.basename(vrs_file))[0]

    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"
    
    camera_label = "camera-rgb"
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    stream_id = provider.get_stream_id_from_label(camera_label)
    w, h = calib.get_image_size()

    closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    query_timestamp = provider.get_image_data_by_index(stream_id, img_id)[1].capture_timestamp_ns

    device_pose = get_nearest_pose(closed_loop_traj, query_timestamp)

    T_world_device = device_pose.transform_world_device

    T_device_rgb_camera = calib.get_transform_device_camera()
    T_world_rgb_camera = T_world_device @ T_device_rgb_camera

    extrinsics = T_world_rgb_camera.to_matrix()

    mesh = o3d.io.read_triangle_mesh("SceneGraph-Drawer/D-Lab-Scan/textured_output.obj")
    pose = np.load("SceneGraph-Drawer/clock_plant/icp_transformation.npy")
    mesh.transform(pose)

    # clock is mask 10
    mask = np.loadtxt("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Drawer/D-Lab-Scan/pred_mask/010.txt")

    points = np.asarray(mesh.vertices)
    points = points[mask.astype(bool)]

    pcd = o3d.io.read_point_cloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Drawer/clock_plant/aria_pointcloud.ply")

    aria_points = np.asarray(pcd.points)

    print(aria_points.shape, points.shape)

    kdtree = cKDTree(aria_points)

    _, indices = kdtree.query(points, k=1)

    closest_points = aria_points[indices]

    points_cam = np.dot(np.linalg.inv(extrinsics), np.hstack((closest_points, np.ones((closest_points.shape[0], 1)))).T)
    points_cam = points_cam.T[:, :3]

    image_data = provider.get_image_data_by_index(stream_id, img_id)
    raw_image = image_data[0].to_numpy_array()
    pinhole = calibration.get_linear_camera_calibration(w, h, calib.get_focal_lengths()[0])
    undistorted_image = calibration.distort_by_calibration(raw_image, pinhole, calib)
    aruco_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

    points_list = []
    
    for point in points_cam:
        image_point = calib.project(point)
        if image_point is None:
            continue
        image_point_int = np.round(image_point).astype(int)
        x, y = image_point_int
        points_list.append([0, x, y])
        aruco_image = cv2.circle(aruco_image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # Red color
    
    points_array = np.array(points_list)
    points_array = np.unique(points_array, axis=0)
    np.save("queries.npy", points_array)
    
    output_image_path = "mask_shelf.jpg"
    aruco_image = cv2.rotate(aruco_image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(output_image_path, aruco_image)



if __name__ == "__main__":
    get_mask_points("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Drawer/clock_plant", 184)