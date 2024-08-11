import open3d as o3d
import numpy as np
import json, cv2
import matplotlib.pyplot as plt

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

def detections_to_bboxes(points, detections):
    bboxes_3d = []
    for file, _, confidence, bbox in detections:
        intrinsics, extrinsics = parse_json(file + ".json")
        image = cv2.imread(file + ".jpg")
        width, height = image.shape[1], image.shape[0]

        if confidence > 0.8:
            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            _, points_3d = project_points_bbox(points, extrinsics, intrinsics, width, height, bbox,)
            pcd_bbox = o3d.geometry.PointCloud()
            pcd_bbox.points = o3d.utility.Vector3dVector(points_3d)
            _, inliers = pcd_bbox.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            pcd_bbox = pcd_bbox.select_by_index(inliers) 
            bbox_3d = pcd_bbox.get_minimal_oriented_bounding_box()
            bboxes_3d += [(bbox_3d, confidence)]

       
    return bboxes_3d
