import open3d as o3d
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    intrinsics = np.array(data["intrinsics"]).reshape(3, 3)
    # projection_matrix = np.array(data["projectionMatrix"]).reshape(4, 4)
    camera_pose = np.array(data["cameraPoseARFrame"]).reshape(4, 4)
    return intrinsics, camera_pose

def get_rotation_matrix_from_normals(n1, n2):
    # Initial normal vector of the bounding box
    n1 = np.array([0, 0, 1])
    # Estimated normal vector of the plane
    n2 = np.array([-0.52742702, -0.02515729, 0.84922779])

    # Calculate the rotation axis
    axis = np.cross(n1, n2)
    axis_unit = axis / np.linalg.norm(axis)

    # Calculate the rotation angle
    cos_theta = np.dot(n1, n2)
    theta = np.arccos(cos_theta)

    # Rodrigues' rotation formula components
    K = np.array([
        [0, -axis_unit[2], axis_unit[1]],
        [axis_unit[2], 0, -axis_unit[0]],
        [-axis_unit[1], axis_unit[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return R

def calculate_rotation_matrix(normal_vector):
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Initial normal vector [0, 0, -1]
    initial_normal = np.array([0, 0, -1])
    
    # Calculate the cross product to find the rotation axis
    rotation_axis = np.cross(initial_normal, normal_vector)
    
    # Calculate the sine and cosine of the rotation angle
    cos_theta = np.dot(initial_normal, normal_vector)
    sin_theta = np.linalg.norm(rotation_axis)
    
    # Normalize the rotation axis
    if sin_theta != 0:
        rotation_axis = rotation_axis / sin_theta
    
    # Rodrigues' rotation formula components
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    
    I = np.eye(3)
    
    # Rotation matrix using Rodrigues' formula
    R = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    return R

def draw_box(image, box, color=(0, 255, 0), thickness=2):
    for i in range(4):
        p1 = (int(box[i][0]), int(box[i][1]))
        p2 = (int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1]))
        image = cv2.line(image, p1, p2, color, thickness)
    cv2.imshow('Transformed Bounding Box', image)
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

def transform_2d_bbox_3(pixels_bbox, equation, intrinsics):
    # Draw the transformed bounding box (example using OpenCV)
    image = cv2.imread("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/frame_00359.jpg")
    # Resize the image to a smaller size for display
    scale_percent = 20  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # transformed_bbox_2d_resized_ = pixels_bbox * (scale_percent / 100)

    # draw_box(resized_image, transformed_bbox_2d_resized_.astype(int))


    a, b, c, d = equation

    pixels_bbox = np.hstack((pixels_bbox, np.ones((pixels_bbox.shape[0], 1))))
    
    print("2D Bounding Box Points:", pixels_bbox)
    
    unproject_bbox = (np.linalg.inv(intrinsics) @ pixels_bbox.T).T

    print("Unprojected 2D Bounding Box Points:", unproject_bbox)

    def compute_depth(x, y):
        return (-d - a*x - b*y) / c
    
    depths = np.array([compute_depth(x, y) for x, y in unproject_bbox[:, :2]])
    
    bbox_3d = np.hstack((unproject_bbox[:, :2], -depths[:, np.newaxis]))
    # bbox_3d = unproject_bbox

    print("3D Bounding Box Points:", bbox_3d)

    # visualize_3d_points(bbox_3d, [1, 0, 0], "Back-projected 3D points")

    # R = calculate_rotation_matrix(np.array([a, b, c]))

    # print("Rotation Matrix:", R)

    # transformed_bbox_3d = np.dot(R, bbox_3d.T).T

    # print("Transformed 3D Bounding Box Points:", transformed_bbox_3d)

    projected_bbox_2d_homogeneous = np.dot(intrinsics, bbox_3d.T).T

    print("Projected 2D Bounding Box Points (Homogeneous):", projected_bbox_2d_homogeneous)

    projected_bbox_2d = projected_bbox_2d_homogeneous[:, :2] / projected_bbox_2d_homogeneous[:, 2, np.newaxis]

    print("Projected 2D Bounding Box Points:", projected_bbox_2d)

    # Scale the transformed points to the resized image dimensions
    transformed_bbox_2d_resized = projected_bbox_2d * (scale_percent / 100)

    draw_box(resized_image, transformed_bbox_2d_resized.astype(int))

def transform_2d_bbox_2(bbox_3d, equation, intrinsics):
    # Plane normal vector and d value
    a, b, c, _ = equation

    # # Corresponding depth values for each point (replace with actual depth values)
    # depth_values = -np.ones((bbox_2d_pixels.shape[0], 1))

    # # Convert 2D points to 3D using depth values
    # bbox_3d = np.hstack((bbox_2d_pixels, depth_values))

    # print(bbox_3d)

    # bbox_3d = np.linalg.inv(intrinsics) @ bbox_3d.T

    # bbox_3d = bbox_3d.T

    print("3D Bounding Box Points:", bbox_3d)

    # Compute the rotation matrix to align the normal vector with the plane normal
    def compute_rotation_matrix(normal_from, normal_to):
        normal_from = normal_from / np.linalg.norm(normal_from)
        normal_to = normal_to / np.linalg.norm(normal_to)
        v = np.cross(normal_from, normal_to)
        s = np.linalg.norm(v)
        c = np.dot(normal_from, normal_to)
        I = np.eye(3)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        return R

    # Normal vector of the bounding box in image coordinates (facing the camera)
    normal_image = np.array([0, 0, -1])

    # Normal vector of the plane
    normal_plane = np.array([a, b, c])

    # Compute the rotation matrix
    R = compute_rotation_matrix(normal_image, normal_plane)

    # Transform the 3D bounding box points
    transformed_bbox_3d = np.dot(R, bbox_3d.T).T

    print("Transformed 3D Bounding Box Points:", transformed_bbox_3d)

    # Project points back to 2D
    transformed_bbox_2d_homogeneous = np.dot(intrinsics, transformed_bbox_3d.T).T
    transformed_bbox_2d = transformed_bbox_2d_homogeneous[:, :2] / transformed_bbox_2d_homogeneous[:, 2, np.newaxis]

    print("Transformed 2D Bounding Box Points:", transformed_bbox_2d)

def transform_2d_bbox(bbox_3d, equation, intrinsics):
    # Plane normal vector and d value
    a, b, c, d = equation

    # Example 2D bounding box corners in pixel coordinates (replace with actual coordinates)
    # bbox_2d_pixels = np.array([
    #     [x1, y1],
    #     [x2, y1],
    #     [x2, y2],
    #     [x1, y2]
    # ])

    # Compute the z values for each bounding box point
    # def compute_z(x, y):
    #     return (-d - a*x - b*y) / c

    # # Lift points to 3D using the calculated z values
    # bbox_3d = np.array([[x, y, compute_z(x, y)] for x, y in bbox_2d_pixels])
    
    # Normal vector of the plane
    n = np.array([a, b, c])
    v = np.array([0, 0, -1])

    # Compute rotation matrix to align n with v
    r = np.cross(n, v)
    r = r / np.linalg.norm(r)

    theta = np.arccos(np.dot(n, v) / (np.linalg.norm(n) * np.linalg.norm(v)))

    K_matrix = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])

    I_matrix = np.eye(3)
    R = I_matrix + np.sin(theta) * K_matrix + (1 - np.cos(theta)) * np.dot(K_matrix, K_matrix)

    # Transform the 3D bounding box points
    transformed_bbox_3d = np.dot(R, bbox_3d.T).T

    # Project points back to 2D
    transformed_bbox_2d_homogeneous = np.dot(intrinsics, transformed_bbox_3d.T).T
    transformed_bbox_2d = transformed_bbox_2d_homogeneous[:, :2] / transformed_bbox_2d_homogeneous[:, 2, np.newaxis]

    # Draw the transformed bounding box (example using OpenCV)
    image = cv2.imread("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/frame_00359.jpg")
    # Resize the image to a smaller size for display
    scale_percent = 20  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Scale the transformed points to the resized image dimensions
    transformed_bbox_2d_resized = transformed_bbox_2d * (scale_percent / 100)

    # Draw the transformed bounding box on the resized image
    for i in range(4):
        y1, x1 = tuple(transformed_bbox_2d_resized[i].astype(int))
        y2, x2 = tuple(transformed_bbox_2d_resized[(i + 1) % 4].astype(int))
        print(x1, y1, x2, y2)
        resized_image = cv2.line(resized_image, (y1, x1), (y2, x2), (0, 0, 255), 1)

    cv2.imshow('Transformed Bounding Box', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def project_points_bbox(points_3d, extrinsics, intrinsics, width, height, bbox, grid=10):
    # transform extrinsics from cam -> world to world -> cam
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

    
    depth_buffer = np.full((height//grid, width//grid), -np.inf)  # Use np.inf to signify initially infinitely far away
    best_points = np.zeros((height//grid, width//grid, 3))  # To store the best 3D point corresponding to each pixel
    best_cam_points = np.zeros((height//grid, width//grid, 3))  # To store the best 3D point corresponding to each pixel
    
    # tmp_bbox = bbox.copy()
    # correct x-coordinate of bbox:
    bbox[0], bbox[2] = width - bbox[2], width - bbox[0]
    bbox = bbox // grid  
    
    for point, img_pt, cam_pt in zip(points_3d, image_points, points_cam):
        x, y = int(img_pt[0]//grid), int(img_pt[1]//grid)
        # TODO: < vs- <= ???
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

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(valid_cam_points)
    # eq, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    
    ### version 3
    # bbox_2d_pixels = np.array([[tmp_bbox[0], tmp_bbox[1]], [tmp_bbox[0], tmp_bbox[3]], [tmp_bbox[2], tmp_bbox[3]], [tmp_bbox[2], tmp_bbox[1]]])
    # transform_2d_bbox_3(bbox_2d_pixels, eq, intrinsics)
    
    ### Not working, version 2
    # # bbox = np.array([best_cam_points[int(bbox[1]), int(bbox[0])],
    # #                 best_cam_points[int(bbox[3]), int(bbox[0])],
    # #                 best_cam_points[int(bbox[3]), int(bbox[2])],
    # #                 best_cam_points[int(bbox[1]), int(bbox[2])]])
    
    # bbox = np.array([[int(bbox[0]*grid), int(bbox[1]*grid)],
    #                 [int(bbox[0]*grid), int(bbox[3]*grid)],
    #                 [int(bbox[2]*grid), int(bbox[3]*grid)],
    #                 [int(bbox[2]*grid), int(bbox[1]*grid)]]) 
    
    
    # transform_2d_bbox_2(bbox, eq, intrinsics)
    
    ### Not working, version 1
    # rot = get_rotation_matrix_from_normals(np.array([0, 0, 1]), np.asarray(eq[:3]))

    
    # H = intrinsics @ rot @ np.linalg.inv(intrinsics)

    # bbox_2d = np.array([
    #     [bbox[0]*grid, bbox[1]*grid],
    #     [bbox[0]*grid, bbox[3]*grid],
    #     [bbox[2]*grid, bbox[3]*grid],
    #     [bbox[2]*grid, bbox[1]*grid]
    # ], dtype=np.float32)

    # # Convert to homogeneous coordinates
    # bbox_2d_homogeneous = cv2.convertPointsToHomogeneous(bbox_2d).reshape(-1, 3).T

    # # Apply the homography
    # bbox_2d_transformed_homogeneous = H @ bbox_2d_homogeneous

    # # Convert back to 2D
    # bbox_2d_transformed = cv2.convertPointsFromHomogeneous(bbox_2d_transformed_homogeneous.T).reshape(-1, 2)
    
    return valid_image_points, valid_points_3d

# Function to visualize points on the image
def visualize_points_on_image(image_path, points_2d):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='r', marker='o')
    plt.savefig("points_on_image.png")

def detections_to_bboxes(dir_path, pcd_path, detections):
    pcd_original = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd_original.points)

    bboxes_3d = []
    for file, category, confidence, bbox in detections:
        # TODO: what are the detecitons constructed of?
        print(file, category, confidence, bbox)
        intrinsics, extrinsics = parse_json(file + ".json")
        image = cv2.imread(file + ".jpg")
        width, height = image.shape[1], image.shape[0]

        # TODO: category 2 is drawer door, but there also exists regular door (0) and regfrigerator door (3)
        if category == "cabinet door" and confidence > 0.8:
            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            _, points_3d = project_points_bbox(points, extrinsics, intrinsics, width, height, bbox, grid=15)
            pcd_bbox = o3d.geometry.PointCloud()
            pcd_bbox.points = o3d.utility.Vector3dVector(points_3d)
            _, inliers = pcd_bbox.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
            pcd_bbox = pcd_bbox.select_by_index(inliers) 
            bbox_3d = pcd_bbox.get_minimal_oriented_bounding_box()
            bboxes_3d += [(bbox_3d, confidence)]

       
    return bboxes_3d


# json_file_path = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/frame_00359.json"
# image_file_path = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/frame_00359.jpg"
# obj_file_path = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/export.obj"
# pcd_file_path = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/mesh_labelled_mask3d_dataset_1_y_up.ply"
# bboxes = np.load("tmp/bboxes.npy")
# confidences = np.load("tmp/confidences.npy")
# classes = np.load("tmp/classes.npy")

# image = cv2.imread(image_file_path)

# intrinsics, projection_matrix, camera_pose = parse_json(json_file_path)

# width, height = image.shape[1], image.shape[0]

# pcd_original = o3d.io.read_point_cloud(pcd_file_path)
# points = np.asarray(pcd_original.points)

# drawers = []

# for bbox, confidence, category in zip(bboxes, confidences, classes):
#     print(bbox, confidence, category)
#     if category == 2 and confidence > 0.7:
#         points_2d, points_3d = project_points_bbox(points, camera_pose, intrinsics, width, height, bbox, grid=15)
#         pcd_bbox = o3d.geometry.PointCloud()
#         pcd_bbox.points = o3d.utility.Vector3dVector(points_3d)
#         _, inliers = pcd_bbox.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
#         pcd_bbox = pcd_bbox.select_by_index(inliers) 
#         bbox_3d = pcd_bbox.get_minimal_oriented_bounding_box()
#         selected_indices = np.array(bbox_3d.get_point_indices_within_bounding_box(pcd_original.points))
#         pcd_tmp = o3d.geometry.PointCloud()
#         pcd_tmp.points = o3d.utility.Vector3dVector(points[selected_indices])
#         pcd_tmp.paint_uniform_color(np.random.rand(3))
#         drawers += [pcd_tmp]
    

# mesh_frame_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
# mesh_frame_marker.transform(camera_pose)

# o3d.visualization.draw_geometries([pcd_original] + drawers + [mesh_frame_marker])

### NOT working, kept it, if I need them at some later point again
# def project_points_with_projection_matrix(points_3d, projection_matrix, width, height):
#     # Homogeneous coordinates for 3D points
#     points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
#     # Project points using the projection matrix
#     points_proj = projection_matrix @ points_3d_hom.T
#     points_proj = points_proj.T
    
#     # Normalize to get image coordinates
#     points_proj[:, :2] /= points_proj[:, 2, np.newaxis]
    
#     # Check image bounds
#     valid = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < width) & \
#             (points_proj[:, 1] >= 0) & (points_proj[:, 1] < height) & \
#             (points_proj[:, 2] > 0)  # Depth must be positive in camera coordinates
    
#     return points_proj[valid, :2], points_3d[valid]
