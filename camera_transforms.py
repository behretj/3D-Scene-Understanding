import cv2
import numpy as np
import open3d as o3d
from projectaria_tools.core.mps.utils import get_nearest_pose
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, image, calibration
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId


def pose_aria_pointcloud(vrsfile, image_index, marker_type=cv2.aruco.DICT_APRILTAG_36h11, save_aria_pcd=True, vis_detection=False, vis_poses=False):
    ### user-defined (vrs_file: data, image_index: image with the aruco marker)
    # vrsfile = "/home/tjark/Documents/aria_data/semantic-corner-1/f5ba674d-50f4-42ce-9adc-675e39fd5ef0.vrs"
    # image_index = 15

    provider = data_provider.create_vrs_data_provider(vrsfile)
    assert provider is not None, "Cannot open file"

    camera_label = "camera-rgb"
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    stream_id = provider.get_stream_id_from_label(camera_label)

    w, h = calib.get_image_size()
    pinhole = calibration.get_linear_camera_calibration(w, h, calib.get_focal_lengths()[0])

    image_container = provider.get_image_data_by_index(stream_id, image_index)

    raw_image = image_container[0].to_numpy_array()
    query_timestamp = image_container[1].capture_timestamp_ns

    undistorted_image = calibration.distort_by_calibration(raw_image, pinhole, calib)
    aruco_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)
    # TODO: horiontal or vertical photo?
    aruco_image = cv2.rotate(aruco_image, cv2.ROTATE_90_CLOCKWISE)

    cam_matrix = np.array([[calib.get_focal_lengths()[0], 0, calib.get_principal_point()[0]],
                        [0, calib.get_focal_lengths()[1], calib.get_principal_point()[1]],
                        [0, 0, 1]])
    
    arucoDict = cv2.aruco.getPredefinedDictionary(marker_type)
    arucoParams = cv2.aruco.DetectorParameters()

    corners, ids, _ = cv2.aruco.detectMarkers(aruco_image, arucoDict, parameters=arucoParams)

    if len(corners) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.147, cam_matrix, 0)
        rotation_3x3, _ = cv2.Rodrigues(rvecs)
        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = rotation_3x3
        T_camera_marker[:3, 3] = tvecs

        if vis_detection:
            # draw Marker border and axes
            cv2.aruco.drawDetectedMarkers(aruco_image, corners, ids)
            cv2.drawFrameAxes(aruco_image, cam_matrix, 0, rvecs, tvecs, 0.1)
            scale = 0.4
            dim = (int(aruco_image.shape[1] * scale), int(aruco_image.shape[0] * scale))
            resized = cv2.resize(aruco_image, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("aruco", resized)
            cv2.waitKey(0)

        # Point cloud creation
        global_points_path = "/home/tjark/Documents/aria_data/semantic-corner-1/mps_f5ba674d-50f4-42ce-9adc-675e39fd5ef0_vrs/slam/semidense_points.csv.gz"
        points = mps.read_global_point_cloud(global_points_path)

        # filter the point cloud using thresholds on the  inverse depth and distance standard deviation, user set
        inverse_distance_std_threshold = 0.005
        distance_std_threshold = 0.001

        filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)

        pcd = o3d.geometry.PointCloud()
        points = np.array([point.position_world for point in filtered_points])

        pcd.points = o3d.utility.Vector3dVector(points)
        # paint point cloud in random color
        pcd.paint_uniform_color(np.random.rand(3))


        closed_loop_path = "/home/tjark/Documents/aria_data/semantic-corner-1/mps_f5ba674d-50f4-42ce-9adc-675e39fd5ef0_vrs/slam/closed_loop_trajectory.csv"
        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

        t_first = provider.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
        difference_to_start = query_timestamp - t_first
        start_trajectory = int(closed_loop_traj[0].tracking_timestamp.total_seconds() * 1e9)
        query_timestamp = start_trajectory + difference_to_start
        
        pose_info = get_nearest_pose(closed_loop_traj, query_timestamp)
        assert pose_info
        T_world_device = pose_info.transform_world_device
        T_device_camera = calib.get_transform_device_camera()
        T_world_camera = T_world_device @ T_device_camera
        T_world_camera = T_world_camera.to_matrix()

        rot_z_270 = np.array([[np.cos(3 * np.pi / 2), -np.sin(3 * np.pi / 2), 0, 0],
                            [np.sin(3 * np.pi / 2), np.cos(3 * np.pi / 2), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        
        T_world_camera = np.dot(T_world_camera, rot_z_270)

        T_world_marker = np.dot(T_world_camera, T_camera_marker)

        if vis_poses:
            mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
            
            mesh_frame_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
            mesh_frame_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
            
            mesh_frame_camera.transform(T_world_camera)
            mesh_frame_marker.transform(T_world_marker)

            world_origin = np.array([0, 0, 0, 1])
            camera_origin = np.dot(T_world_camera, world_origin)[:3]
            marker_origin = np.dot(T_world_marker, world_origin)[:3]


            sphere_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            sphere_marker.paint_uniform_color([1, 0, 0]) # red
            sphere_marker.translate(marker_origin)

            sphere_camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            sphere_camera.paint_uniform_color([0, 1, 0]) # green
            sphere_camera.translate(camera_origin)

            o3d.visualization.draw_geometries([pcd, mesh_frame_world, mesh_frame_camera, mesh_frame_marker, sphere_camera, sphere_marker])
        
        if save_aria_pcd:
            o3d.io.write_point_cloud("aria_pointcloud.ply", pcd)
        
        return T_world_marker


def pose_ipad_pointcloud(image_index, marker_type=cv2.aruco.DICT_APRILTAG_36h11, vis_detection=False, vis_poses=False):
    image = cv2.imread("/home/tjark/Documents/aria_data/2024_04_15_17_23_57/frame_00030.jpg")

    # image is frame0030.jpg
    cam_matrix = np.array([[1597.8880615234375, 0, 946.49566650390625],
                            [0, 1597.8880615234375, 711.8023681640625],
                            [0, 0, 1]])
    
    arucoDict = cv2.aruco.getPredefinedDictionary(marker_type)
    arucoParams = cv2.aruco.DetectorParameters()

    # TODO: replace with actual json file
    camera_info = {
            "projectionMatrix": [1.6644667387008667, 0, 0.013546168804168701, 0, 0, 2.2192890644073486, -0.010691165924072266, 0, 0, 0, -0.9999997615814209, -0.00099999981466680765, 0, 0, -1, 0],
            "intrinsics": [1597.8880615234375, 0, 946.49566650390625, 0, 1597.8880615234375, 711.8023681640625, 0, 0, 1],
            "cameraPoseARFrame": [0.0031753906514495611, 0.9853176474571228, 0.1707017719745636, 2.1585512161254883, -0.69526088237762451, -0.12051788717508316, 0.70858144760131836, 0.034648377448320389, 0.71875041723251343, -0.12093230336904526, 0.684670090675354, -0.024456746876239777, 0, 0, 0, 0.99999994039535522]
        }


    corners, ids, _ = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if len(corners) > 0:
        # TODO: check distortion coefficients of iPad
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.15, cam_matrix, 0)
        rotation_3x3, _ = cv2.Rodrigues(rvecs)
        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = rotation_3x3
        T_camera_marker[:3, 3] = tvecs

        if vis_detection:
            # draw Marker border and axes
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.drawFrameAxes(image, cam_matrix, 0, rvecs, tvecs, 0.1)
            scale = 0.4
            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("aruco", resized)
            cv2.waitKey(0)
        
        T_world_camera = np.array(camera_info["cameraPoseARFrame"]).reshape(4, 4)
        
        rot_y_180 = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        
        T_world_camera = np.dot(T_world_camera, rot_y_180)

        T_world_marker = np.dot(T_world_camera, T_camera_marker)

        if vis_poses:
            pcd = o3d.io.read_point_cloud("/home/tjark/Documents/aria_data/tjark_scene_01_17_31_28.ply")

            mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
            
            mesh_frame_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
            mesh_frame_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
            
            mesh_frame_camera.transform(T_world_camera)
            mesh_frame_marker.transform(T_world_marker)

            world_origin = np.array([0, 0, 0, 1])
            camera_origin = np.dot(T_world_camera, world_origin)[:3]
            marker_origin = np.dot(T_world_marker, world_origin)[:3]


            sphere_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            sphere_marker.paint_uniform_color([1, 0, 0]) # red
            sphere_marker.translate(marker_origin)

            sphere_camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            sphere_camera.paint_uniform_color([0, 1, 0]) # green
            sphere_camera.translate(camera_origin)

            o3d.visualization.draw_geometries([pcd, mesh_frame_world, mesh_frame_camera, mesh_frame_marker, sphere_camera, sphere_marker])
        
        return T_world_marker

def transform_ipad_to_aria_pointcloud(pointcloud_path, T_world_marker_ipad, T_world_marker_aria):
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    T_world_marker_ipad = np.linalg.inv(T_world_marker_ipad)
    pcd.transform(T_world_marker_ipad)
    pcd.transform(T_world_marker_aria)
    o3d.io.write_point_cloud("transformed_pointcloud.ply", pcd)

