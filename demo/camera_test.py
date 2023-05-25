import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from pyntcloud import PyntCloud


def pcd_frame():
    pass


def img_stream():
    while True:
        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()

        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Display the depth and color images
        cv2.imshow('Depth Image', depth_image)
        cv2.imshow('Color Image', color_image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def pcd_test():
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic)


    # Create point cloud from depth and color frames
    # pc = rs.pointcloud()
    # pc.map_to(color_frame)
    # pcd = pc.calculate(depth_frame)

    # pcd.export_to_ply("1.ply", color_frame);
    #
    # # ply_point_cloud = o3d.data.PLYPointCloud()
    # # print(ply_point_cloud.path)
    # pcd = o3d.io.read_point_cloud("1.ply")
    # print(pcd)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print(1)


def main():
    pcd_test()


if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.depth, 0)
    # config.enable_stream(rs.stream.color, 0)
    config.enable_device('115422250069')
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    try:
        main()
    finally:
        pipeline.stop()
        # cv2.destroyAllWindows()
