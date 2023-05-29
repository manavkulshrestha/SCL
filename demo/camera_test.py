import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from matplotlib import pyplot as plt
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

    for i in range(100):
        frames = pipeline.wait_for_frames()
        print(i)

    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # plt.imshow(color_image)
    # plt.show()
    # plt.imshow(depth_image)
    # plt.show()

    points = rs.pointcloud()
    points.map_to(color_frame)
    point_cloud = points.calculate(depth_frame)

    point_cloud.export_to_ply("1.ply", color_frame)

    pcd = o3d.io.read_point_cloud("1.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    print(1)


def main():
    pcd_test()


if __name__ == '__main__':
    # pipeline = rs.pipeline()
    # config = rs.config()
    # # config.enable_device('115422250069')
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 1280 × 720 is max depth res
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    # pipeline.start(config)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 0)  # Enable depth stream
    config.enable_stream(rs.stream.color, 0)  # Enable color stream
    pipeline.start(config)

    try:
        main()
    finally:
        pipeline.stop()
