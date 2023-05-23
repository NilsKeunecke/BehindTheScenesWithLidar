from __future__ import annotations
from typing import Any
import open3d as o3d
import numpy as np
from tqdm import tqdm
import struct
import os
import yaml
import cv2

def read_pose(path: str, idx: int) -> np.ndarray[Any, float]:
    pose = np.eye(4)
    cam2body = np.zeros_like(pose)
    cam2body = np.array([[0,0,1,0],[1,0,0,0], [0,1,0,0], [0,0,0,1]])
    with open(path, "rb") as f:
        pose[:3, :] = np.fromstring(f.readlines()[idx], dtype=float, sep=' ').reshape([3, 4])
    # pose = cam2body.T.dot(pose)
    # pose_data = np.loadtxt(path)
    # poses_seq = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
    # poses_seq = np.concatenate((poses_seq, np.zeros_like(poses_seq[:, :1, :])), axis=1)
    # poses_seq[:, 3, 3] = 1
    # print(poses_seq, pose)
    # exit()
    return pose


def label_to_color(label_path: str, color_map_path: str) -> np.ndarray[Any, Any]:
    label_list = []
    with open (label_path, "rb") as f:
        byte = f.read(4)
        while byte:
            label, _ = struct.unpack("hh", byte)
            label_list.append(label)
            byte = f.read(4)
    color_map = yaml.safe_load(open(color_map_path, 'r'))["color_map"]
    return [color_map[label] for label in label_list]

def visualize_data_point(lidar_path: str, color_list: list[tuple], pose: np.ndarray[Any, float]) -> None:
    size_float = 4
    pcd_list = []
    label_list = []
    with open (lidar_path, "rb") as f:
        byte = f.read(size_float*4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            pcd_list.append([x, y, z])
            byte = f.read(size_float*4)
 
    points = [pose[:3, :3].dot(np.array(point)) for point in pcd_list]
    points += pose[:3, 3]
    points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(points)
    pcd.colors = v3d(color_list)
    return pcd


if __name__ == "__main__":
    sequence = "00"
    base_path = "/Volumes/External/dataset/sequences/"
    pose_path =  os.path.join("/Volumes/External/dataset/poses_dvso/00.txt")
    semantic_kitti_yaml_path = "/Users/nilskeunecke/BehindTheScenes/datasets/kitti_semantic/semantic-kitti.yaml"
    pcd_list=[]

    for scan in tqdm(range(10)):
        lidar_path = os.path.join(base_path,  f"{sequence}/velodyne/{scan:06d}.bin")
        label_path = os.path.join(base_path, f"{sequence}/labels/{scan:06d}.label")
        pose = read_pose(path=pose_path, idx=scan)
        print(pose)
        color_list = label_to_color(label_path=label_path, color_map_path=semantic_kitti_yaml_path)

        pcd_list.append(visualize_data_point(lidar_path=lidar_path, color_list=color_list, pose=pose))

        #load Image
        img2 = cv2.imread(os.path.join(base_path, f"{sequence}/image_2/{scan:06d}.png"))
        img3 = cv2.imread(os.path.join(base_path, f"{sequence}/image_3/{scan:06d}.png"))
        joint_img = np.vstack([img2, img3])
        cv2.imshow("Current camera frames", joint_img)
        o3d.visualization.draw_geometries(pcd_list)

