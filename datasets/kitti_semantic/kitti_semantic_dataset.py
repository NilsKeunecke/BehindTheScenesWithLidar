from __future__ import annotations
from typing import Any
import numpy as np
import torch
import os
import struct
import logging
import cv2
import time
import open3d as o3d
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset


BASE_SIZES = {
    "00": (376, 1241),
    "01": (376, 1241),
    "02": (376, 1241),
    "03": (375, 1242),
    "04": (370, 1226),
    "05": (370, 1226),
    "06": (370, 1226),
    "07": (370, 1226),
    "08": (370, 1226),
    "09": (370, 1226),
    "10": (370, 1226),
    "11": (370, 1226),
    "12": (370, 1226),
    "13": (376, 1241),
    "14": (376, 1241),
    "15": (376, 1241),
    "16": (376, 1241),
    "17": (376, 1241),
    "18": (376, 1241),
    "19": (376, 1241),
    "20": (376, 1241),
    "21": (376, 1241),
}

class KittiSemanticDataset(Dataset):
    def __init__(self, data_path: str, train: bool, target_image_size=(370, 1226)) -> None:
        self.base_path = data_path
        self.target_image_size = target_image_size
        self.train = train
        if self.train:
            #self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            self.sequences = ['04'] # Use for testing as this is very small sequence
        else:
            self.sequences = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.color_map = yaml.safe_load(open(os.path.join(self.base_path, "semantic-kitti.yaml"), 'r'))["color_map"]
        if train:
            self.labels = self._load_labels()
        self.calib = self._load_calibrations()
        self.poses = self._load_poses()
        
        # self.merged_scans = self._load_merged_scans() # Very memory intensive
        # self.scan_list = self._load_pointclouds() # Very memory intensive
        # self.images = self._load_images() # Very memory intensive
        self.length = sum([len(x) for x in self.poses])
        super().__init__() 

    def __len__(self) -> int:
        return self.length
    
    def get_sequence_index(self, index: int):
        for dataset_index, seq_poses in enumerate(self.poses):
            if index >= len(seq_poses):
                index = index - len(seq_poses)
            else:
                return dataset_index, index
        return None, None
    
    def _load_poses(self) -> list[list[np.array[Any, Any]]]:
        """ This is the T_w2cam pose."""
        seq_pose_list = []
        for s in self.sequences:
            pose_list = []
            pose_path =  os.path.join(self.base_path, f"sequences/{s}/poses.txt")
            with open(pose_path, "rb") as f:
                data = f.readlines()
            for x in data:
                pose = np.eye(4, dtype=float)
                pose[:3, :] = np.fromstring(x, dtype=float, sep=' ').reshape([3, 4])
                pose_list.append(pose)
            seq_pose_list.append(pose_list)
        return seq_pose_list

    def _load_calibrations(self) -> list[dict]:
        calib_list = []
        for s in self.sequences:
            im_size = BASE_SIZES[s]
            calib_file = os.path.join(self.base_path, f"sequences/{s}/calib.txt")
            calib_file_data = {}
            with open(calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass
            # Create 3x4 projection matrices
            P_rect_20 = np.reshape(calib_file_data['P2'], (3, 4))
            P_rect_30 = np.reshape(calib_file_data['P3'], (3, 4))

            # Compute the rectified extrinsics from cam0 to camN
            T_cam1 = np.eye(4, dtype=np.float32)
            T_cam1[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
            T_cam2 = np.eye(4, dtype=np.float32)
            T_cam2[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

            # Create the 4x4 lidar tp ref matrix
            T_w_lidar = np.eye(4)
            T_w_lidar[:3, :4] = np.reshape(calib_file_data['Tr'], (3,4)) 

            # Assemble the K camera projection matrix
            K = P_rect_20[:3, :3]

            r_orig = im_size[0] / im_size[1]
            r_target = self.target_image_size[0] / self.target_image_size[1]

            if r_orig >= r_target:
                new_height = r_target * im_size[1]
                crop_height = im_size[0] - ((im_size[0] - new_height) // 2) * 2
                box = ((im_size[0] - new_height) // 2, 0, crop_height, int(im_size[1]))

                c_x = K[0, 2] / im_size[1]
                c_y = (K[1, 2] - (im_size[0] - new_height) / 2) / new_height

                rescale = im_size[1] / self.target_image_size[1]

            else:
                new_width = im_size[0] / r_target
                crop_width = im_size[1] - ((im_size[1] - new_width) // 2) * 2
                box = (0, (im_size[1] - new_width) // 2, im_size[0], crop_width)

                c_x = (K[0, 2] - (im_size[1] - new_width) / 2) / new_width
                c_y = K[1, 2] / im_size[0]

                rescale = im_size[0] / self.target_image_size[0]

            f_x = K[0, 0] / self.target_image_size[1] / rescale
            f_y = K[1, 1] / self.target_image_size[0] / rescale

            box = tuple([int(x) for x in box])

            # Replace old K with new K
            K[0, 0] = f_x * 2.
            K[1, 1] = f_y * 2.
            K[0, 2] = c_x * 2 - 1
            K[1, 2] = c_y * 2 - 1

            calib_list.append({
                "K": K,
                "T_w_cam0": T_cam1,
                "T_w_cam1": T_cam2,
                "T_w_lidar": T_w_lidar
            })
        return calib_list

    def _load_pointclouds(self) -> list[list[np.ndarray[Any, Any]]]:
        seq_scan_list = []
        for s in self.sequences:
            scan_list = []
            for file in sorted(os.listdir(os.path.join(self.base_path, f"sequences/{s}/velodyne"))):
                scan = np.fromfile(os.path.join(os.path.join(self.base_path, f"sequences/{s}/velodyne"), file), dtype=np.float32)
                scan = scan.reshape((-1, 4))
                scan_list.append(scan)
            seq_scan_list.append(scan_list)
        return seq_scan_list
    
    def load_pointcloud(self, seq, idx) -> np.ndarray[Any, Any]:
        scan = np.fromfile(os.path.join(self.base_path, f"sequences/{self.sequences[seq]}/velodyne/{idx:06d}.bin"), dtype=np.float32)
        return scan.reshape((-1, 4))

    def _load_labels(self) -> list[list[np.ndarray[Any, Any]]]:
        seq_label_list = []
        for s in self.sequences:
            label_list = []
            for f in sorted(os.listdir(os.path.join(self.base_path, f"sequences/{s}/labels"))):
                with open(os.path.join(os.path.join(self.base_path, f"sequences/{s}/labels"), f), "rb") as label:
                    labels = np.fromfile(label, dtype=np.uint16)
                    labels = labels[::2]
                    labels = labels.reshape((-1)).tolist()
                label_list.append(np.array(labels))
            seq_label_list.append(label_list)
        return seq_label_list

    def _load_images(self) -> list[list[list[np.ndarray[Any, Any]]]]:
        seq_image_list = []
        for s in self.sequences:
            image_list = []
            for left, right in zip(sorted(os.listdir(os.path.join(self.base_path, f"sequences/{s}/image_2"))), sorted(os.listdir(os.path.join(self.base_path, f"sequences/{s}/image_3")))):
                img2 = cv2.imread(os.path.join(self.base_path, f"sequences/{s}/image_2/{left}"))
                img3 = cv2.imread(os.path.join(self.base_path, f"sequences/{s}/image_3/{right}"))
                image_list.append([img2, img3])
            seq_image_list.append(image_list)
        return seq_image_list

    def load_image_pair(self, seq, idx) -> np.ndarray[Any, Any]:
        slurm_path = "/storage/group/dataset_mirrors/kitti_odom_grey/sequences/"
        img2 = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, f"sequences/{self.sequences[seq]}/image_2/{idx:06d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        img3 = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, f"sequences/{self.sequences[seq]}/image_3/{idx:06d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        return [img2, img3]
    
    def preprocess_image(self, img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.target_image_size:
            img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1)) * 2 - 1
        return img

    def label_to_color(self, label: int) -> None:
        if not self.color_map:
            self.color_map = yaml.safe_load(open(os.path.join(self.base_path, "semantic-kitti.yaml"), 'r'))["color_map"]
        return self.color_map[label]
    
    def _load_merged_scans(self) -> list[list[np.ndarray]]:
        seq_merged_scans = []
        for s in self.sequences:
            merged_scans_list = []
            if not os.path.isdir(os.path.join(self.base_path, f"merged_scans/{s}/")):
                logging.warning(f"No merged scans found for sequence {s}. Skipping.")
                continue
            for f in sorted(os.listdir(os.path.join(self.base_path, f"merged_scans/{s}/"))):
                with open(os.path.join(os.path.join(self.base_path, f"merged_scans/{s}/"), f), "rb") as merged_scans_file:
                    merged_scans = np.load(merged_scans_file)
                    merged_scans_list.append(merged_scans)
            seq_merged_scans.append(merged_scans_list)
        return seq_merged_scans
    
    def load_merged_scan(self, seq: int, idx: int) -> np.ndarray[Any, Any]:
        merged_scan = np.load(os.path.join(self.base_path, f"merged_scans/{self.sequences[seq]}/{idx:06d}.npz"))
        merged_scan = merged_scan["arr_0"]
        return merged_scan

    def __getitem__(self, index: int) -> Any:
        _start_time = time.time()

        sequence_index, index = self.get_sequence_index(index)
        if sequence_index is None:
            raise IndexError()
        
        imgs = self.load_image_pair(sequence_index, index)
        imgs = [self.preprocess_image(img) for img in imgs]
        pose_left = self.poses[sequence_index][index] @ self.calib[sequence_index]["T_w_cam0"]
        pose_right = self.poses[sequence_index][index] @ self.calib[sequence_index]["T_w_cam1"]
        poses = [pose_left.astype(np.float32), pose_right.astype(np.float32)]
        projs = [self.calib[sequence_index]["K"], self.calib[sequence_index]["K"]]
        merged_scan_compressed = self.load_merged_scan(sequence_index, index)
        merged_scan = np.zeros([merged_scan_compressed.shape[0], 7])
        indices = merged_scan_compressed[:, 3].astype(int)
        merged_scan[:, :3] = merged_scan_compressed[:, :3]
        merged_scan[:, 3:6] = np.array([self.poses[0][i][3, :3] for i in indices])
        merged_scan[:, 6] = merged_scan_compressed[:, 4]

        _proc_time = np.array(time.time() - _start_time)

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "depths": [],
            "merged_scan": merged_scan.astype(np.float32),
            "sequence": np.array([sequence_index], np.int32),
            "ids": np.array(index, np.int32),
            "t__get_item__": np.array([_proc_time])
        }

        return data

    def visualize_sequence(self, seq: int) -> None:
        pcd_list = []
        for idx in tqdm(range(len(self.poses[seq]))):
            pcd = o3d.geometry.PointCloud()
            v3d = o3d.utility.Vector3dVector
            if self.train:
                color_list = [self.color_map[label] for label in self.labels[seq][idx]]
            scan = self._load_pointcloud(seq, idx)
            points = np.ones_like(scan)
            points[:, :3] = scan[:, :3]
            points = np.array([self.poses[seq][idx].dot(self.calib[seq]["T_w_lidar"]).dot(np.array(point)) for point in points])
            pcd.points = v3d(points[:, :3])
            if self.train:
                pcd.colors = v3d(color_list)
            pcd_list.append(pcd)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        for pcd in pcd_list:
            vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])
        vis.run()

if __name__ == "__main__":
    dataset = KittiSemanticDataset(
        data_path="/storage/slurm/keunecke/semantickitti",
        train=True)
    x = dataset[100]
    print(x)
    # dataset.visualize_sequence(0)
