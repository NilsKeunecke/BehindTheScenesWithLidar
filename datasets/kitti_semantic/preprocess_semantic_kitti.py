import numpy as np
from tqdm  import tqdm
import os
import cv2
from copy import deepcopy
from time import time
import open3d as o3d
from kitti_semantic_dataset import KittiSemanticDataset
    
def preprocess_dataset(data: KittiSemanticDataset, visualize: bool = False, ignore_moving: bool = True, foresight_range: int = 50) -> None:
    t_start = time()
    for seq_idx, seq in tqdm(enumerate(data.sequences)): # For each sequence
        for pose_idx in tqdm(range(len(data.poses[seq_idx]) - foresight_range)): # Go over every pose
            pose = data.poses[seq_idx][pose_idx]
            point_list = []
            if visualize:
                imgs = data.load_image_pair(seq_idx, pose_idx)
                left_img, right_img = imgs[0], imgs[1]
            for next_scan_idx in range(1, foresight_range+1): # Check the next scans
                target_scan, target_label, target_pose = data.load_pointcloud(seq_idx, pose_idx+next_scan_idx), data.labels[seq_idx][pose_idx+next_scan_idx], data.poses[seq_idx][pose_idx+next_scan_idx]
                
                # Lidar to world
                lidar_points = deepcopy(target_scan)
                lidar_points[:, 3] = 1.0
                world_points = target_pose.dot(data.calib[seq_idx]["T_w_lidar"]).dot(lidar_points.T).T

                # World to cameras
                scan_pts_im0 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(pose.dot(world_points.T))[:3, :]).T
                scan_pts_im1 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam1"].dot(pose.dot(world_points.T))[:3, :]).T
                scan_pts_im0[:, :2] = scan_pts_im0[:, :2] / scan_pts_im0[:, 2][..., None]
                scan_pts_im1[:, :2] = scan_pts_im1[:, :2] / scan_pts_im1[:, 2][..., None]

                # check if in bounds of either image
                val_inds = ((scan_pts_im0[:, 0] >= 0) & (scan_pts_im0[:, 1] >= 0)) | ((scan_pts_im1[:, 0] >= 0) & (scan_pts_im1[:, 1] >= 0))
                val_inds = val_inds & (((scan_pts_im0[:, 0] < data.target_image_size[1]) & (scan_pts_im0[:, 1] < data.target_image_size[0])) | ((scan_pts_im1[:, 0] < data.target_image_size[1]) & (scan_pts_im1[:, 1] < data.target_image_size[0])))
                val_inds = val_inds & ((scan_pts_im0[:, 2] > 0) | (scan_pts_im1[:, 2] > 0))
                
                if ignore_moving:
                    val_inds = val_inds & (target_label < 250)
                target_scan_data_point = np.zeros([target_scan[val_inds].shape[0], 7])
                target_scan_data_point[:, :3] = target_scan[val_inds, :3]
                target_scan_data_point[:, 3:6] = target_pose[:3, 3]
                target_scan_data_point[:, 6] = target_label[val_inds]
                point_list.append(target_scan_data_point)

                # Visualize scan projection
                if visualize:
                    pcd = o3d.geometry.PointCloud()
                    v3d = o3d.utility.Vector3dVector
                    pcd.points = v3d(target_scan[val_inds, :3])
                    pcd.colors = v3d(np.array([data.color_map[x] for x in np.array(target_label)[val_inds]]))
                    cv2.imshow("Projected LiDAR Scan", np.vstack([left_img, right_img]))
                    cv2.waitKey(0)
                    o3d.visualization.draw_geometries([pcd])

            # Save all points in new pcd
            if not os.path.isdir(os.path.join(data.base_path, f"merged_scans/{seq}")):
                os.makedirs(os.path.join(data.base_path, f"merged_scans/{seq}/"))
            ms_path = os.path.join(data.base_path, f"merged_scans/{seq}/{pose_idx:06d}.npz")
            if os.path.isfile(ms_path):
                os.remove(ms_path)
            merged_scan = np.concatenate(point_list, axis=0, dtype=np.float16)
            np.savez_compressed(ms_path, merged_scan)
    print(f"Total time {int(time() - t_start)} sec")


if __name__ == "__main__":
    path = "/Users/nilskeunecke/semantic-kitti_partly"

    # Preprocess Train data
    train_data = KittiSemanticDataset(path, train=True, target_image_size=(370,1226))
    print("Load success")
    preprocess_dataset(train_data, visualize=False)

