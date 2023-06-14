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

            # Iterate over the next scans (default=50)
            for next_scan_idx in range(0, foresight_range+1):
                target_scan, target_label, target_pose = data.load_pointcloud(seq_idx, pose_idx+next_scan_idx), data.labels[seq_idx][pose_idx+next_scan_idx], data.poses[seq_idx][pose_idx+next_scan_idx]
                
                # Lidar to world
                lidar_points = deepcopy(target_scan)
                lidar_points[:, 3] = 1.0
                world_points = target_pose.dot(data.calib[seq_idx]["T_w_lidar"]).dot(lidar_points.T).T

                # World to cameras
                scan_pts_im0 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(np.linalg.inv(pose).dot(world_points.T))[:3, :]).T
                scan_pts_im1 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam1"].dot(np.linalg.inv(pose).dot(world_points.T))[:3, :]).T
                scan_pts_im0[:, :2] = scan_pts_im0[:, :2] / scan_pts_im0[:, 2][..., None]
                scan_pts_im1[:, :2] = scan_pts_im1[:, :2] / scan_pts_im1[:, 2][..., None]

                # check if in bounds of either image
                val_inds = ((scan_pts_im0[:, 0] >= -1) & (scan_pts_im0[:, 1] >= -1)) | ((scan_pts_im1[:, 0] >= -1) & (scan_pts_im1[:, 1] >= -1))
                val_inds = val_inds & (((scan_pts_im0[:, 0] < 1) & (scan_pts_im0[:, 1] < 1)) | ((scan_pts_im1[:, 0] < 1) & (scan_pts_im1[:, 1] < 1)))
                val_inds = val_inds & ((scan_pts_im0[:, 2] > 0) | (scan_pts_im1[:, 2] > 0))
                
                if ignore_moving and next_scan_idx != 0:
                    val_inds = val_inds & (target_label < 250)
                target_scan_data_point = np.zeros([target_scan[val_inds].shape[0], 5])
                target_scan_data_point[:, :3] = world_points[val_inds, :3]
                target_scan_data_point[:, 3] = pose_idx + next_scan_idx
                target_scan_data_point[:, 4] = target_label[val_inds]
                point_list.append(target_scan_data_point)

                # Visualize scan projection
                if visualize:
                    print("Info ", pose_idx, pose_idx+next_scan_idx, pose, target_pose)
                    pcd = o3d.geometry.PointCloud()
                    v3d = o3d.utility.Vector3dVector
                    pcd.points = v3d(world_points[val_inds, :3])
                    pcd.colors = v3d(np.array([data.color_map[x] for x in np.array(target_label)[val_inds]]))
                    left_img_points = np.copy(left_img)
                    for point, color in zip(scan_pts_im0[val_inds, :2], target_label[val_inds]):
                        color = data.label_to_color(color)
                        color.reverse()
                        cv2.circle(left_img_points, (int((point[0] *.5 + 0.5) * data.target_image_size[1]), int((point[1] *.5 + 0.5) * data.target_image_size[0])), 1, color, 1)
                    cv2.imshow("Projected LiDAR Scan", np.vstack([left_img_points, right_img]))
                    cv2.waitKey(0)
                    o3d.visualization.draw_geometries([pcd])
                    exit()

            # Save all points in new pcd
            if not os.path.isdir(os.path.join(data.base_path, f"merged_scans/{seq}")):
                os.makedirs(os.path.join(data.base_path, f"merged_scans/{seq}/"))
            ms_path = os.path.join(data.base_path, f"merged_scans/{seq}/{pose_idx:06d}.npz")
            if os.path.isfile(ms_path):
                os.remove(ms_path)
            merged_scan = np.concatenate(point_list, axis=0)

            # Subsample the merged scan to save memory
            indices = np.arange(0, merged_scan.shape[0])
            np.random.shuffle(indices)
            merged_scan = merged_scan[indices[:2048*100]] # Arbitrary to downsize merged_scans
            np.savez_compressed(ms_path, merged_scan.astype(np.float16))
    print(f"Total time {int(time() - t_start)} sec")


if __name__ == "__main__":
    path = "/Users/nilskeunecke/semantic-kitti_partly"

    # Preprocess Train data
    train_data = KittiSemanticDataset(path, train=True, target_image_size=(370,1226))
    print("Load success")
    preprocess_dataset(train_data, visualize=False, ignore_moving=True)

