import numpy as np
from tqdm  import tqdm
import os
import cv2
from time import time
import open3d as o3d
from kitti_semantic_dataset import KittiSemanticDataset
    
def preprocess_dataset(data: KittiSemanticDataset, visualize: bool = False, ignore_moving: bool = True, max_distance: int = 70, foresight_range: int = 50) -> None:
    t_start = time()
    for seq_idx, seq in tqdm(enumerate(data.sequences)): # For each sequence
        for pose_idx in tqdm(range(len(data.poses[seq_idx]) - foresight_range)): # Go over every pose
            pose = data.poses[seq_idx][pose_idx]
            covisibility_mask = []
            if visualize:
                imgs = data._load_image_pair(seq_idx, pose_idx)
                left_img, right_img = imgs[0], imgs[1]
            for next_scan_idx in range(1, foresight_range): # Check the next scans
                target_scan, target_label, target_pose = data._load_pointcloud(seq_idx, pose_idx+next_scan_idx), data.labels[seq_idx][pose_idx+next_scan_idx], data.poses[seq_idx][pose_idx+next_scan_idx]
                mask = np.zeros(len(target_scan))
                if np.linalg.norm(pose[:3,3]-target_pose[:3,3]) > max_distance:
                    covisibility_mask.append(mask)
                    continue
                if visualize:
                    frustum_list = []
                    color_list = []
                for idx, (point3d, label) in enumerate(zip(target_scan[::50], target_label[::50])): # Go overy every point and project it into scan
                    if ignore_moving and label > 250: # Ignore moving objects
                        continue
                    point = np.ones_like(point3d)
                    point[:3] = point3d[:3] # Make coordinate homogenous
                    world_point = target_pose.dot(data.calib[seq_idx]["T_w_lidar"]).dot(np.array(point))
                    
                    # Project into cam0
                    pixel_cam0 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(pose.dot(world_point))[:3])
                    pixel_cam0[0]/= pixel_cam0[2]
                    pixel_cam0[1]/= pixel_cam0[2]
                    if max_distance > pixel_cam0[2] > 0 and 0 <= pixel_cam0[0] < data.target_image_size[1] and 0 <= pixel_cam0[1] < data.target_image_size[0]:
                        if visualize:
                            left_img[int(pixel_cam0[1]), int(pixel_cam0[0])] =  data.color_map[label]
                            frustum_list.append(point)
                            color_list.append(data.color_map[label])
                        mask[idx] = 1

                    # Project into cam1
                    pixel_cam1 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam1"].dot(pose.dot(world_point))[:3])
                    pixel_cam1[0]/= pixel_cam1[2]
                    pixel_cam1[1]/= pixel_cam1[2]
                    if max_distance > pixel_cam1[2] > 0 and 0 <= pixel_cam1[0] < data.target_image_size[1] and 0 <= pixel_cam1[1] < data.target_image_size[0]:
                        if visualize:
                            right_img[int(pixel_cam1[1]), int(pixel_cam1[0])] =  data.color_map[label]
                            frustum_list.append(point)
                            color_list.append(data.color_map[label])
                        mask[idx] = 1

                covisibility_mask.append(mask)
                # Visualize scan projection
                if visualize:
                    pcd = o3d.geometry.PointCloud()
                    v3d = o3d.utility.Vector3dVector
                    pcd.points = v3d(np.array(frustum_list)[:, :3])
                    pcd.colors = v3d(np.array(color_list))
                    cv2.imshow("Projected LiDAR Scan", np.vstack([left_img, right_img]))
                    cv2.waitKey(0)
                    o3d.visualization.draw_geometries([pcd])

            # Build covisibility array
            if not os.path.isdir(os.path.join(data.base_path, f"covisibility/{seq}")):
                os.makedirs(os.path.join(data.base_path, f"covisibility/{seq}/"))
            cov_path = os.path.join(data.base_path, f"covisibility/{seq}/{pose_idx:06d}.npz")
            if os.path.isfile(cov_path):
                os.remove(cov_path)
            max_scan_length = max([len(m) for m in covisibility_mask])
            padded_cov_mask = np.zeros([len(data.poses[seq_idx]), max_scan_length])
            for i, cv in enumerate(covisibility_mask):
                padded_cov_mask[i,:len(cv)] = np.array(cv)
            np.savez_compressed(cov_path, np.array(padded_cov_mask))
            # exit()
    print(f"Total time {int(time() - t_start)} sec")


if __name__ == "__main__":
    path = "/Volumes/External 1/dataset"

    # Preprocess Train data
    train_data = KittiSemanticDataset(path, train=True, target_image_size=(370,1226))
    print("Load success")
    preprocess_dataset(train_data, visualize=False)

