import numpy as np
from tqdm  import tqdm
import os
import cv2
from scipy import stats
from copy import deepcopy
from time import time
import open3d as o3d
from kitti_semantic_dataset import KittiSemanticDataset
    
def preprocess_dataset(data: KittiSemanticDataset, visualize: bool = False, ignore_moving: bool = True, foresight_range: int = 50) -> None:
    t_start = time()
    for seq_idx, seq in tqdm(enumerate(data.sequences)): # For each sequence
        for pose_idx in tqdm(range(0, len(data.poses[seq_idx]) - foresight_range)): # Go over every pose
            pose = data.poses[seq_idx][pose_idx]
            point_list = []
            points_in_first_scan = 0

            if visualize:
                imgs = data.load_image_pair(seq_idx, pose_idx)
                left_img, right_img = imgs[0], imgs[1]

            # Iterate over the next scans (default=50)
            for next_scan_idx in range(foresight_range+1):
                target_scan, target_label, target_pose = data.load_pointcloud(seq_idx, pose_idx+next_scan_idx), data.labels[seq_idx][pose_idx+next_scan_idx], data.poses[seq_idx][pose_idx+next_scan_idx]
                
                # Lidar to world
                lidar_points = deepcopy(target_scan)
                lidar_points[:, 3] = 1.0
                # world_points = target_pose.dot(np.linalg.inv(data.calib[seq_idx]["T_w_lidar"])).dot(lidar_points.T).T
                world_points = target_pose.dot(data.calib[seq_idx]["T_w_lidar"]).dot(lidar_points.T).T # Should be wrong but works


                # World to cameras
                scan_pts_im0 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(np.linalg.inv(pose).dot(world_points.T))[:3, :]).T
                # scan_pts_im1 = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam1"].dot(np.linalg.inv(pose).dot(world_points.T))[:3, :]).T
                scan_pts_im0[:, :2] = scan_pts_im0[:, :2] / scan_pts_im0[:, 2][..., None]
                # scan_pts_im1[:, :2] = scan_pts_im1[:, :2] / scan_pts_im1[:, 2][..., None]

                # check if in bounds of either image
                val_inds = ((scan_pts_im0[:, 0] >= -1) & (scan_pts_im0[:, 1] >= -1)) & ((scan_pts_im0[:, 0] < 1) & (scan_pts_im0[:, 1] < 1))
                # val_inds = val_inds | (((scan_pts_im1[:, 0] >= -1) & (scan_pts_im1[:, 1] >= -1)) & ((scan_pts_im1[:, 0] < 1) & (scan_pts_im1[:, 1] < 1)))
                val_inds = val_inds & ((scan_pts_im0[:, 2] > 0)) # | (scan_pts_im1[:, 2] > 0))

                # check if points are close enough to the camera pose 
                val_inds = val_inds & (np.linalg.norm(world_points[:, :3] - pose[:3, 3], axis=-1) <= 100)
                
                if ignore_moving and next_scan_idx != 0:
                    val_inds = val_inds & (target_label < 250)
                target_scan_data_point = np.zeros([target_scan[val_inds].shape[0], 5])
                target_scan_data_point[:, :3] = world_points[val_inds, :3]
                target_scan_data_point[:, 3] = pose_idx + next_scan_idx
                target_scan_data_point[:, 4] = target_label[val_inds]
                
                if points_in_first_scan == 0:
                    points_in_first_scan = target_scan_data_point.shape[0]
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

            compression_strategy = "patches"
            print("Subsampling..")
            if compression_strategy == "subsample":
                # Subsample the merged scan to save memory
                indices = np.arange(0, merged_scan.shape[0])
                # np.random.shuffle(indices)
                merged_scan = merged_scan[indices[:points_in_first_scan]] # 2048*100]] # Arbitrary to downsize merged_scans
                np.savez_compressed(ms_path, merged_scan.astype(np.float16))
            elif compression_strategy == "patches":
                max_points_per_bin = 64
                sampling = "random_weighted_by_distance"
                
                sampled_merged_scan = merged_scan[:2]
                points = deepcopy(merged_scan)
                points[:, 3] = 1.0
                projected_points = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(np.linalg.inv(pose).dot(points[:, :4].T))[:3, :]).T
                projected_points[:, :2] = projected_points[:, :2] / projected_points[:, 2][..., None]
                assert (projected_points[:, :2].max() < 1) & (projected_points[:, :2].min() > -1)
                binx = np.arange(-1, 1, 2/128)
                biny = np.arange(-1, 1, 2/40)
                statistics = stats.binned_statistic_2d(projected_points[:, 0], projected_points[:, 1], None, 'count', bins=[binx, biny])
                for idx in range(129*41):
                    vals = statistics.binnumber == idx
                    num_vals = np.sum(vals)
                    if num_vals == 0:
                        continue
                    elif num_vals <= max_points_per_bin:
                        sampled_merged_scan = np.concatenate((sampled_merged_scan, merged_scan[vals]), axis=0)
                    elif num_vals > max_points_per_bin:
                        candidates = merged_scan[vals]
                        if sampling == "random":
                            indices = np.arange(0, candidates.shape[0])
                            np.random.shuffle(indices)
                            sampled_merged_scan = np.concatenate((sampled_merged_scan,                              [indices[:max_points_per_bin]]), axis=0)
                        elif sampling == "closest":
                            dist = np.sqrt(np.einsum('ij,ij->i', candidates[:, :3]-pose[:3, 3],  candidates[:, :3]-pose[:3, 3]))
                            indices = np.argsort(dist, axis=-1)[:max_points_per_bin]
                            sampled_merged_scan = np.concatenate((sampled_merged_scan, candidates[indices]), axis=0)
                        elif sampling == "random_weighted_by_distance":
                            indices = np.arange(0, candidates.shape[0])
                            dist = np.sqrt(np.einsum('ij,ij->i', candidates[:, :3]-pose[:3, 3],  candidates[:, :3]-pose[:3, 3]))
                            probs = dist / np.sum(dist)
                            np.random.choice(indices, max_points_per_bin, replace=False, p=probs)
                        else:
                            raise NotImplementedError("Please select a valid sampling method!")
                np.savez_compressed(ms_path, sampled_merged_scan.astype(np.float16))
                # points = deepcopy(sampled_merged_scan)
                # points[:, 3] = 1.0
                # projected_points = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(np.linalg.inv(pose).dot(points[:, :4].T))[:3, :]).T
                # projected_points[:, :2] = projected_points[:, :2] / projected_points[:, 2][..., None]
                # binx = np.arange(-1, 1, 2/128)
                # biny = np.arange(-1, 1, 2/40)
                # statistics = stats.binned_statistic_2d(projected_points[:, 0], projected_points[:, 1], None, 'count', bins=[binx, biny])
                # occurences = np.bincount(statistics.binnumber)
                # from matplotlib import pyplot as plt
                # fig = plt.figure()
                # plt.title("Number of Lidar points projected into the camera image after sampling. (128x40 bins)")
                # occurences = (occurences).reshape([129, 41])[1:, 1:]
                # img = plt.imshow(occurences.T, cmap='viridis')
                # fig.colorbar(img)
                # plt.show()
                # print(statistics)

                    

    print(f"Total time {int(time() - t_start)} sec")


if __name__ == "__main__":
    path = "/Users/nilskeunecke/semantic-kitti_partly"

    # Preprocess Train data
    train_data = KittiSemanticDataset(path, train=True, target_image_size=(370,1226))
    print("Load success")
    preprocess_dataset(train_data, visualize=False, ignore_moving=True)

