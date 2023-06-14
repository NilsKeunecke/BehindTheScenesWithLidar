from __future__ import annotations
import open3d as o3d
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from kitti_semantic_dataset import KittiSemanticDataset
import cv2

def visualize_scans(data) -> None:
    for seq_idx, seq in enumerate(data.sequences):
        pcd_list = []
        for pose_idx in tqdm(range(1)):
            pose_idx += 50
            scan, label, pose = data.load_pointcloud(seq_idx, pose_idx), data.labels[seq_idx][pose_idx], data.poses[seq_idx][pose_idx]
            scan = data[pose_idx]["merged_scan"]
            label = scan[:, 6]
            scan = scan[:, :4]
            world_points = deepcopy(scan)
            world_points[:, 3] = 1.0
            pcd = o3d.geometry.PointCloud()
            v3d = o3d.utility.Vector3dVector
            val_inds = label >= 0
            val_inds = val_inds #& (label < 250)
            pcd.points = v3d(world_points[val_inds, :3])
            pcd.colors = v3d(np.array([data.color_map[x] for x in np.array(label)[val_inds]]))
            pcd_list.append(pcd)

        # Visualize Points in cv2
        imgs = data.load_image_pair(seq_idx, 50)
        normalized_cam, right_img = imgs[0], imgs[1]
        normal_cam = deepcopy(normalized_cam)
        for idx, pcd in enumerate(pcd_list):
            idx += 50
            points = np.ones([np.array(pcd.points).shape[0], 4])
            points[:, :3] = np.array(pcd.points)
            colors = np.array(pcd.colors)
            new_K = np.array([[707.0912,   0.,     601.8873],
                                [  0.,     707.0912, 183.1104],
                                [  0.,       0.,       1.    ]])
            normal_points = new_K.dot(data.calib[seq_idx]["T_w_cam0"].dot(np.linalg.inv(data.poses[seq_idx][idx]).dot(points.T))[:3, :]).T
            normalized_points = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(np.linalg.inv(data.poses[seq_idx][idx]).dot(points.T))[:3, :]).T

            normal_points[:, :2] = normal_points[:, :2] / normal_points[:, 2][..., None]
            normalized_points[:, :2] = normalized_points[:, :2] / normalized_points[:, 2][..., None]

            normalized_val_inds = (normalized_points[:, 0] >= -1) & (normalized_points[:, 1] >= -1)
            normalized_val_inds = normalized_val_inds & ((normalized_points[:, 0] < 1) & (normalized_points[:, 1] < 1))
            normalized_val_inds = normalized_val_inds & (normalized_points[:, 2] > 0)

            normal_val_inds = (normal_points[:, 0] >= 0) & (normal_points[:, 1] >= 0)
            normal_val_inds = normal_val_inds & ((normal_points[:, 0] < data.target_image_size[1]) & (normal_points[:, 1] < data.target_image_size[0]))
            normal_val_inds = normal_val_inds & (normal_points[:, 2] > 0)

            print(np.sum(normal_val_inds.astype(np.float32) - normalized_val_inds.astype(np.float32)))

            for normalized_point, color in zip(normalized_points[normalized_val_inds, :2], colors[normalized_val_inds]):
                bgr_color = np.array([color[2], color[1], color[0]])
                cv2.circle(normalized_cam, (int((normalized_point[0] *.5 + 0.5) * data.target_image_size[1]), int((normalized_point[1] *.5 + 0.5) * data.target_image_size[0])), 1, bgr_color, 1)
            
            for normal_point, color in zip(normal_points[normal_val_inds, :2], colors[normal_val_inds]):
                bgr_color = np.array([color[2], color[1], color[0]])
                cv2.circle(normal_cam, (int(normal_point[0]), int(normal_point[1])), 1, bgr_color, 1)
            
            cv2.imshow("Projected LiDAR Scan", np.vstack([normalized_cam, normal_cam]))
            cv2.waitKey(0)


        # Visualize Pcd in Open3d
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        for pcd in pcd_list:
            vis.add_geometry(pcd)
        vis.run()

if __name__ == "__main__":
    path = "/Users/nilskeunecke/semantic-kitti_partly"

    # Preprocess Train data
    train_data = KittiSemanticDataset(path, train=True, target_image_size=(370,1226))
    visualize_scans(train_data)

