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
        modifier = 150
        for pose_idx in tqdm(range(1)):
            pose_idx += modifier
            scan = data[pose_idx]["merged_scan"]
            label = scan[:, 6]
            scan = scan[:, :4]
            lidar_points = deepcopy(scan)
            lidar_points[:, 3] = 1.0
            pcd = o3d.geometry.PointCloud()
            v3d = o3d.utility.Vector3dVector
            val_inds = label >= 0
            val_inds = val_inds #& (label < 250)
            pcd.points = v3d(lidar_points[val_inds, :3])
            pcd.colors = v3d(np.array([data.color_map[x] for x in np.array(label)[val_inds]]))
            pcd_list.append(pcd)

        # Visualize Points in cv2
        imgs = data.load_image_pair(seq_idx, modifier)
        normalized_cam, right_img = imgs[0], imgs[1]
        normal_cam = deepcopy(normalized_cam)
        for idx, pcd in enumerate(pcd_list):
            idx += modifier
            points = np.ones([np.array(pcd.points).shape[0], 4])
            points[:, :3] = np.array(pcd.points)
            colors = np.array(pcd.colors)

            normalized_points = data.calib[seq_idx]["K"].dot(data.calib[seq_idx]["T_w_cam0"].dot(np.linalg.inv(data.poses[seq_idx][idx]).dot(points.T))[:3, :]).T
            normalized_points[:, :2] = normalized_points[:, :2] / normalized_points[:, 2][..., None]

            normalized_val_inds = (normalized_points[:, 0] >= -1) & (normalized_points[:, 1] >= -1)
            normalized_val_inds = normalized_val_inds & ((normalized_points[:, 0] < 1) & (normalized_points[:, 1] < 1))
            normalized_val_inds = normalized_val_inds & (normalized_points[:, 2] > 0)

            for normalized_point, color in zip(normalized_points[normalized_val_inds, :2], colors[normalized_val_inds]):
                bgr_color = np.array([color[2], color[1], color[0]])
                cv2.circle(normalized_cam, (int((normalized_point[0] *.5 + 0.5) * data.target_image_size[1]), int((normalized_point[1] *.5 + 0.5) * data.target_image_size[0])), 1, bgr_color, 1)
                        
            cv2.imshow("Projected LiDAR Scan", normalized_cam)
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

