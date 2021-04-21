from os import listdir
from os.path import isfile, join
from numpy.core.fromnumeric import transpose
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torch
from pose_loader import NeRFPoseLoader

from run_nerf_helpers import get_embedder

TEST_POSE_PATH = 'test_data'
# TEST_IMG_PATH = 'test_data/test_img'
TEST_IMG_PATH = 'test_data/images_8'
LINESPACE = 40

class NeRFImageLoader(Dataset):
    DIR_FORMAT = "Cartesian" # "Cartesian" or "Original"
    def __init__(self, test_img_path, pose_loader) -> None:

        x_embedder, _ = get_embedder(10)
        d_embedder, _ = get_embedder(4)

        image_file_names = [f for f in listdir(test_img_path) if isfile(join(test_img_path, f))]

        self.rays_and_rgb = []
        for img_name, (R_mat, t_vec, bound), ii in zip(image_file_names, pose_loader.pose_list, range(20)):
            if ii > 2:
                break
            print("Sampling rays from image " + img_name)
            img = cv.imread(join(test_img_path, img_name))

            camera_translation = t_vec
            cam_x, cam_y, cam_z = camera_translation[0], camera_translation[1], camera_translation[2]
            # iterate through rows and columns, every pixel
            for u in range(img.shape[0]): # height
                for v in range(img.shape[1]): # width
                    # March through ray
                    xx = v/pose_loader.fx
                    yy = u/pose_loader.fy
                    z, delta = np.linspace(bound[0], bound[1], num=LINESPACE, endpoint=False, retstep = True)
                    x = xx*z
                    y = yy*z

                    ################## 
                    # Transform x,y,z from camera frame into world frame
                    p_c = np.array([x, y, z])
                    p_w = R_mat.dot(p_c) + camera_translation.reshape((3,1))
                    ##################
                    # if self.DIR_FORMAT == "Original":
                    #     theta = np.arctan2(cam_x-p_w[0], cam_z - p_w[2])
                    #     phi = np.arctan2(cam_y-p_w[1], cam_z-p_w[2])
                    #     ray = np.asarray([p_w[0], p_w[1], p_w[2], theta, phi])
                    # el
                    if self.DIR_FORMAT == "Cartesian":
                        d = camera_translation.reshape((3,1)) - p_w
                        ray = np.asarray([p_w[0], p_w[1], p_w[2], d[0], d[1], d[2]])
                        x_emb = x_embedder(torch.tensor(p_w))
                        d_emb = d_embedder(torch.tensor(d))
                        # print(x_emb.shape)
                        # print(d_emb.shape)
                        x_emb = x_emb.reshape((3, 20, LINESPACE))
                        d_emb = d_emb.reshape((3, 8, LINESPACE))
                        x_emb = x_emb.permute(2,1,0) # 10, 20, 3
                        d_emb = d_emb.permute(2,1,0)

                        x_emb = x_emb.reshape(LINESPACE, -1)
                        d_emb = d_emb.reshape(LINESPACE, -1)

                        # x_emb = x_emb.reshape((-1, LINESPACE))
                        # d_emb = d_emb.reshape((-1, LINESPACE))
                        xd_emb = torch.cat([x_emb, d_emb], 1) # the dim has 10*84
                        # print(xd_emb.shape)
                        # ray_embedded = np.stack([x_emb, d_emb])
                    
                    self.rays_and_rgb.append((ray, xd_emb, img[u, v], delta))
            print("Added " + str(len(self.rays_and_rgb)) + " rays")

    
    def __len__(self):
        return len(self.rays_and_rgb)

    def __getitem__(self, idx):
        # An item should be sampled points (5d) on one ray, and the label of the ray, which is rgb
        sampled_ray = self.rays_and_rgb[idx][0]
        sampled_ray_emb = self.rays_and_rgb[idx][1]
        label = self.rays_and_rgb[idx][2]
        delta = self.rays_and_rgb[idx][3]
        return sampled_ray, sampled_ray_emb, label, delta

    def ndc_rays(H, W, focal, near, rays_o, rays_d):
        """Normalized device coordinate rays.

        Space such that the canvas is a cube with sides [-1, 1] in each axis.

        Args:
        H: int. Height in pixels.
        W: int. Width in pixels.
        focal: float. Focal length of pinhole camera.
        near: float or array of shape[batch_size]. Near depth bound for the scene.
        rays_o: array of shape [batch_size, 3]. Camera origin.
        rays_d: array of shape [batch_size, 3]. Ray direction.

        Returns:
        rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
        rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
        """
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1./(W/(2.*focal)) * \
            (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
        d1 = -1./(H/(2.*focal)) * \
            (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    def get_rays_np(H, W, focal, c2w):
        """Get ray origins, directions from a pinhole camera."""
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d


def main():
    pose_loader = NeRFPoseLoader(TEST_POSE_PATH)
    img_loader = NeRFImageLoader(TEST_IMG_PATH, pose_loader)
    return True

if __name__ == '__main__':
    main()
