import numpy as np
from load_llff import load_llff_data

# Load pose

TEST_POSE_PATH = 'test_data'
DOWN_SAMPLE_FACTOR = 8
RECENTER = True
BD_FACTOR = 0.75
SPHERIFY = True

class NeRFPoseLoader():
    def __init__(self, file_path) -> None:
        poses, bds, render_poses, i_test = load_llff_data(file_path, DOWN_SAMPLE_FACTOR,
                                                                  RECENTER, BD_FACTOR, SPHERIFY)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        #print('Loaded llff', images.shape, render_poses.shape, hwf, TEST_DATA_PATH)

        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = np.array([H, W, focal])
        Rotation = poses[:, :, :3]
        Translation = poses[:, :, 3]

        self.rotation = Rotation #(20, 3, 3)
        self.translation = Translation #(20, 3)
        self.hwf = hwf #(3,)
        self.bounds = bds #(20, 2)

        self.height = self.hwf[0]
        self.width = self.hwf[1]
        self.fx = self.hwf[2]
        self.fy = self.hwf[2]

        self.pose_list = []
        for i in range(poses.shape[0]):
            self.pose_list.append((self.rotation[i, :, :], self.translation[i, :], self.bounds[i, :]))
        pass

def main():
    nerf_pose_loader = NeRFPoseLoader(TEST_POSE_PATH)
    test_pose_rotation = nerf_pose_loader.rotation
    test_pose_tranlation = nerf_pose_loader.translation
    test_pose_hwf = nerf_pose_loader.hwf
    test_pose_bounds = nerf_pose_loader.bounds

    print(test_pose_rotation.shape)
    print(test_pose_tranlation)
    print(test_pose_hwf.shape)
    print(test_pose_bounds)
    print(len(nerf_pose_loader.pose_list))
    return True

if __name__ == '__main__':
    main()