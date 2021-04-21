from torch import tensor
from pose_loader import NeRFPoseLoader
from image_loader import NeRFImageLoader
from model import NeRF
from model_with_skip import NeRF_Res
from ray_int import integrate_ray

import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from os import path


BATCH_SIZE = 1

TEST_POSE_PATH = 'test_data'
TEST_IMG_PATH = 'test_data/test_img'
MODEL_PATH = 'model_trained/test1/50.pth"'

nerf_loader = NeRFPoseLoader(TEST_POSE_PATH)
img_to_ray = NeRFImageLoader(TEST_IMG_PATH, nerf_loader)

img_loader = torch.utils.data.DataLoader(
    img_to_ray, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

nerf_model = NeRF_Res().to('cuda')
if not os.path.isfile(MODEL_PATH):
    raise ValueError('{} does not exist. Please provide a valid path for pretrained model!'.format(MODEL_PATH))
nerf_model.load_state_dict(torch.load(MODEL_PATH))
print('Load model successfully from: {}'.format(MODEL_PATH))

nerf_model.eval()
with torch.no_grad():
    for i, data in enumerate(img_loader, 0):
        ray, ray_emb, label, delta = data
        ray, ray_emb, label, delta = ray.to('cuda'), ray_emb.to('cuda'), label.to('cuda'), delta.to('cuda')
        pred_rgb, pred_sigma = nerf_model(ray_emb.float().to('cuda'))
        C_out = integrate_ray(BATCH_SIZE, outputs.to("cuda"), sigmas.to("cuda"), delta.to("cuda"))
        # pred = pred.cpu().detach().numpy()[0]
        # np.save(os.path.join(save_dir, 'pred_{}'.format(filename)), pred)

print('Done.')



optimizer = optim.SGD(nerf_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(img_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        ray, ray_emb, label, delta = data
        # print(ray_emb.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # ray_emb = ray_emb.type(torch.DoubleTensor)
        # print(ray_emb.dtype)
        outputs, sigmas = nerf_model(ray_emb.float().to("cuda"))
        # print(outputs.shape)
        # print(sigmas.shape)

        # Integration factory

        ##############################
        #  integrate along the ray
        #  outputs (BATCH_SIZE, number of sampled points of a ray, 4)
        num_sampled_points = outputs.shape[1]
        # C_out (BATCH_SIZE, 3)
        C_out = integrate_ray(BATCH_SIZE, outputs.to(
            "cuda"), sigmas.to("cuda"), delta.to("cuda"))

        ##############################

        ##############################
        #  define loss function
        criterion = nn.MSELoss()

        loss = criterion(C_out, label.float())
        ##############################
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    if epoch % 50 == 0:
        torch.save(nerf_model.state_dict(), path.join("model_trained/test1", str(epoch)+".pth"))

# def main():

#     return True

# if __name__ == '__main__':
#     main()
