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
import csv

BATCH_SIZE = 64
TEST_POSE_PATH = 'test_data'
TEST_IMG_PATH = 'test_data/images_8'
nerf_loader = NeRFPoseLoader(TEST_POSE_PATH)
img_to_ray = NeRFImageLoader(TEST_IMG_PATH, nerf_loader)

img_loader = torch.utils.data.DataLoader(
    img_to_ray, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

nerf_model = NeRF_Res().to("cuda")

optimizer = optim.SGD(nerf_model.parameters(), lr=0.001, momentum=0.9)
nerf_model.train()

for epoch in range(300):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(img_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        ray, ray_emb, label, delta = data
        # print(ray_emb.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs, sigmas = nerf_model(ray_emb.float().to("cuda"))

        ##############################
        #  integrate along the ray

        #  outputs (BATCH_SIZE, number of sampled points of a ray, 4)
        num_sampled_points = outputs.shape[1]
        # C_out (BATCH_SIZE, 3)
        C_out = integrate_ray(BATCH_SIZE, outputs.to(
            "cuda"), sigmas.to("cuda"), delta.to("cuda"))

        ##############################

        ##############################
        # define loss function

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
    if epoch % 5 == 0:
        torch.save(nerf_model.state_dict(), path.join("model_trained/test1", str(epoch)+".pth"))
        with open('loss.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(str(running_loss))

# def main():

#     return True

# if __name__ == '__main__':
#     main()
