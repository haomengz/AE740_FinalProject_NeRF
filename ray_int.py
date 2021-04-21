# functionalities to integrate the rgb and sigma(density) along a ray to get the RGB of pixels

import torch 
import math

def integrate_ray(batch_size, outputs, sigmas, delta):
    num_sampled_points = outputs.shape[1]
    C_out = torch.zeros((batch_size, 3))

    for i in range(batch_size):
        T_exponent = 0
        last_output = torch.zeros((3,))
        for j in range(num_sampled_points):
            T = math.exp(T_exponent)
            C_out[i] = last_output.to("cuda") + T * (1 - math.exp( - sigmas[i][j] * delta[i])) * outputs[i][j][:]
            last_output = C_out[i]
            T_exponent -= sigmas[i][j] * delta[i]

    return C_out