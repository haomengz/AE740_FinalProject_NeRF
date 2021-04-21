import torch 
from torch import nn

class NeRF(nn.Module):
    def __init__(self, depth = 8, width = 256, input_dim = 5, output_dim = 4):
        super(NeRF, self).__init__()
        act_fn = nn.ReLU()
        self.layers = act_fn(nn.Linear(input_dim, width)) # Input is x, y, z, theta, phi
        for i in range(depth):
            self.layers = self.layers(act_fn(nn.Linear(width, width)))
        self.layers = self.layers(act_fn(nn.Linear(width, width/2)))
        self.layers = self.layers((nn.Linear(width/2, output_dim))) # Output is R, G, B, theta
        return

    def forward(self, input):
        return self.layers(input)

def main():
    nerf_model = NeRF()
    return True

if __name__ == '__main__':
    main()