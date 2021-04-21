import torch 
from torch import nn

# Input is encoded locations and directions
# Output is the RGB and densities of the sample points

class NeRF_Res(nn.Module):
    def __init__(self, depth = 8, width = 256, input_dim = 60, output_dim = 3):
        super(NeRF_Res, self).__init__()

        self.firstFivelayers = nn.Sequential(nn.Linear(60, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU()
                                            )

        self.fourLayers = nn.Sequential(nn.Linear(60 + width, width),
                                        nn.ReLU(),
                                        nn.Linear(width, width),
                                        nn.ReLU(),
                                        nn.Linear(width, width),
                                        nn.ReLU(),
                                        nn.Linear(width, width),
                                        nn.ReLU(),
                                        nn.Linear(width, width + 1)
                                        )

        self.lastTwoLayers = nn.Sequential(nn.Linear(int(24 + width), int(width / 2)),
                                            nn.ReLU(),
                                            nn.Linear(int(width / 2), output_dim),
                                            nn.Sigmoid()
                                            )   

        self.act_fn =  nn.ReLU()                       
        return

    def forward(self, input):
        output = self.firstFivelayers(input[:, :, :60])
        output = torch.cat([input[:, :, :60], output], dim = -1)
        output = self.fourLayers(output)
        sigma = self.act_fn(output[:, :, 0])
        output = output[:, :, 1:]

        output = torch.cat([output, input[:, :, 60:]], dim = -1)
        output = self.lastTwoLayers(output)
        return output, sigma

def main():
    nerf_model = NeRF_Res()
    return True

if __name__ == '__main__':
    main()